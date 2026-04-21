import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss, generalized_box_iou, box_convert
from scipy.optimize import linear_sum_assignment
import cv2
import math
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

class SVHNDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
        self.images = {img['id']: img for img in self.coco['images']}
        self.img_to_anns = {img['id']: [] for img in self.coco['images']}
        for ann in self.coco['annotations']:
            self.img_to_anns[ann['image_id']].append(ann)
        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        anns = self.img_to_anns.get(img_id, [])
        bboxes, class_labels = [], []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            
            x_max = min(orig_w, max(0, x + bw))
            y_max = min(orig_h, max(0, y + bh))
            x_min = min(orig_w, max(0, x))
            y_min = min(orig_h, max(0, y))
            new_bw = x_max - x_min
            new_bh = y_max - y_min
            
            if new_bw > 1 and new_bh > 1:
                bboxes.append([x_min, y_min, new_bw, new_bh])
                class_labels.append(ann['category_id'] - 1)
        
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes, class_labels = transformed['image'], transformed['bboxes'], transformed['class_labels']
            
        _, h, w = image.shape
        target_bboxes = [[(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h] for x, y, bw, bh in bboxes]
        return image, {
            'boxes': torch.tensor(target_bboxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor(class_labels, dtype=torch.int64),
            'orig_size': torch.tensor([orig_h, orig_w])
        }

def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]

def gen_sineembed_for_position(pos_tensor, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    
    sin_x = pos_tensor[..., 0:1] * scale / dim_t
    sin_y = pos_tensor[..., 1:2] * scale / dim_t
    sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=-1).flatten(-2)
    sin_y = torch.stack((sin_y[..., 0::2].sin(), sin_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((sin_y, sin_x), dim=-1)

class ConditionalDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, 1024), nn.ReLU(), nn.Dropout(0.1), nn.Linear(1024, d_model))
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, query_pos, pos):
        q = k = tgt + query_pos
        tgt = self.norms[0](tgt + self.dropout(self.self_attn(q, k, tgt)[0]))
        
        tgt = self.norms[1](tgt + self.dropout(self.cross_attn(tgt + query_pos, memory + pos, memory)[0]))
        tgt = self.norms[2](tgt + self.dropout(self.ffn(tgt)))
        return tgt

class ConditionalDETR(nn.Module):
    def __init__(self, num_classes=10, num_queries=100, hidden_dim=256):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.conv = nn.Conv2d(1024, hidden_dim, 1)
        
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, 1024, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.decoder = nn.ModuleList([ConditionalDecoderLayer(hidden_dim, 8) for _ in range(6)])
        
        self.refpoint_embed = nn.Embedding(num_queries, 2) 
        self.query_content_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4), nn.Sigmoid())
        
    def forward(self, x):
        feat = self.backbone(x)
        h = self.conv(feat)
        
        b, c, hh, ww = h.shape
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, hh, device=x.device), torch.linspace(0, 1, ww, device=x.device), indexing='ij')
        pos = gen_sineembed_for_position(torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(b,1,1,1)).flatten(1, 2)
        
        memory = self.encoder(h.flatten(2).permute(0, 2, 1) + pos)
        
        ref_pts = self.refpoint_embed.weight.sigmoid().unsqueeze(0).repeat(b, 1, 1)
        query_pos = gen_sineembed_for_position(ref_pts)
        tgt = self.query_content_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        
        aux_outputs = []
        for layer in self.decoder:
            tgt = layer(tgt, memory, query_pos, pos)
            aux_outputs.append({'pred_logits': self.class_embed(tgt), 'pred_boxes': self.bbox_embed(tgt)})
        
        out = aux_outputs[-1].copy()
        if self.training: out['aux_outputs'] = aux_outputs[:-1]
        return out

class HungarianMatcherAndLoss(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def match(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].sigmoid().flatten(0, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        if len(tgt_ids) == 0: 
            return [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)) for _ in range(bs)]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, 'cxcywh', 'xyxy'), box_convert(tgt_bbox, 'cxcywh', 'xyxy'))
        
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _compute_loss(self, outputs, targets):
        indices = self.match(outputs, targets)
        
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], 10, dtype=torch.int64, device=src_logits.device)
        
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) > 0:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes[idx] = target_classes_o
            
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], 11], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, alpha=0.25, gamma=2.0, reduction="sum") / max(1, len(idx[0]))

        if len(idx[0]) == 0: return loss_ce
        
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / len(idx[0])
        loss_giou = (1 - torch.diag(generalized_box_iou(box_convert(src_boxes, 'cxcywh', 'xyxy'), box_convert(target_boxes, 'cxcywh', 'xyxy')))).sum() / len(idx[0])

        return loss_ce + 5.0 * loss_bbox + 2.0 * loss_giou

    def forward(self, outputs, targets):
        loss = self._compute_loss(outputs, targets)
        if 'aux_outputs' in outputs:
            for aux_outputs in outputs['aux_outputs']:
                loss += self._compute_loss(aux_outputs, targets)
        return loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    IMG_SIZE = 640
    BATCH_SIZE = 12
    ACCUMULATION_STEPS = 4 
    LR = 1e-4
    EPOCHS = 40

    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

    dataset_train = SVHNDataset("./dataset/train", "./dataset/train.json", transform)
    loader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    dataset_valid = SVHNDataset("./dataset/valid", "./dataset/valid.json", transform)
    loader_valid = DataLoader(dataset_valid, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    model = ConditionalDETR().to(device)
    
    params = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": LR * 0.1}
    ]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    criterion = HungarianMatcherAndLoss().to(device) 
    scheduler = MultiStepLR(optimizer, milestones=[20, 25], gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0  
        optimizer.zero_grad()
        
        pbar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for i, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(device) / 255.0
            targets =[{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss = loss / ACCUMULATION_STEPS
            
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(loader_train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
            
        avg_train_loss = epoch_train_loss / len(loader_train)

        model.eval()
        epoch_val_loss = 0  
        
        pbar_val = tqdm(loader_valid, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]")
        with torch.no_grad():
            for imgs, targets in pbar_val:
                imgs = imgs.to(device) / 255.0
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(imgs)
                    val_loss = criterion(outputs, targets)
                
                epoch_val_loss += val_loss.item()
                pbar_val.set_postfix(val_loss=val_loss.item())
                
        avg_val_loss = epoch_val_loss / len(loader_valid)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "detr_svhn_best.pth")
            print(f"New record (Valid Loss: {best_val_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"cond_detr_epoch_{epoch+1}.pth")
         
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")    

if __name__ == "__main__":
    main()