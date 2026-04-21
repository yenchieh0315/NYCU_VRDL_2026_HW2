import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import math
from tqdm import tqdm

weight_path = "cond_detr_epoch_40.pth"

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
        
        aux_outputs =[]
        for layer in self.decoder:
            tgt = layer(tgt, memory, query_pos, pos)
            aux_outputs.append({'pred_logits': self.class_embed(tgt), 'pred_boxes': self.bbox_embed(tgt)})
        
        out = aux_outputs[-1].copy()
        if self.training: out['aux_outputs'] = aux_outputs[:-1]
        return out

def main():
    TEST_IMG_DIR = "./dataset/test"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ConditionalDETR().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    
    transform_test = A.Compose([
        A.Resize(640, 640),
        ToTensorV2()
    ])
    
    predictions = []
    test_files =[f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Predict {len(test_files)} picture.")
    
    with torch.no_grad():
        for file_name in tqdm(test_files, desc="Testing"):
            img_path = os.path.join(TEST_IMG_DIR, file_name)
            img_id = int(os.path.splitext(file_name)[0])
            
            image = cv2.imread(img_path)
            orig_h, orig_w = image.shape[:2] 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transformed = transform_test(image=image)
            img_tensor = (transformed['image'] / 255.0).unsqueeze(0).to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(img_tensor)
                
            scores = outputs['pred_logits'].sigmoid()[0].float()
            boxes = outputs['pred_boxes'][0].float()
            
            max_scores, labels = scores.max(dim=-1)
            
            topk_scores, topk_indices = torch.topk(max_scores, min(6, len(max_scores)))
            keep_indices = topk_indices[topk_scores > 0.01]
            if len(keep_indices) == 0:
                keep_indices = topk_indices[:1]
            
            keep_boxes = boxes[keep_indices]
            keep_scores = max_scores[keep_indices]
            keep_labels = labels[keep_indices]
            
            for box, score, label in zip(keep_boxes, keep_scores, keep_labels):
                cx, cy, w, h = box.cpu().numpy()
                abs_w = w * orig_w
                abs_h = h * orig_h
                abs_x = (cx * orig_w) - (abs_w / 2)
                abs_y = (cy * orig_h) - (abs_h / 2)
                
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label.item()) + 1, 
                    "bbox":[float(abs_x), float(abs_y), float(abs_w), float(abs_h)],
                    "score": float(score.item())
                })
                
    with open("pred.json", "w") as f:
        json.dump(predictions, f, indent=4)
        
    print(f"Prediction complete. Detect {len(predictions)} Bounding Boxes.")

if __name__ == "__main__":
    main()