import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 本地路径
local_repo = "/root/autodl-tmp/wound_ai_project/dinov2"
weight_path = "/root/autodl-tmp/wound_ai_project/models/dinov2_vitl14_pretrain.pth"

# 加载模型结构（不加载预训练权重）
model = torch.hub.load(local_repo, 'dinov2_vitl14', source='local', pretrained=False)
# 加载本地权重
state_dict = torch.load(weight_path, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# DINOv2 预处理（Large 模型同样使用 518x518）
transform = T.Compose([
    T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(518),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def extract_features_from_folder(folder_path, output_npy):
    """提取文件夹内所有图片的 DINOv2 特征，保存为 .npy"""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png','.jpg','.jpeg'))]
    image_files.sort()
    features = []
    for file in tqdm(image_files, desc=f"Extracting {folder_path}"):
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().flatten()  # 维度应为 1024
        features.append(feat)
    features = np.array(features)
    np.save(output_npy, features)
    print(f"Saved {output_npy}, shape {features.shape}")

if __name__ == "__main__":
    train_crop_dir = "/root/autodl-tmp/wound_ai_project/data/crop_data/train"
    val_crop_dir   = "/root/autodl-tmp/wound_ai_project/data/crop_data/val"
    
    extract_features_from_folder(train_crop_dir, "dino_train.npy")
    extract_features_from_folder(val_crop_dir,   "dino_val.npy") #ai辅助生成：Deepseek 2026-4-3