"""
融合 DINOv2 特征 + DFUTissueSegNet 组织占比 训练严重程度分类器
假设：
- 已有 dino_train.npy, dino_val.npy (DINOv2 特征)
- 已有 y_train.npy, y_val.npy (基于 Wagner 规则的严重程度标签)
- 已有训练好的 DFUTissueSegNet 模型权重 (best_model.pth)
"""
#由ai辅助生成：Deepseek 2026-4-11
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ==================== 配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 路径
unet_weight = "/root/autodl-tmp/wound_ai_project/models/best_model.pth"  # 你的 DFUTissueSegNet 权重
train_img_dir = "/root/autodl-tmp/wound_ai_project/data/crop_data/train"   # 训练集伤口小图
val_img_dir   = "/root/autodl-tmp/wound_ai_project/data/crop_data/val"     # 验证集伤口小图

dino_train = np.load("./label/dino_train.npy")   # (N_train, 1024)
dino_val   = np.load("./label/dino_val.npy")     # (N_val, 1024)
y_train    = np.load("./label/y_train.npy")
y_val      = np.load("./label/y_val.npy")

# 输出文件
ratios_train_out = "ratios_train_DFUTissueSegNet.npy"
ratios_val_out   = "ratios_val_DFUTissueSegNet.npy"
classifier_out   = "fusion_classifier.pkl"
scaler_out       = "fusion_scaler.pkl"

# ==================== 加载 DFUTissueSegNet 模型 ====================
def load_unet(weight_path):
    model = Unet(
        encoder_name='mit_b3',
        encoder_weights=None,
        classes=4,
        activation=None,
    )
    state = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

model = load_unet(unet_weight)
print("DFUTissueSegNet model loaded.")

# ==================== 预处理函数（与训练时一致） ====================
preprocessing_fn = get_preprocessing_fn('mit_b3', 'imagenet')

def preprocess_image(img_bgr):
    """img_bgr: BGR image (H,W,3), return tensor (1,3,256,256)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
    img_norm = preprocessing_fn(img_resized)
    img_tensor = torch.from_numpy(img_norm).float().permute(2,0,1).unsqueeze(0)
    return img_tensor.to(DEVICE)

def predict_mask(img_bgr):
    with torch.no_grad():
        logits = model(preprocess_image(img_bgr))
        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (256,256)
    return mask

def compute_ratios(mask):
    """mask: 0背景, 1纤维蛋白, 2肉芽, 3焦痂"""
    counts = np.bincount(mask.flatten(), minlength=4)
    fibrin = counts[1]   # 纤维蛋白
    gran   = counts[2]   # 肉芽
    callus = counts[3]   # 焦痂
    total = fibrin + gran + callus
    if total == 0:
        return 0.0, 0.0, 0.0
    return fibrin/total, gran/total, callus/total

# ==================== 计算组织占比 ====================
def compute_ratios_for_folder(img_dir, output_file):
    if os.path.exists(output_file):
        print(f"Loading existing ratios from {output_file}")
        return np.load(output_file)
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    ratios = []
    for fname in tqdm(img_files, desc=f"Processing {os.path.basename(img_dir)}"):
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}, skip")
            ratios.append([0.0,0.0,0.0])
            continue
        mask = predict_mask(img)
        fibr, gran, call = compute_ratios(mask)
        ratios.append([fibr, gran, call])
    ratios = np.array(ratios, dtype=np.float32)
    np.save(output_file, ratios)
    print(f"Saved ratios to {output_file}, shape {ratios.shape}")
    return ratios

print("Computing ratios for training set...")
ratios_train = compute_ratios_for_folder(train_img_dir, ratios_train_out)
print("Computing ratios for validation set...")
ratios_val   = compute_ratios_for_folder(val_img_dir, ratios_val_out)

# ==================== 拼接特征 ====================
assert dino_train.shape[0] == ratios_train.shape[0]
assert dino_val.shape[0] == ratios_val.shape[0]
X_train = np.concatenate([dino_train, ratios_train], axis=1)   # (N, 1024+3)
X_val   = np.concatenate([dino_val, ratios_val], axis=1)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==================== 训练逻辑回归 ====================
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train_scaled, y_train)

# ==================== 评估 ====================
y_pred = clf.predict(X_val_scaled)
acc = accuracy_score(y_val, y_pred)
print(f"\n验证集准确率: {acc:.4f}")
print("混淆矩阵:")
print(confusion_matrix(y_val, y_pred))
print("分类报告:")
target_names = ['轻度', '中度', '重度', '溃烂']
print(classification_report(y_val, y_pred, target_names=target_names, zero_division=0))

# ==================== 保存模型 ====================
joblib.dump(clf, classifier_out)
joblib.dump(scaler, scaler_out)
print(f"分类器已保存为 {classifier_out} 和 {scaler_out}")