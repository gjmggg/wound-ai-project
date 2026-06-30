"""
根据 Wagner 简化规则，从原始三色掩码生成严重程度标签 (0=轻度,1=中度,2=重度,3=溃烂/危重)
规则优先级：危重 > 重度 > 轻度 > 中度
- 危重 (3)：焦痂（蓝色）占比 > 50%
- 重度 (2)：焦痂占比 > 20% (且不满足危重)
- 轻度 (0)：肉芽（绿色）占比 > 60% (且不满足上述)
- 中度 (1)：纤维蛋白（红色）占比 > 40% (且不满足上述)
- 其余情况归为中度 (1)
"""

#脚本由ai辅助生成：Deepseek，2026-4-5

import numpy as np
import cv2
import os
from tqdm import tqdm

# ==================== 配置路径 ====================
# 请根据你的实际目录修改
train_mask_dir = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/Labeled/Original/Palette/Train"  # 训练集三色掩码
val_mask_dir   = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/Labeled/Original/Palette/Val"     # 验证集三色掩码

# 输出文件路径
output_train = "./label/y_train.npy"
output_val   = "./label/y_val.npy"

# ==================== 颜色映射（BGR 格式） ====================
# 根据你的描述：红色=纤维蛋白，绿色=肉芽，蓝色=焦痂
# OpenCV 读取为 BGR，因此：
# 红色 (0,0,255) -> 纤维蛋白
# 绿色 (0,255,0) -> 肉芽
# 蓝色 (255,0,0) -> 焦痂
COLOR_FIBRIN = (0, 0, 255)   # BGR 红色
COLOR_GRAN   = (0, 255, 0)   # BGR 绿色
COLOR_CALLUS = (255, 0, 0)   # BGR 蓝色

def color_to_label_mask(color_mask):
    """
    将彩色三色掩码转换为单通道标签图
    返回: label (H, W), 值: 0背景, 1纤维蛋白, 2肉芽, 3焦痂
    """
    h, w = color_mask.shape[:2]
    label = np.zeros((h, w), dtype=np.uint8)
    
    # 红色 -> 纤维蛋白 (1)
    red_mask = (color_mask[:,:,0] == COLOR_FIBRIN[0]) & (color_mask[:,:,1] == COLOR_FIBRIN[1]) & (color_mask[:,:,2] == COLOR_FIBRIN[2])
    label[red_mask] = 1
    
    # 绿色 -> 肉芽 (2)
    green_mask = (color_mask[:,:,0] == COLOR_GRAN[0]) & (color_mask[:,:,1] == COLOR_GRAN[1]) & (color_mask[:,:,2] == COLOR_GRAN[2])
    label[green_mask] = 2
    
    # 蓝色 -> 焦痂 (3)
    blue_mask = (color_mask[:,:,0] == COLOR_CALLUS[0]) & (color_mask[:,:,1] == COLOR_CALLUS[1]) & (color_mask[:,:,2] == COLOR_CALLUS[2])
    label[blue_mask] = 3
    
    return label

def compute_ratios(label_mask):
    """从单通道标签图计算三种组织的占比"""
    counts = np.bincount(label_mask.flatten(), minlength=4)
    fibrin = counts[1]   # 纤维蛋白
    gran   = counts[2]   # 肉芽
    callus = counts[3]   # 焦痂
    total = fibrin + gran + callus
    if total == 0:
        return 0.0, 0.0, 0.0
    return fibrin/total, gran/total, callus/total

def wagner_rule_to_severity(fibrin_ratio, gran_ratio, callus_ratio):
    """
    应用 Wagner 简化规则判定严重程度
    优先级：危重 > 重度 > 轻度 > 中度
    返回: 0=轻度, 1=中度, 2=重度, 3=溃烂/危重
    """
    # 危重：焦痂 > 50%
    if callus_ratio > 0.5:
        return 3
    # 重度：焦痂 > 20%（且不满足危重）
    if callus_ratio > 0.2:
        return 2
    # 轻度：肉芽 > 60%
    if gran_ratio > 0.6:
        return 0
    # 中度：纤维蛋白 > 40%
    if fibrin_ratio > 0.4:
        return 1
    # 其余归为中度
    return 1

def process_folder(mask_dir, output_path):
    """处理整个文件夹，生成标签数组"""
    if not os.path.exists(mask_dir):
        print(f"目录不存在: {mask_dir}")
        return
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    mask_files.sort()
    labels = []
    for fname in tqdm(mask_files, desc=f"处理 {os.path.basename(mask_dir)}"):
        mask_path = os.path.join(mask_dir, fname)
        color_mask = cv2.imread(mask_path)
        if color_mask is None:
            print(f"无法读取 {mask_path}，跳过")
            continue
        label_mask = color_to_label_mask(color_mask)
        fibrin, gran, callus = compute_ratios(label_mask)
        severity = wagner_rule_to_severity(fibrin, gran, callus)
        labels.append(severity)
    labels = np.array(labels, dtype=np.int32)
    np.save(output_path, labels)
    print(f"已保存 {output_path}, 样本数 {len(labels)}")

if __name__ == "__main__":
    process_folder(train_mask_dir, output_train)
    process_folder(val_mask_dir, output_val)
    print("完成！")