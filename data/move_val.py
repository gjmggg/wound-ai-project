import os
import shutil

# 设置路径
train_dir = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/Labeled/Original/Palette/Train"
val_dir   = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/Labeled/Original/Palette/Val"

# 验证集编号列表
val_ids = [
    "0944", "0945", "0948", "0951", "0956", "0964", "0954", "1001",
    "0939", "1008", "0975", "0931", "0940", "0947", "1009", "0998"
]

# 创建 val 目录
os.makedirs(val_dir, exist_ok=True)

# 移动文件
for fid in val_ids:
    src = os.path.join(train_dir, f"{fid}.png")
    dst = os.path.join(val_dir, f"{fid}.png")
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")
    else:
        print(f"Warning: {src} not found")

print("Done.")#ai辅助生成：Deepseek 2026-4-03