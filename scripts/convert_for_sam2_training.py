import os
import json
import base64

#脚本由AI辅助生成用于转换成SAM2所需的格式：豆包 2026-3-22

# 配置路径
INPUT_IMG_DIR = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/SA1B_Custom_Format/images"
INPUT_ANNO_DIR = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/SA1B_Custom_Format/labels"
OUTPUT_ANNO_DIR = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/SA1B_Official_Format/labels"

# 创建输出目录
os.makedirs(OUTPUT_ANNO_DIR, exist_ok=True)

# 遍历所有标注文件
for anno_file in os.listdir(INPUT_ANNO_DIR):
    if not anno_file.endswith(".json"):
        continue
    
    # 读取自定义标注
    anno_path = os.path.join(INPUT_ANNO_DIR, anno_file)
    with open(anno_path, "r", encoding="utf-8") as f:
        custom_anno = json.load(f)
    
    # 补充官方字段
    basename = os.path.splitext(anno_file)[0]
    img_file = f"{basename}.png"
    img_path = os.path.join(INPUT_IMG_DIR, img_file)
    
    # 构建官方格式标注
    official_anno = {
        "image": {
            "image_id": basename,
            "width": custom_anno["image"]["width"],
            "height": custom_anno["image"]["height"],
            "file_name": img_file
        },
        "annotations": []
    }
    
    # 处理每个掩码标注
    for idx, ann in enumerate(custom_anno["annotations"]):
        # 补充缺失字段
        seg = ann["segmentation"]
        seg["encoding"] = "rle"  # 必须添加
        
        # 计算掩码面积（简化版）
        area = 0
        if "counts" in seg:
            # 简易RLE面积计算（仅示例）
            counts = seg["counts"].encode('latin-1') if isinstance(seg["counts"], str) else seg["counts"]
            area = sum(counts[i] for i in range(1, len(counts), 2))
        
        official_ann = {
            "id": idx,
            "segmentation": seg,
            "area": area,
            "bbox": [0, 0, custom_anno["image"]["width"], custom_anno["image"]["height"]],  # 简化包围盒
            "predicted_iou": 0.9,
            "point_coords": [[custom_anno["image"]["width"]//2, custom_anno["image"]["height"]//2]],
            "stability_score": 0.95,
            "crop_box": [0, 0, custom_anno["image"]["width"], custom_anno["image"]["height"]],
            "crop_box_area": custom_anno["image"]["width"] * custom_anno["image"]["height"]
        }
        official_anno["annotations"].append(official_ann)
    
    # 保存官方格式标注
    output_anno_path = os.path.join(OUTPUT_ANNO_DIR, anno_file)
    with open(output_anno_path, "w", encoding="utf-8") as f:
        json.dump(official_anno, f, indent=2)

print(f"✅ 转换完成！共处理 {len(os.listdir(INPUT_ANNO_DIR))} 个标注文件")
print(f"输出路径: {OUTPUT_ANNO_DIR}")