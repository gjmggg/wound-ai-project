import os
import json
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
#脚本由AI辅助生成用于转换成SAM2输入所需的 SA-1B 格式：豆包 2026-3-22

def png_to_rle(mask: np.ndarray) -> dict:
    """将PNG掩码转换为SA-1B格式的RLE编码（保持二进制字符串格式）"""
    # 确保掩码是二值化的（0=背景，1=前景）
    mask = (mask > 0).astype(np.uint8)
    # 转换为COCO RLE格式（counts为二进制bytes）
    rle = mask_utils.encode(np.asfortranarray(mask))
    # 转换size为Python原生int（避免numpy类型序列化报错）
    rle["size"] = [int(s) for s in rle["size"]]
    return rle

def convert_to_sa1b_custom(
    img_dir: str,
    mask_dir: str,
    output_dir: str,
    img_suffix: str = ".jpg",
    mask_suffix: str = ".png",
    start_image_id: int = 1  # 图片ID起始值
):
    """
    转换为自定义SA-1B格式（JSON文件名与图片名完全一致）：
    {
        "image": {"image_id": xx, "width": xx, "height": xx, "file_name": xx},
        "annotations": [{"area": xx, "segmentation": {...}}]
    }
    """
    # 创建输出目录
    sa1b_img_dir = os.path.join(output_dir, "images")
    sa1b_anno_dir = os.path.join(output_dir, "annotations")
    os.makedirs(sa1b_img_dir, exist_ok=True)
    os.makedirs(sa1b_anno_dir, exist_ok=True)

    # 遍历所有图片并按文件名排序
    img_filenames = sorted([f for f in os.listdir(img_dir) if f.endswith(img_suffix)])
    current_image_id = start_image_id

    for img_filename in tqdm(img_filenames, desc="转换为自定义SA-1B格式"):
        # 1. 提取图片basename（核心：保证JSON文件名与图片名一致）
        basename = os.path.splitext(img_filename)[0]  # 如 "001.jpg" → "001"
        json_filename = f"{basename}.json"            # 生成 "001.json"

        # 2. 匹配对应掩码文件
        mask_filename = basename + mask_suffix
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            print(f"跳过：{img_filename} 无对应掩码 {mask_filename}")
            continue

        # 3. 加载图片和掩码
        img_path = os.path.join(img_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        height, width = img_np.shape[0], img_np.shape[1]

        # 4. 过滤空掩码
        if np.sum(mask_np) == 0:
            print(f"跳过：{img_filename} 掩码为空")
            continue

        # 5. 构建目标格式的JSON标注
        sa1b_anno = {
            "image": {
                "image_id": current_image_id,  # 连续数字ID（可改为int(basename)如果basename是数字）
                "width": int(width),           # Python原生int
                "height": int(height),         # Python原生int
                "file_name": img_filename      # 原始图片名（如001.jpg）
            },
            "annotations": []
        }

        # 6. 处理每个目标（按灰度值区分多目标）
        unique_values = np.unique(mask_np)
        unique_values = [v for v in unique_values if v != 0]  # 排除背景

        for value in unique_values:
            obj_mask = (mask_np == value).astype(np.uint8)
            rle = png_to_rle(obj_mask)
            
            # 计算目标面积（Python原生int）
            area = int(np.sum(obj_mask))

            # 添加标注到列表
            sa1b_anno["annotations"].append({
                "area": area,
                "segmentation": rle  # 包含二进制counts的RLE
            })

        # 7. 保存图片和JSON标注
        # 复制图片到SA-1B目录（保持原文件名）
        img.save(os.path.join(sa1b_img_dir, img_filename))
        
        # 保存JSON标注（文件名与图片名完全一致：如001.json）
        anno_path = os.path.join(sa1b_anno_dir, json_filename)
        
        # 自定义JSON编码器：处理二进制counts的序列化
        class BytesEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, bytes):
                    return obj.decode('latin-1')  # 二进制转字符串（保证可逆）
                return super().default(obj)
        
        with open(anno_path, "w", encoding="utf-8") as f:
            json.dump(sa1b_anno, f, cls=BytesEncoder, indent=2)

        # 更新image_id（若basename是数字，可改为 current_image_id = int(basename)）
        current_image_id = int(basename)

    print(f"\n✅ 转换完成！")
    print(f"- 图片保存至：{sa1b_img_dir}（文件名不变）")
    print(f"- 标注保存至：{sa1b_anno_dir}（JSON文件名与图片名完全一致）")
    print(f"- 生成有效样本数：{current_image_id - start_image_id}")

if __name__ == "__main__":
    # 配置你的实际路径
    INPUT_IMG_DIR = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/Labeled/Original/Images/TrainVal"
    INPUT_MASK_DIR = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/Labeled/Original/Annotations/TrainVal"
    OUTPUT_SA1B_DIR = "/root/autodl-tmp/wound_ai_project/data/DFUTissue/SA1B_Custom_Format"

    # 执行转换
    convert_to_sa1b_custom(
        img_dir=INPUT_IMG_DIR,
        mask_dir=INPUT_MASK_DIR,
        output_dir=OUTPUT_SA1B_DIR,
        img_suffix=".png",       # 你的图片后缀（如.jpg/.png）
        mask_suffix=".png",      # 你的掩码后缀（如.png/.bmp）
        start_image_id=1         # 图片ID起始值（若basename是数字，可注释此行并修改脚本内逻辑）
    )