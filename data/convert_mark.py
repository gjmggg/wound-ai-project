import os
import cv2
import numpy as np
from tqdm import tqdm
#ai辅助生成：元宝 2026-4-4

def convert_labels_to_binary_masks(label_dir, output_dir):
    """
    将单通道标签图转换为黑白掩码图
    
    参数:
    - label_dir: 单通道标签图目录路径
    - output_dir: 输出黑白掩码图目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    if not label_files:
        print(f"在 {label_dir} 目录中未找到图片文件")
        return
    
    print(f"找到 {len(label_files)} 个标签文件")
    print(f"开始转换...")
    
    for filename in tqdm(label_files, desc="转换进度"):
        # 读取标签图
        label_path = os.path.join(label_dir, filename)
        label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if label_img is None:
            print(f"警告: 无法读取文件 {filename}")
            continue
        
        # 检查是否为单通道标签图
        unique_values = np.unique(label_img)
        if not np.all(np.isin(unique_values, [0, 1, 2, 3])):
            print(f"警告: {filename} 的像素值不在预期范围 {unique_values} 内")
            # 如果不是标准标签，尝试提取非零区域
            binary_mask = (label_img > 0).astype(np.uint8) * 255
        else:
            # 标准转换：将非0值转换为255
            binary_mask = np.where(label_img > 0, 255, 0).astype(np.uint8)
        
        # 保存为8位黑白掩码图
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.png')
        cv2.imwrite(output_path, binary_mask)
        
        # 可选：显示转换前后的对比
        if np.random.random() < 0.1:  # 10%的概率显示示例
            print(f"\n示例转换: {filename}")
            print(f"  原图像素值: {np.unique(label_img)}")
            print(f"  掩码像素值: {np.unique(binary_mask)}")
    
    print(f"\n✅ 转换完成!")
    print(f"输入目录: {label_dir}")
    print(f"输出目录: {output_dir}")
    print(f"转换了 {len(label_files)} 个文件")

def verify_conversion(output_dir, num_samples=3):
    """
    验证转换结果
    
    参数:
    - output_dir: 输出目录路径
    - num_samples: 随机检查的文件数量
    """
    if not os.path.exists(output_dir):
        print(f"输出目录 {output_dir} 不存在")
        return
    
    output_files = [f for f in os.listdir(output_dir) 
                    if f.lower().endswith('.png')]
    
    if not output_files:
        print("没有找到转换后的文件")
        return
    
    print(f"\n🔍 验证转换结果:")
    print(f"输出目录中共有 {len(output_files)} 个文件")
    
    # 随机检查几个文件
    import random
    sample_files = random.sample(output_files, min(num_samples, len(output_files)))
    
    for filename in sample_files:
        filepath = os.path.join(output_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            unique_values = np.unique(img)
            height, width = img.shape
            
            print(f"\n文件: {filename}")
            print(f"  尺寸: {width}x{height}")
            print(f"  像素值: {unique_values}")
            print(f"  前景像素比例: {np.sum(img > 0) / (height * width):.2%}")
        else:
            print(f"无法读取文件: {filename}")

if __name__ == "__main__":
    # 设置路径
    label_dir = "/root/autodl-tmp/sam2-main/wound_dataset1/Annotations"  # 单通道标签图路径
    output_dir = "/root/autodl-tmp/sam2-main/heibaiyanma"  # 输出黑白掩码图路径
    
    # 运行转换
    convert_labels_to_binary_masks(label_dir, output_dir)
    
    # 验证结果
    verify_conversion(output_dir)
    
    # 可选：统计信息
    print(f"\n📊 统计信息:")
    print("单通道标签图中的标签含义:")
    print("  0: 背景")
    print("  1: 肉芽组织")
    print("  2: 纤维蛋白")
    print("  3: 焦痂")
    print("黑白掩码图:")
    print("  0: 背景")
    print("  255: 伤口区域（肉芽+纤维蛋白+焦痂）")