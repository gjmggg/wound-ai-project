# test_dfu_segmentation.py
#!/usr/bin/env python
# coding: utf-8
"""
糖尿病足溃疡四分类分割模型测试脚本
功能：加载训练好的模型权重，对单张伤口图片进行分割预测和可视化分析
注意：此脚本仅用于推理，不会开始训练
"""
#ai辅助生成：元宝 2026-4-06
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 修复OpenMP警告
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU 0

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from datetime import datetime

# 医学调色板和类别定义
MEDICAL_CONFIG = {
    'palette': [
        [0, 0, 0],        # 0: 背景 - 黑色
        [225, 0, 0],      # 1: 纤维蛋白 (Fibrin) - 红色
        [0, 255, 0],      # 2: 肉芽 (Granulation) - 绿色
        [0, 0, 225]       # 3: 焦痂 (Callus) - 蓝色
    ],
    'class_names': {
        0: '背景 (Background)',
        1: '纤维蛋白 (Fibrin)',
        2: '肉芽 (Granulation)', 
        3: '焦痂 (Callus)'
    },
    'descriptions': {
        0: '正常皮肤/非溃疡区域',
        1: '干燥、坚硬的坏死组织',
        2: '新生血管和结缔组织',
        3: '黄色/白色的纤维蛋白渗出物'
    }
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='糖尿病足溃疡四分类分割模型测试')
    parser.add_argument('--weight_path', type=str, required=True,
                       default='/root/autodl-tmp/DFUTissueSegNet-main/checkpoints/MiT+pscse_padded_aug_mit_b3_sup_2026-04-05_22-49-27/best_model.pth',
                       help='模型权重文件路径')
    parser.add_argument('--image_path', type=str, required=True,
                       default='/root/autodl-tmp/wound_ai_project/data/crop_data/test/0925.png',
                       help='待测试的伤口图片路径')
    parser.add_argument('--gt_path', type=str, default='/root/autodl-tmp/DFUTissueSegNet-main/DFUTissue/Labeled/Padded/Annotations/Test/0925.png',
                       help='真实标注路径（可选，用于计算指标）')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/DFUTissueSegNet-main/result',
                       help='输出结果保存目录')
    parser.add_argument('--no_display', action='store_true', default=False,
                       help='不显示图表，仅保存结果')
    parser.add_argument('--save_all', action='store_true', default=True,
                       help='保存所有中间结果')
    return parser.parse_args()

def load_model(weight_path, device):
    """加载训练好的模型"""
    print("🤖 正在加载模型...")
    
    import segmentation_models_pytorch as smp
    
    # 模型参数（必须与训练时一致）
    n_classes = 4
    ENCODER = 'mit_b3'
    ACTIVATION = 'softmax2d'
    
    # 创建模型
    model = smp.Unet(
        encoder_name='mit_b3',
        encoder_weights=None,
        classes=4,
        activation='softmax2d',
        decoder_attention_type='scse',  # 添加注意力机制
        decoder_use_batchnorm=True,     # 启用批归一化
        decoder_channels=(256, 128, 64, 32, 16),  # 与训练一致
        encoder_depth=5,                # 与训练一致
    )

    
    
    # 加载权重
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint.get('epoch', '未知')
            print(f"✅ 模型加载成功! 来自epoch: {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ 模型加载成功!")
            epoch = '未知'
            
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    return model, epoch

def preprocess_image(image_path, target_size=(256, 256)):
    """预处理伤口图片"""
    print(f"📷 处理图片: {os.path.basename(image_path)}")
    
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    original_img = img.copy()
    
    # 调整大小
    img_resized = cv2.resize(img, target_size)
    
    # 标准化（ImageNet标准）
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    
    # 转换为tensor
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
    
    return original_img, img_tensor, (original_h, original_w), img_resized

def predict(model, img_tensor, device):
    """进行预测"""
    print("🔍 进行分割预测...")
    
    with torch.no_grad():
        # 添加批次维度
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # 模型预测
        prediction = model(img_batch)
        
        # 获取类别索引
        pred_class = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
        
        # 获取各类别概率
        probs = torch.softmax(prediction.squeeze(0), dim=0).cpu().numpy()
    
    print(f"✅ 预测完成! 形状: {pred_class.shape}")
    return pred_class, probs

def create_colored_prediction(pred_class):
    """创建彩色分割图"""
    h, w = pred_class.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls in range(4):
        mask = pred_class == cls
        colored[mask] = MEDICAL_CONFIG['palette'][cls]
    
    return colored

def analyze_medical_plausibility(pred_class):
    """分析预测结果的医学合理性"""
    print("\n" + "="*60)
    print("🏥 医学合理性分析")
    print("="*60)
    
    unique_classes, class_counts = np.unique(pred_class, return_counts=True)
    total_pixels = pred_class.size
    
    # 统计信息
    stats = {}
    for cls, count in zip(unique_classes, class_counts):
        stats[cls] = {
            'count': count,
            'percentage': (count / total_pixels) * 100
        }
    
    # 1. 总体分析
    background_pct = stats[0]['percentage'] if 0 in stats else 0
    ulcer_pct = 100 - background_pct
    
    print(f"📊 总体统计:")
    print(f"  • 图片尺寸: {pred_class.shape}")
    print(f"  • 总像素数: {total_pixels:,}")
    print(f"  • 溃疡区域: {ulcer_pct:.1f}%")
    print(f"  • 背景区域: {background_pct:.1f}%")
    
    # 2. 各类组织分析
    print(f"\n🔬 组织类型分析:")
    for cls in range(4):
        if cls in stats:
            info = stats[cls]
            print(f"  • {MEDICAL_CONFIG['class_names'][cls]}:")
            print(f"     像素数: {info['count']:,}")
            print(f"     占比: {info['percentage']:.1f}%")
    
    # 3. 医学评估
    print(f"\n💡 医学评估:")
    
    # 溃疡区域大小评估
    if ulcer_pct < 5:
        print("  ⚠️ 溃疡区域过小 (<5%)，可能：")
        print("    - 模型未完全识别伤口")
        print("    - 图片中伤口确实很小")
        print("    - 预测阈值可能过高")
    elif ulcer_pct > 50:
        print("  ⚠️ 溃疡区域过大 (>50%)，可能：")
        print("    - 模型过度分割")
        print("    - 伤口确实很大")
        print("    - 背景识别不准确")
    else:
        print("  ✅ 溃疡区域比例合理")
    
    # 组织分布评估
    if 3 in stats and stats[3]['percentage'] > 30:  # 大量纤维蛋白
        print("  ⚠️ 大量纤维蛋白，可能表示：")
        print("    - 感染性伤口")
        print("    - 渗出物较多")
        print("    - 需要清创处理")
    
    if 2 in stats and stats[2]['percentage'] > 10:  # 较多肉芽
        print("  ✅ 检测到肉芽组织，表明：")
        print("    - 伤口正在愈合")
        print("    - 有新生血管形成")
    
    if 1 in stats and stats[1]['percentage'] > 15:  # 较多焦痂
        print("  ⚠️ 较多焦痂组织，建议：")
        print("    - 考虑清创处理")
        print("    - 保持伤口湿润")
    
    return stats

def visualize_results(original_img, original_resized, pred_class, probs, stats, output_dir, show=True):
    """可视化所有结果"""
    print("\n🎨 生成可视化图表...")
    
    # 创建彩色预测图
    pred_colored = create_colored_prediction(pred_class)
    
    # 创建叠加图
    overlay = cv2.addWeighted(original_resized, 0.6, pred_colored, 0.4, 0)
    
    # 创建完整可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 原始图片
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(original_img)
    ax1.set_title('原始伤口图片', fontsize=12, fontweight='bold', color='darkblue')
    ax1.axis('off')
    
    # 2. 调整后图片
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(original_resized)
    ax2.set_title('调整后图片 (256×256)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. 预测类别索引
    ax3 = plt.subplot(2, 4, 3)
    im3 = ax3.imshow(pred_class, cmap='jet', vmin=0, vmax=3)
    ax3.set_title('预测类别索引', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, 
                 ticks=[0, 1, 2, 3], 
                 label='类别编号')
    
    # 4. 医学彩色分割
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(pred_colored)
    ax4.set_title('医学彩色分割图', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='black', label='背景'),
        plt.Rectangle((0,0),1,1, facecolor='blue', label='焦痂'),
        plt.Rectangle((0,0),1,1, facecolor='green', label='肉芽'),
        plt.Rectangle((0,0),1,1, facecolor='red', label='纤维蛋白')
    ]
    ax4.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=9)
    
    # 5. 叠加显示
    ax5 = plt.subplot(2, 4, 5)
    ax5.imshow(overlay)
    ax5.set_title('叠加显示 (40%透明度)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. 概率图 - 背景
    ax6 = plt.subplot(2, 4, 6)
    im6 = ax6.imshow(probs[0], cmap='hot', vmin=0, vmax=1)
    ax6.set_title('背景概率图', fontsize=12, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # 7. 概率图 - 焦痂
    ax7 = plt.subplot(2, 4, 7)
    im7 = ax7.imshow(probs[1], cmap='hot', vmin=0, vmax=1)
    ax7.set_title('焦痂概率图', fontsize=12, fontweight='bold')
    ax7.axis('off')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # 8. 组织分布饼图
    ax8 = plt.subplot(2, 4, 8)
    
    # 准备饼图数据（跳过背景）
    labels = []
    sizes = []
    colors = []
    
    for cls in range(1, 4):  # 只显示组织类型
        if cls in stats:
            labels.append(f"{MEDICAL_CONFIG['class_names'][cls]}\n{stats[cls]['percentage']:.1f}%")
            sizes.append(stats[cls]['count'])
            # 转换颜色为matplotlib格式
            color = [c/255 for c in MEDICAL_CONFIG['palette'][cls]]
            colors.append(color)
    
    if sizes:  # 如果有组织类型
        wedges, texts, autotexts = ax8.pie(
            sizes, labels=labels, colors=colors, 
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 9}
        )
        ax8.set_title('组织类型分布', fontsize=12, fontweight='bold')
    else:
        ax8.text(0.5, 0.5, '未检测到组织类型', 
                ha='center', va='center', fontsize=12)
        ax8.axis('off')
    
    # 添加标题和时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.suptitle('糖尿病足溃疡组织分割分析\n' + current_time, 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存图表
    viz_path = os.path.join(output_dir, 'visualization_summary.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化图表已保存: {viz_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return viz_path

def save_predictions(pred_class, pred_colored, probs, original_img, output_dir):
    """保存所有预测结果"""
    print("\n💾 保存预测结果...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存原始预测（单通道）
    pred_path = os.path.join(output_dir, 'prediction_mask.png')
    cv2.imwrite(pred_path, pred_class.astype(np.uint8))
    print(f"  ✅ 预测掩码: {pred_path}")
    
    # 2. 保存彩色预测
    pred_colored_bgr = cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR)
    colored_path = os.path.join(output_dir, 'prediction_colored.png')
    cv2.imwrite(colored_path, pred_colored_bgr)
    print(f"  ✅ 彩色预测: {colored_path}")
    
    # 3. 保存原始图片
    original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    original_path = os.path.join(output_dir, 'original_image.png')
    cv2.imwrite(original_path, original_bgr)
    print(f"  ✅ 原始图片: {original_path}")
    
    # 4. 保存各组织类型的分割图
    for cls in range(1, 4):  # 跳过背景
        tissue_mask = (pred_class == cls).astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f'tissue_{cls}_{MEDICAL_CONFIG["class_names"][cls].split()[0]}.png')
        cv2.imwrite(mask_path, tissue_mask)
    
    # 5. 保存概率图
    for cls in range(4):
        prob_map = (probs[cls] * 255).astype(np.uint8)
        prob_path = os.path.join(output_dir, f'probability_class_{cls}.png')
        cv2.imwrite(prob_path, prob_map)
    
    print(f"  ✅ 所有结果已保存到: {output_dir}")
    
    return {
        'mask': pred_path,
        'colored': colored_path,
        'original': original_path
    }

def evaluate_with_ground_truth(pred_class, gt_path, output_dir):
    """与真实标注对比评估"""
    if not gt_path or not os.path.exists(gt_path):
        return None
    
    print("\n📈 与真实标注对比评估...")
    
    # 加载真实标注
    ground_truth = cv2.imread(gt_path, 0)  # 灰度图
    if ground_truth is None:
        print("  ⚠️ 无法加载真实标注")
        return None
    
    # 调整到相同尺寸
    if ground_truth.shape != pred_class.shape:
        ground_truth = cv2.resize(ground_truth, (pred_class.shape[1], pred_class.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # 计算指标
    from sklearn.metrics import confusion_matrix, classification_report
    
    y_true = ground_truth.flatten()
    y_pred = pred_class.flatten()
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # 分类报告
    target_names = ['背景', '焦痂', '肉芽', '纤维蛋白']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # 计算各类IoU
    ious = {}
    for cls in range(4):
        pred_cls = (y_pred == cls)
        true_cls = (y_true == cls)
        
        intersection = np.logical_and(pred_cls, true_cls).sum()
        union = np.logical_or(pred_cls, true_cls).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        
        ious[cls] = iou
    
    # 平均IoU（跳过背景）
    foreground_ious = [iou for cls, iou in ious.items() if cls != 0]
    mean_iou = np.mean(foreground_ious) if foreground_ious else 0.0
    
    # 像素级准确率
    accuracy = (y_pred == y_true).sum() / len(y_true)
    
    # 打印结果
    print("  📊 性能指标:")
    print(f"    像素级准确率: {accuracy:.4f}")
    print(f"    平均IoU (跳过背景): {mean_iou:.4f}")
    for cls in range(4):
        print(f"    {target_names[cls]} IoU: {ious[cls]:.4f}")
    
    # 保存评估结果
    eval_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(eval_path, 'w') as f:
        f.write("糖尿病足溃疡分割模型评估报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("性能指标:\n")
        f.write(f"- 像素级准确率: {accuracy:.4f}\n")
        f.write(f"- 平均IoU (跳过背景): {mean_iou:.4f}\n")
        for cls in range(4):
            f.write(f"- {target_names[cls]} IoU: {ious[cls]:.4f}\n")
        
        f.write("\n分类报告:\n")
        for cls_name in target_names:
            if cls_name in report:
                f.write(f"\n{cls_name}:\n")
                f.write(f"  精确率: {report[cls_name]['precision']:.4f}\n")
                f.write(f"  召回率: {report[cls_name]['recall']:.4f}\n")
                f.write(f"  F1分数: {report[cls_name]['f1-score']:.4f}\n")
    
    print(f"  ✅ 评估报告已保存: {eval_path}")
    
    return {
        'accuracy': accuracy,
        'ious': ious,
        'mean_iou': mean_iou,
        'report': report
    }

def generate_summary_report(args, epoch, stats, eval_results, output_dir):
    """生成汇总报告"""
    print("\n📄 生成测试报告...")
    
    report_path = os.path.join(output_dir, 'test_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("糖尿病足溃疡四分类分割模型测试报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("测试配置:\n")
        f.write(f"- 权重文件: {args.weight_path}\n")
        f.write(f"- 测试图片: {args.image_path}\n")
        f.write(f"- 模型epoch: {epoch}\n")
        f.write(f"- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 输出目录: {output_dir}\n\n")
        
        f.write("预测统计:\n")
        f.write(f"- 图片尺寸: {stats.get('shape', '未知')}\n")
        f.write(f"- 总像素数: {stats.get('total_pixels', 0):,}\n")
        
        for cls in range(4):
            if cls in stats:
                f.write(f"- {MEDICAL_CONFIG['class_names'][cls]}:\n")
                f.write(f"    像素数: {stats[cls]['count']:,}\n")
                f.write(f"    占比: {stats[cls]['percentage']:.1f}%\n")
        
        f.write("\n医学评估:\n")
        ulcer_pct = 100 - (stats[0]['percentage'] if 0 in stats else 0)
        f.write(f"- 溃疡区域占比: {ulcer_pct:.1f}%\n")
        
        if ulcer_pct < 5:
            f.write("- 评估: 溃疡区域过小\n")
        elif ulcer_pct > 50:
            f.write("- 评估: 溃疡区域过大\n")
        else:
            f.write("- 评估: 溃疡区域比例合理\n")
        
        if eval_results:
            f.write("\n性能评估 (与真实标注对比):\n")
            f.write(f"- 像素级准确率: {eval_results['accuracy']:.4f}\n")
            f.write(f"- 平均IoU (跳过背景): {eval_results['mean_iou']:.4f}\n")
            for cls, iou in eval_results['ious'].items():
                f.write(f"- {MEDICAL_CONFIG['class_names'][cls]} IoU: {iou:.4f}\n")
        
        f.write("\n生成的文件:\n")
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.txt')):
                file_size = os.path.getsize(os.path.join(output_dir, file)) / 1024
                f.write(f"- {file} ({file_size:.1f} KB)\n")
    
    print(f"✅ 测试报告已保存: {report_path}")
    return report_path

def main():
    """主函数"""
    print("="*60)
    print("🩺 糖尿病足溃疡四分类分割模型测试")
    print("="*60)
    
    # 解析参数
    args = parse_arguments()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 使用设备: {device}")
    
    try:
        # 1. 加载模型
        model, epoch = load_model(args.weight_path, device)
        
        # 2. 预处理图片
        original_img, img_tensor, original_size, original_resized = preprocess_image(args.image_path)
        
        # 3. 进行预测
        pred_class, probs = predict(model, img_tensor, device)
        
        # 4. 医学合理性分析
        stats = analyze_medical_plausibility(pred_class)
        stats['shape'] = pred_class.shape
        stats['total_pixels'] = pred_class.size
        
        # 5. 可视化结果
        viz_path = visualize_results(
            original_img, original_resized, pred_class, probs, 
            stats, output_dir, show=not args.no_display
        )
        
        # 6. 保存预测结果
        pred_colored = create_colored_prediction(pred_class)
        saved_files = save_predictions(pred_class, pred_colored, probs, original_img, output_dir)
        
        # 7. 与真实标注对比（如果提供）
        eval_results = None
        if args.gt_path:
            eval_results = evaluate_with_ground_truth(pred_class, args.gt_path, output_dir)
        
        # 8. 生成汇总报告
        report_path = generate_summary_report(args, epoch, stats, eval_results, output_dir)
        
        print("\n" + "="*60)
        print("✅ 测试完成!")
        print(f"📁 所有结果已保存到: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()