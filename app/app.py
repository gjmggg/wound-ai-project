#ai 辅助生成：Deepseek 2026-4-11
import gradio as gr
import numpy as np
import torch
import cv2
import sys
import os
import pickle
import faiss
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import joblib
import requests
import json
import warnings
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ---------- 路径（请根据实际情况修改）----------
SAM2_CONFIG = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = "./models/checkpoint60.pt"
UNET_WEIGHT = "/root/autodl-tmp/DFUTissueSegNet-main/checkpoints/MiT+pscse_padded_aug_mit_b3_sup_2026-04-05_22-49-27/best_model.pth"
CLASSIFIER_PATH = "./fusion_classifier.pkl"
SCALER_PATH = "/root/autodl-tmp/wound_ai_project/fusion_scaler.pkl"
DINO_LOCAL_DIR = "/root/autodl-tmp/wound_ai_project/dinov2"
DINO_WEIGHT = "/root/autodl-tmp/wound_ai_project/models/dinov2_vitl14_pretrain.pth"
DEEPSEEK_API_KEY = "your api key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# 知识库路径
DOCTOR_KB_PATH = "/root/autodl-tmp/wound_ai_project/knowledge_base/doctor_rag_index.pkl"
PATIENT_KB_PATH = "/root/autodl-tmp/wound_ai_project/knowledge_base/patient_rag_index.pkl"

# ==================== 1. 加载 SAM2（交互式） ====================
print("Loading SAM2...")
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT)
sam2_model.to(DEVICE)
sam2_model.eval()
predictor = SAM2ImagePredictor(sam2_model)

click_x, click_y = 0, 0
def set_click(img, evt: gr.SelectData):
    global click_x, click_y
    click_x = evt.index[0]
    click_y = evt.index[1]
    return f"已点击坐标：({click_x}, {click_y})"

def sam2_segment(img_np):
    if img_np is None:
        return None
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    input_point = np.array([[click_x, click_y]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)]
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    return mask_uint8

# ==================== 2. 加载 DFUTissueSegNet ====================
print("Loading DFUTissueSegNet...")
unet = Unet(encoder_name='mit_b3', encoder_weights=None, classes=4, activation='softmax2d', decoder_attention_type='scse')
state = torch.load(UNET_WEIGHT, map_location='cpu')
unet.load_state_dict(state['state_dict'])
unet.to(DEVICE)
unet.eval()

def preprocess_unet(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0)
    return img_tensor.to(DEVICE)

def predict_unet_mask(img_bgr):
    with torch.no_grad():
        input_tensor = preprocess_unet(img_bgr)
        logits = unet(input_tensor)
        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return mask

def compute_ratios(mask):
    counts = np.bincount(mask.flatten(), minlength=4)
    fibrin = counts[1]
    gran = counts[2]
    callus = counts[3]
    total = fibrin + gran + callus
    if total == 0:
        return 0.0, 0.0, 0.0
    return fibrin/total, gran/total, callus/total

# ==================== 3. 加载本地 DINOv2 ====================
print("Loading local DINOv2...")
sys.path.insert(0, DINO_LOCAL_DIR)
from dinov2.models.vision_transformer import vit_large
model_dino = vit_large(patch_size=14, init_values=1.0, block_chunks=0, img_size=518)
state_dict = torch.load(DINO_WEIGHT, map_location='cpu')
model_dino.load_state_dict(state_dict, strict=True)
model_dino.to(DEVICE)
model_dino.eval()

def preprocess_dinov2(img_rgb):
    img = cv2.resize(img_rgb, (518, 518), interpolation=cv2.INTER_LINEAR)
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)
    return img_tensor.to(DEVICE)

def extract_dinov2_feat(img_rgb):
    with torch.no_grad():
        feat = model_dino(preprocess_dinov2(img_rgb))
        if feat.dim() == 3:
            feat = feat[:, 0, :]
        return feat.cpu().numpy().flatten()

# ==================== 4. 加载融合分类器 ====================
print("Loading fusion classifier...")
clf = joblib.load(CLASSIFIER_PATH)
scaler = joblib.load(SCALER_PATH)
severity_names = ['轻度', '中度', '重度', '溃烂']

# ==================== 5. 伤口形状特征提取 ====================
def extract_shape_features(mask_binary):
    if mask_binary is None or np.sum(mask_binary) == 0:
        return {"area": 0, "perimeter": 0, "aspect_ratio": 0, "roundness": 0}
    mask = (mask_binary > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0, "perimeter": 0, "aspect_ratio": 0, "roundness": 0}
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    roundness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    return {"area": area, "perimeter": perimeter, "aspect_ratio": aspect_ratio, "roundness": roundness}

# ==================== 6. 知识库加载与检索 ====================
print("Loading knowledge bases...")
with open(DOCTOR_KB_PATH, "rb") as f:
    doctor_kb = pickle.load(f)
with open(PATIENT_KB_PATH, "rb") as f:
    patient_kb = pickle.load(f)

doctor_embeddings = doctor_kb["embeddings"]   # numpy array (n_docs, dim)
doctor_chunks = doctor_kb["chunks"]
patient_embeddings = patient_kb["embeddings"]
patient_chunks = patient_kb["chunks"]

# 可选：归一化以便于余弦相似度（如果原向量未归一化）
# doctor_embeddings = doctor_embeddings / np.linalg.norm(doctor_embeddings, axis=1, keepdims=True)
# patient_embeddings = patient_embeddings / np.linalg.norm(patient_embeddings, axis=1, keepdims=True)

# 加载 embedding 模型（必须与构建知识库时使用的模型一致）
embed_model = SentenceTransformer('/root/autodl-tmp/wound_ai_project/models/paraphrase-multilingual-MiniLM-L12-v2')

def retrieve_context(severity, user_type, top_k=3):
    query = f"{severity} 伤口护理建议"
    query_emb = embed_model.encode([query], normalize_embeddings=True)[0]  # (dim,)
    
    if user_type == "医生":
        # 计算余弦相似度（假设向量已归一化）
        sims = np.dot(doctor_embeddings, query_emb)  # (n_docs,)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        contexts = [doctor_chunks[i] for i in top_indices if i < len(doctor_chunks)]
    else:
        sims = np.dot(patient_embeddings, query_emb)
        top_indices = np.argsort(sims)[-top_k:][::-1]
        contexts = [patient_chunks[i] for i in top_indices if i < len(patient_chunks)]
    
    return "\n\n".join(contexts)

# ==================== 7. DeepSeek API 调用（含知识库） ====================
def call_deepseek(severity, ratios, shape_features, user_type):
    fibr, gran, call = ratios

    # ✅ 愈合阶段判断（变量名修正）
    if gran > 0.6:
        healing_stage = "增生期（肉芽组织丰富，愈合良好）"
    elif fibr > 0.4:
        healing_stage = "炎症期/感染期（纤维蛋白较多，需清创）"
    elif call > 0.2:
        healing_stage = "坏死期（焦痂/坏死组织，需干预）"
    else:
        healing_stage = "混合期（需进一步评估）"

    shape_desc = (
        f"伤口面积约{shape_features['area']:.0f}像素，"
        f"周长{shape_features['perimeter']:.1f}像素，"
        f"长宽比{shape_features['aspect_ratio']:.2f}，"
        f"圆度{shape_features['roundness']:.2f}。"
    )

    # ✅ 检索知识库
    kb_context = retrieve_context(severity, user_type)

    prompt = f"""你是一位专业的伤口护理医生。请根据以下信息以及知识库中的参考建议，生成一份针对 **{user_type}** 的详细报告。

【伤口形状特征】
{shape_desc}

【组织成分占比（AI 估算）】
- 肉芽组织：{gran:.1%}
- 纤维蛋白/腐肉：{fibr:.1%}
- 焦痂/坏死组织：{call:.1%}

【严重程度等级】
{severity}

【知识库参考建议】
{kb_context}

请按照以下格式输出报告（语言专业、清晰、实用）：

### 1. 伤口形状评估
（根据提供的形状特征描述）

### 2. 组织成分分析
（解释各组织占比的临床意义）

### 3. 愈合阶段判断
{healing_stage}

### 4. 特征评估
（综合形状和成分，评估当前愈合进程的优缺点）

### 5. 结论与具体建议
（请特别参考知识库内容，给出适合 {user_type} 的详细建议，如居家护理、就医指征、清创建议等）
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",  
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,        
        "max_tokens": 2048
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        # ✅ 打印真实错误，方便调试
        print("=" * 60)
        print("DeepSeek API ERROR:")
        print(e)
        print("Payload:")
        print(payload)
        print("=" * 60)

        return f"""【模拟报告】
由于 API 调用失败，无法生成完整报告。

伤口严重程度：{severity}
肉芽组织：{gran:.1%}
纤维蛋白：{fibr:.1%}
焦痂：{call:.1%}

建议：
1. 保持伤口清洁干燥
2. 避免受压
3. 若红肿、流脓或发黑，请立即就医"""

# ==================== 8. 主处理函数 ====================
def process_wound(image_np, mask_np, user_type):
    if mask_np is None:
        return None, "未检测到伤口", "无", "请重新点击伤口区域并分割"
    if len(mask_np.shape) == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    if np.sum(mask_np) == 0:
        return None, "未检测到伤口", "无", "请重新点击伤口区域并分割"
    
    coords = cv2.findNonZero(mask_np)
    x, y, w, h = cv2.boundingRect(coords)
    wound_crop_bgr = image_np[y:y+h, x:x+w]
    wound_crop_rgb = cv2.cvtColor(wound_crop_bgr, cv2.COLOR_BGR2RGB)
    
    shape_features = extract_shape_features(mask_np)
    
    mask_unet = predict_unet_mask(wound_crop_bgr)
    fibr, gran, call = compute_ratios(mask_unet)
    
    feat = extract_dinov2_feat(wound_crop_rgb)
    feat_concat = np.concatenate([feat, [fibr, gran, call]])
    feat_scaled = scaler.transform([feat_concat])
    pred_label = clf.predict(feat_scaled)[0]
    severity = severity_names[pred_label]
    
    report = call_deepseek(severity, (fibr, gran, call), shape_features, user_type)
    
    mask_unet_resized = cv2.resize(mask_unet, (w, h), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[mask_unet_resized == 1] = [255, 0, 0]
    color_mask[mask_unet_resized == 2] = [0, 255, 0]
    color_mask[mask_unet_resized == 3] = [0, 0, 255]
    
    ratios_str = f"肉芽 {gran:.1%} | 纤维蛋白 {fibr:.1%} | 焦痂 {call:.1%}"
    return color_mask, severity, ratios_str, report

# ==================== 9. Gradio 界面 ====================
import gradio as gr
import os

# 消除 OpenMP 警告
os.environ["OMP_NUM_THREADS"] = "1"

# ============ 配色方案 ============
PRIMARY_COLOR = "#2E86AB"
BG_COLOR = "#F8F9FA"

# ============ 自定义 CSS（核心：控制字体和排版） ============
custom_css = """
/* 全局字体 */
body {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    background-color: #F8F9FA;
}

/* ========== 1. 标题栏（缩小） ========== */
.header {
    background: linear-gradient(135deg, #2E86AB 0%, #3A9BC6 100%);
    padding: 1rem 1rem;              /* ✅ 从 2rem → 1rem */
    border-radius: 0 0 15px 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    text-align: center;
    color: white;
}

.header h1 {
    font-size: 1.6rem !important;      /* ✅ 从 2.2rem → 1.6rem */
    font-weight: 600 !important;
    margin: 0 !important;
}

.header p {
    font-size: 0.95rem !important;     /* ✅ 副标题缩小 */
    opacity: 0.9;
    margin-top: 0.3rem !important;
}

/* ========== 2. 卡片（缩小内边距） ========== */
.identity-card {
    background: white;
    border-radius: 12px;
    padding: 1rem;                    /* ✅ 从 2rem → 1rem */
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin: 1rem auto;
    max-width: 600px;
    border: 1px solid #DEE2E6;
}

.identity-card h2,
.identity-card h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #2E86AB !important;
    margin-bottom: 0.8rem !important;
}

.identity-card p {
    font-size: 0.95rem !important;
    color: #6C757D !important;
}

/* ========== 3. Radio 选项 ========== */
.radio-group {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
}

.radio-item {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.9rem 1.2rem;
    border: 2px solid #DEE2E6;
    border-radius: 10px;
    background: #F8F9FA;
    cursor: pointer;
    transition: all 0.2s ease;
}

.radio-item:hover {
    border-color: #2E86AB;
    background: #E3F2FD;
}

.radio-item.selected {
    border-color: #2E86AB;
    background: #BBDEFB;
}

.radio-item label {
    font-size: 1.1rem !important;   /* ✅ 选项文字放大 */
    font-weight: 500 !important;
}

/* ========== 4. 确认按钮 ========== */
.confirm-btn {
    width: 100%;
    padding: 1rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    background: #2E86AB !important;
    color: white !important;
    border: none !important;
}

/* ========== 5. 诊断区域（左右栏） ========== */
.diagnosis-row {
    gap: 10px !important;           /* ✅ 关键：左右栏间距缩小 */
}

/* 左右卡片 */
.diagnosis-col {
    padding: 1rem !important;        /* ✅ 卡片内边距缩小 */
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    background: white;
}

.diagnosis-col h3 {
    font-size: 1.2rem !important;
    margin-bottom: 0.8rem !important;
}

/* ========== 6. 报告字体（重点） ========== */
.report-textbox textarea {
    font-size: 16px !important;        /* ✅ 报告字体放大 */
    line-height: 1.6 !important;
    font-family: 'Microsoft YaHei', sans-serif !important;
}

/* ========== 7. 按钮组 ========== */
.button-group {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
}

.button-group button {
    flex: 1;
    font-size: 0.95rem !important;
}
"""

with gr.Blocks(
    title="糖尿病足伤口智能诊断系统",
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Noto Sans SC"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono")
    )
) as demo:

    # ============ 顶部标题 ============
    gr.HTML("""
    <div class="header">
        <h1>糖尿病足伤口智能诊断系统</h1>
        <p>AI 辅助 · 专业诊断 · 精准护理</p>
    </div>
    """)

    # ============ 身份选择区域 ============
    with gr.Row(visible=True) as identity_row:
        with gr.Column(elem_classes="identity-card"):
            gr.Markdown("### 👤 请选择您的身份")
            gr.Markdown("不同身份将获得针对性的建议")

            user_type = gr.Radio(
                choices=["患者", "医生"],
                value="患者",
                label="身份选择",
                interactive=True,
                elem_classes="radio-group"
            )

            confirm_btn = gr.Button(
                "✅ 确认身份，进入诊断",
                elem_classes="confirm-btn"
            )

    # ============ 诊断流程区域（保持不变，但应用了CSS） ============
    with gr.Row(visible=False, elem_classes="diagnosis-row") as diagnosis_row:
        # 左侧：图像与分割
        with gr.Column(scale=1, elem_classes="diagnosis-col"):
            gr.Markdown("### 📷 伤口图像与分割")
            input_img = gr.Image(
                type="numpy",
                label="上传足部照片",
                height=300
            )
            tip = gr.Label(
                "👉 点击图片上的伤口中心位置",
                value="等待点击..."
            )
            with gr.Row():
                seg_btn = gr.Button("✂️ 分割伤口", variant="secondary")
                clear_btn = gr.Button("🔄 重新选择", variant="secondary")
            mask_output = gr.Image(
                type="numpy",
                label="🔍 SAM2 分割结果",
                height=300
            )

        # 右侧：诊断结果与报告
        with gr.Column(scale=1, elem_classes="diagnosis-col"):
            gr.Markdown("### 📋 诊断结果与建议")
            diagnose_btn = gr.Button(
                "🏥 开始诊断",
                variant="primary",
                size="lg"
            )
            triple_mask = gr.Image(
                type="numpy",
                label="🎨 组织分割结果",
                height=300
            )
            severity_text = gr.Textbox(
                label="📊 严重程度",
                interactive=False
            )
            ratios_text = gr.Textbox(
                label="📈 组织占比（估算）",
                interactive=False
            )
            report_text = gr.Textbox(
                label="📄 诊断报告（仅供参考）",
                lines=12,
                interactive=False,
                elem_classes="report-textbox"
            )

    # ============ 交互逻辑 ============
    user_state = gr.State(value="患者")

    def on_confirm(identity):
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            identity
        )

    def on_clear():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "患者",
            None, None, None, None, None, "等待点击..."
        )

    confirm_btn.click(
        fn=on_confirm,
        inputs=user_type,
        outputs=[identity_row, diagnosis_row, user_state]
    )

    clear_btn.click(
        fn=on_clear,
        outputs=[
            identity_row, diagnosis_row, user_state,
            input_img, tip, mask_output,
            triple_mask, severity_text, ratios_text, report_text
        ]
    )

    input_img.select(set_click, inputs=input_img, outputs=tip)
    seg_btn.click(fn=sam2_segment, inputs=input_img, outputs=mask_output)
    diagnose_btn.click(
        fn=process_wound,
        inputs=[input_img, mask_output, user_state],
        outputs=[triple_mask, severity_text, ratios_text, report_text]
    )

# ============ 启动 ============
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9100)