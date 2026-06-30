#!/usr/bin/env python3
"""通用RAG索引构建工具 - 支持多知识库（医生/患者）【按段落分块】"""
#ai辅助生成：豆包 2026-4-11
import pickle
import numpy as np
from pathlib import Path
# ====== 新增：配置HF国内镜像 ======
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# =================================
from sentence_transformers import SentenceTransformer

# ============ 配置区域 ============
ROOT = Path(__file__).parent / "knowledgebase"
EXCLUDE_DIRS = {'构建脚本', '__pycache__', '.git'}
CHUNK_SIZE = 500
OVERLAP = 50
# 角色与知识库目录映射
ROLE_KNOWLEDGE_MAP = {
    "doctor": "guideline1",    # 医生对应guideline1
    "patient": "guideline2"    # 患者对应guideline2
}
# =================================

print("=" * 60)
print("多角色RAG索引构建工具【按段落分块版】")
print("=" * 60)

# 加载模型
print("\n[1/6] 加载向量编码模型...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
print("✓ 模型加载完成")

# 文本提取函数
def extract_text_from_pdf(pdf_path):
    """从PDF提取文本"""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"  ✗ PDF提取失败: {pdf_path.name} - {e}")
        return ""

def extract_text_from_docx(docx_path):
    """从Word文档提取文本"""
    try:
        from docx import Document
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"  ✗ Word提取失败: {docx_path.name} - {e}")
        return ""

def extract_text_from_txt(txt_path):
    """从TXT提取文本"""
    try:
        return txt_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return txt_path.read_text(encoding='gbk')
        except Exception as e:
            print(f"  ✗ TXT读取失败: {txt_path.name} - {e}")
            return ""

# ========================
# ✅ 核心改进：按段落分块
# ========================
def chunk_text(text):
    """
    按完整段落分块，适配你的txt文档结构
    每一个分级段落 = 1个文本块
    """
    if not text.strip():
        return []
    
    # 按换行分割段落
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # 过滤过短的无效内容
    valid_chunks = [p for p in paragraphs if len(p) > 50]
    
    return valid_chunks

# 单知识库构建函数
def build_knowledge_base(role, kb_dir):
    """为指定角色构建知识库索引"""
    print(f"\n[开始构建 {role} 知识库（{kb_dir}）]")
    
    # 1. 检查目录是否存在
    kb_path = ROOT / kb_dir
    if not kb_path.exists():
        print(f"  ✗ 错误：{kb_dir} 目录不存在，请检查路径")
        return False
    
    # 2. 扫描文档
    print(f"\n[2/6] 扫描 {kb_dir} 下的文档...")
    files = []
    for pattern in ['*.pdf', '*.PDF', '*.txt', '*.TXT', '*.docx', '*.DOCX']:
        for file in kb_path.rglob(pattern):
            if any(ex in file.parts for ex in EXCLUDE_DIRS):
                continue
            if file.stat().st_size < 100:  # 跳过太小的文件
                continue
            files.append(file)
    
    print(f"  ✓ 找到 {len(files)} 个文档")
    if len(files) == 0:
        print(f"  ✗ 警告：{kb_dir} 下无有效文档，跳过该知识库构建")
        return False
    
    # 3. 提取文本并分块
    print(f"\n[3/6] 提取 {kb_dir} 文本并按段落分块...")
    all_chunks = []
    all_sources = []

    for i, file in enumerate(files, 1):
        # 根据文件类型提取文本
        if file.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(file)
        elif file.suffix.lower() == '.docx':
            text = extract_text_from_docx(file)
        else:  # .txt
            text = extract_text_from_txt(file)

        if not text.strip():
            continue

        # 按段落分块
        chunks = chunk_text(text)
        relative_path = str(file.relative_to(kb_path))

        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(relative_path)

    print(f"  ✓ 总共 {len(all_chunks)} 个文本块（按段落）")
    if len(all_chunks) == 0:
        print(f"  ✗ 警告：{kb_dir} 无有效文本块，跳过该知识库构建")
        return False
    
    # 4. 生成向量
    print(f"\n[4/6] 为 {kb_dir} 生成向量（这可能需要几分钟）...")
    embeddings = model.encode(all_chunks,
                             convert_to_tensor=False,
                             show_progress_bar=True,
                             batch_size=32)

    print(f"  ✓ 向量维度: {embeddings.shape}")
    
    # 5. 保存索引
    print(f"\n[5/6] 保存 {role} 知识库索引文件...")
    index_data = {
        'embeddings': embeddings,
        'chunks': all_chunks,
        'sources': all_sources,
        'role': role,
        'kb_dir': kb_dir
    }

    output_file = ROOT / f'{role}_rag_index.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(index_data, f)

    print(f"  ✓ 索引已保存: {output_file}")
    print(f"  ✓ 文件大小: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    return True

# 批量构建所有角色的知识库
print("\n[开始批量构建多角色知识库]")
build_results = {}
for role, kb_dir in ROLE_KNOWLEDGE_MAP.items():
    build_results[role] = build_knowledge_base(role, kb_dir)

# 输出构建总结
print("\n" + "=" * 60)
print("构建总结")
print("=" * 60)
for role, success in build_results.items():
    status = "成功" if success else "失败"
    print(f"{role}（{ROLE_KNOWLEDGE_MAP[role]}）知识库构建: {status}")

# 生成角色-索引文件映射
mapping_file = ROOT / 'role_index_mapping.pkl'
with open(mapping_file, 'wb') as f:
    pickle.dump({
        role: str(ROOT / f'{role}_rag_index.pkl') 
        for role in ROLE_KNOWLEDGE_MAP.keys()
    }, f)
print(f"\n✓ 角色-索引映射文件已保存: {mapping_file}")

print("\n" + "=" * 60)
print("多知识库构建完成！【按段落分块】")
print("=" * 60)
print("使用说明：")
print("  - 医生角色调用: doctor_rag_index.pkl")
print("  - 患者角色调用: patient_rag_index.pkl")
print("\n下一步：运行 python query_rag.py 测试不同角色的查询")