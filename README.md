# wound-ai-project
## 创建虚拟环境
conda create -n wound_ai python=3.10
conda activate wound_ai

## 安装依赖
pip install -r requirements.txt

## 下载模型权重
由于文件大小超过 GitHub 100MB 限制，需手动下载并放置到指定路径
整个models文件该连接获取
https://pan.baidu.com/s/1rXlDPuoZByYry9cVKJvkCw?pwd=ix55 

models文件结构如下（路径可在 app.py 中自行修改）：

DINOv2 权重：从官方文件下载dinov2_vitl14_pretrain.pth并放在 ./models/dinov2_vitl14_pretrain.pth，由于文件太大，未直接上传，需自行下载

SAM2 微调权重：本作品微调后的权重为checkpoint60.pt，已上传至 ./models/checkpoint60.pt

DUFUTissueSegNet组织分割权重已放入 ./models/best_model.pth

同时paraphrase-multilingual-MiniLM-L12-v2也放至./models下


## 申请 DeepSeek API 密钥，并在 app.py 中填入
DEEPSEEK_API_KEY = "your-api-key-here"

## 运行系统
python app.py
终端会显示本地访问端口，例如 http://0.0.0.0:9100，则本地端口为9100
可借助AutoDLSSH隧道工具访问
