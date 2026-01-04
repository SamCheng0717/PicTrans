# Qwen-Image-Edit-2511 Inpaint Server

在 3090 GPU 服务器上部署 AI 图像修复服务。

## 硬件要求

- GPU: RTX 3090 24GB (或更高)
- 推荐使用 int4 量化，显存占用约 17GB

## 安装

```bash
# 1. 创建虚拟环境
conda create -n qwen-inpaint python=3.10
conda activate qwen-inpaint

# 2. 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.5 使用 cu117 (兼容)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# 或者 CUDA 11.8+
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 diffusers (需要最新版本)
pip install git+https://github.com/huggingface/diffusers

# 4. 安装其他依赖
pip install bitsandbytes accelerate fastapi uvicorn pillow transformers
```

### CUDA 版本对照

| 服务器 CUDA | PyTorch 安装 |
|-------------|--------------|
| 11.5 / 11.6 / 11.7 | `cu117` |
| 11.8 | `cu118` |
| 12.x | `cu121` |

## 启动服务

```bash
cd server

# 使用默认 GPU 0
python qwen_inpaint_server.py --port 8765

# 指定使用 GPU 2
python qwen_inpaint_server.py --port 8765 --gpu 2

# 使用本地模型路径（避免 HuggingFace 限速）
python qwen_inpaint_server.py --port 8765 --gpu 2 --model /data/models/Qwen-Image-Edit-2511
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 监听地址 |
| `--port` | 8765 | 监听端口 |
| `--gpu` | 0 | 使用的 GPU 编号 |
| `--model` | Qwen/Qwen-Image-Edit-2511 | 模型路径（HF 模型名或本地路径） |
| `--reload` | - | 开发模式，自动重载 |

### 解决 HuggingFace 限速问题

如果遇到 429 Rate Limit 错误，有两种解决方案：

**方案一：设置 HuggingFace Token**
```bash
# 登录 https://huggingface.co 获取 token
export HF_TOKEN=hf_your_token_here
python qwen_inpaint_server.py --port 8765 --gpu 2
```

**方案二：预下载模型到本地**
```bash
# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir /data/models/Qwen-Image-Edit-2511

# 然后指定本地路径启动
python qwen_inpaint_server.py --port 8765 --gpu 2 --model /data/models/Qwen-Image-Edit-2511
```

服务启动后：
- API 地址: `http://<server-ip>:8765`
- 文档地址: `http://<server-ip>:8765/docs`
- 健康检查: `http://<server-ip>:8765/health`

## API 接口

### POST /inpaint

请求体：
```json
{
    "image": "<base64 编码的原图>",
    "mask": "<base64 编码的 mask，白色=要擦除>",
    "prompt": "Remove all text and symbols..."
}
```

响应：
```json
{
    "success": true,
    "output": "<base64 编码的结果图>"
}
```

## 客户端使用

在 PicTrans 中使用：

```bash
# 指定 Qwen 服务器地址
python cli.py input/test.jpg -t ko --inpaint qwen --qwen-url http://192.168.1.100:8765
```

## 显存优化

如果显存不足，可以启用 4-bit 量化：

```python
# 修改 qwen_inpaint_server.py 中的模型加载代码
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,  # 启用 4-bit 量化
)
```
