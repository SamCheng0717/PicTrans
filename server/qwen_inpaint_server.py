"""
Qwen-Image-Edit-2511 Inpaint 服务器
部署在 3090 GPU 服务器上

使用方法:
1. 安装依赖:
   pip install torch diffusers bitsandbytes accelerate fastapi uvicorn pillow

2. 启动服务:
   python qwen_inpaint_server.py --port 8765 --gpu 2

3. 客户端调用:
   POST http://<server-ip>:8765/inpaint
   {
       "image": "<base64>",
       "mask": "<base64>",
       "prompt": "Remove all text..."
   }
"""
import os
import io
import base64
import argparse
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# 全局变量
pipeline = None
device = None
gpu_id = 0  # 默认使用 GPU 0
model_path = "Qwen/Qwen-Image-Edit-2511"  # 模型路径


class InpaintRequest(BaseModel):
    """Inpaint 请求"""
    image: str  # base64 编码的原图
    mask: str   # base64 编码的 mask
    prompt: Optional[str] = "Remove all text and symbols in the masked area. Reconstruct the background naturally."


class InpaintResponse(BaseModel):
    """Inpaint 响应"""
    success: bool
    output: Optional[str] = None  # base64 编码的结果图
    error: Optional[str] = None


def load_model():
    """加载 Qwen-Image-Edit-2511 模型"""
    global pipeline, device, gpu_id
    import torch

    print("正在加载 Qwen-Image-Edit-2511 模型...")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        # 由于使用了 CUDA_VISIBLE_DEVICES，选中的 GPU 会被映射为 cuda:0
        gpu_count = torch.cuda.device_count()
        print(f"可见 GPU 数量: {gpu_count}")

        # 使用 cuda:0（因为 CUDA_VISIBLE_DEVICES 已经选择了指定的 GPU）
        device = "cuda:0"
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("警告: 未检测到 GPU，将使用 CPU（会很慢）")

    try:
        from diffusers import QwenImageEditPipeline

        # 使用 4-bit 量化减少显存占用
        print(f"从 {model_path} 加载模型...")
        pipeline = QwenImageEditPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,  # 4-bit 量化，显存占用约 10-12GB
        )
        pipeline.to(device)

        print(f"模型加载完成！运行在 {device}")
        return True

    except ImportError:
        print("错误: 请安装 diffusers 库")
        print("pip install git+https://github.com/huggingface/diffusers")
        return False
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def base64_to_pil(base64_str: str) -> Image.Image:
    """Base64 转 PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def pil_to_base64(image: Image.Image) -> str:
    """PIL Image 转 Base64"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# FastAPI 应用
app = FastAPI(
    title="Qwen-Image-Edit-2511 Inpaint Server",
    description="AI 图像修复服务，用于擦除文字并重建背景"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "device": device
    }


@app.post("/inpaint", response_model=InpaintResponse)
async def inpaint(request: InpaintRequest):
    """执行 inpaint 操作"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 解码图像
        image = base64_to_pil(request.image)
        mask = base64_to_pil(request.mask).convert("L")

        print(f"[Inpaint] 图像尺寸: {image.size}, Mask 尺寸: {mask.size}")

        # 执行 inpaint
        # 注意：不同版本的 diffusers 可能有不同的参数
        with torch.inference_mode():
            result = pipeline(
                prompt=request.prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=30,
                guidance_scale=7.5,
            )

        output_image = result.images[0]

        # 转换为 base64
        output_base64 = pil_to_base64(output_image)

        print(f"[Inpaint] 完成！")

        return InpaintResponse(
            success=True,
            output=output_base64
        )

    except Exception as e:
        print(f"[Inpaint] 错误: {e}")
        import traceback
        traceback.print_exc()

        return InpaintResponse(
            success=False,
            error=str(e)
        )


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    if not load_model():
        print("警告: 模型加载失败，服务可能无法正常工作")


def main():
    global gpu_id, model_path

    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 Inpaint Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--gpu", type=int, default=0, help="使用的 GPU 编号 (默认: 0)")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511",
                       help="模型路径，可以是 HF 模型名或本地路径")
    parser.add_argument("--reload", action="store_true", help="开发模式，自动重载")

    args = parser.parse_args()

    # 设置 GPU
    gpu_id = args.gpu
    model_path = args.model

    # 也可以通过环境变量限制可见的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"设置 CUDA_VISIBLE_DEVICES={args.gpu}")
    print(f"模型路径: {model_path}")

    print(f"启动服务: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print(f"使用 GPU: {args.gpu}")

    uvicorn.run(
        "qwen_inpaint_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
