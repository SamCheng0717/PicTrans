"""
Qwen-Image-Edit-2511 Inpaint 客户端
用于调用 3090 服务器上的 AI 修复服务
"""
import cv2
import numpy as np
import httpx
import base64
import io
from PIL import Image
from typing import Optional


class QwenInpaintClient:
    """Qwen-Image-Edit-2511 API 客户端"""

    def __init__(self, api_url: str = "http://localhost:8765"):
        """
        初始化客户端

        Args:
            api_url: 3090 服务器的 API 地址
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = 120  # 2分钟超时（AI 修复比较慢）

        # 固定的 prompt，只做擦除和背景重建
        self.prompt = (
            "Remove all text and symbols in the masked area. "
            "Reconstruct the background naturally. "
            "Keep the original style, color, and lighting. "
            "Do not add any new elements."
        )

    def _image_to_base64(self, image: np.ndarray) -> str:
        """将 OpenCV 图像 (BGR) 转换为 base64"""
        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """将 mask (单通道) 转换为 base64"""
        pil_mask = Image.fromarray(mask)

        buffer = io.BytesIO()
        pil_mask.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _base64_to_image(self, base64_str: str) -> np.ndarray:
        """将 base64 转换回 OpenCV 图像 (BGR)"""
        image_data = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_data))

        # RGB -> BGR
        rgb_array = np.array(pil_image)
        if len(rgb_array.shape) == 2:
            # 灰度图转 BGR
            return cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
        elif rgb_array.shape[2] == 4:
            # RGBA -> BGR
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        else:
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    async def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        调用 Qwen-Image-Edit-2511 进行背景修复

        Args:
            image: 原始图像 (BGR, numpy array)
            mask: 二值 mask (白色=要擦除的区域)

        Returns:
            修复后的图像 (BGR, numpy array)
        """
        # 转换为 base64
        image_base64 = self._image_to_base64(image)
        mask_base64 = self._mask_to_base64(mask)

        # 构建请求
        payload = {
            "image": image_base64,
            "mask": mask_base64,
            "prompt": self.prompt
        }

        # 发送请求
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_url}/inpaint",
                json=payload
            )
            response.raise_for_status()

        # 解析响应
        result = response.json()

        if not result.get("success"):
            raise RuntimeError(f"Qwen inpaint failed: {result.get('error', 'Unknown error')}")

        # 转换回图像
        output_base64 = result.get("output")
        if not output_base64:
            raise RuntimeError("Qwen inpaint returned empty output")

        return self._base64_to_image(output_base64)

    def inpaint_sync(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """同步版本的 inpaint 方法"""
        import asyncio
        return asyncio.run(self.inpaint(image, mask))

    async def health_check(self) -> bool:
        """检查服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.api_url}/health")
                return response.status_code == 200
        except Exception:
            return False
