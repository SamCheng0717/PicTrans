"""
IOPaint HTTP API 客户端
用于调用 IOPaint 服务进行 AI 背景修复
"""
import io
import cv2
import numpy as np
import httpx
from PIL import Image
from typing import Optional

from ..config import config


class IOPaintClient:
    """IOPaint API 客户端"""

    def __init__(self):
        """
        初始化客户端

        配置从 config.inpaint 读取：
        - iopaint_api_url: IOPaint 服务地址
        - iopaint_timeout: 请求超时时间
        - iopaint_model: 使用的修复模型
        """
        self.api_url = config.inpaint.iopaint_api_url.rstrip('/')
        self.timeout = config.inpaint.iopaint_timeout

        # IOPaint 参数
        self.model_name = config.inpaint.iopaint_model
        self.ldm_steps = config.inpaint.iopaint_ldm_steps
        self.no_half = config.inpaint.iopaint_no_half
        self.size_limit = config.inpaint.iopaint_size_limit

        print(f"[IOPaint-Client] 初始化完成")
        print(f"  API 地址: {self.api_url}")
        print(f"  模型: {self.model_name}")
        print(f"  超时: {self.timeout}s")

    def _image_to_base64(self, image: np.ndarray) -> str:
        """
        将 OpenCV 图像（BGR numpy array）转换为 base64 字符串

        Args:
            image: BGR 格式的 numpy array, shape=(h, w, 3), dtype=uint8

        Returns:
            base64 编码的 PNG 图像字符串
        """
        import base64

        # BGR -> RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # 转换为 PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')

        # Base64 编码
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """
        将 mask（单通道灰度图）转换为 base64 字符串

        Args:
            mask: 单通道 numpy array, shape=(h, w), dtype=uint8
                  255=待修复区域（白色），0=保留区域（黑色）

        Returns:
            base64 编码的 PNG mask 字符串
        """
        import base64

        pil_mask = Image.fromarray(mask)

        buffer = io.BytesIO()
        pil_mask.save(buffer, format='PNG')

        # Base64 编码
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _base64_to_image(self, base64_str: str) -> np.ndarray:
        """
        将 base64 字符串转换回 OpenCV 图像（BGR）

        Args:
            base64_str: base64 编码的图像字符串

        Returns:
            BGR 格式的 numpy array
        """
        import base64

        # 解码 base64
        image_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(image_bytes))

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
        调用 IOPaint 进行背景修复

        Args:
            image: 原始图像（BGR, numpy array）
            mask: 二值 mask（白色=要擦除的区域, 黑色=保留）

        Returns:
            修复后的图像（BGR, numpy array）

        Raises:
            httpx.HTTPStatusError: HTTP 错误（4xx/5xx）
            httpx.TimeoutException: 请求超时
            RuntimeError: IOPaint 返回无效数据
        """
        import base64

        # 保存原始尺寸
        original_h, original_w = image.shape[:2]

        print(f"[IOPaint-Client] 开始修复: {original_w}x{original_h}")

        # 转换为 base64 字符串
        image_base64 = self._image_to_base64(image)
        mask_base64 = self._mask_to_base64(mask)

        # 构建 JSON 请求
        payload = {
            'image': image_base64,
            'mask': mask_base64,
            'ldm_steps': self.ldm_steps,
            'ldm_sampler': 'plms',
            'hd_strategy': 'Crop',
            'hd_strategy_crop_trigger_size': 800,
            'hd_strategy_crop_margin': 128,
            'hd_strategy_resize_limit': int(self.size_limit)
        }

        # 发送请求
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_url}/api/v1/inpaint",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            # 检查 HTTP 状态
            response.raise_for_status()

        # 解析响应
        if not response.content:
            raise RuntimeError("IOPaint 返回空响应")

        # 尝试解析为 JSON（可能返回 base64）或直接是图像二进制
        try:
            # 如果响应是 JSON 格式
            response_json = response.json()
            if isinstance(response_json, dict) and 'image' in response_json:
                # 从 JSON 中提取 base64 图像
                output_image = self._base64_to_image(response_json['image'])
            else:
                # 响应可能直接是 base64 字符串
                output_image = self._base64_to_image(response.text)
        except:
            # 响应是直接的图像二进制数据
            image_bytes = response.content
            pil_image = Image.open(io.BytesIO(image_bytes))
            rgb_array = np.array(pil_image)
            if len(rgb_array.shape) == 2:
                output_image = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
            elif rgb_array.shape[2] == 4:
                output_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
            else:
                output_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        # 确保输出尺寸与输入一致
        if output_image.shape[:2] != (original_h, original_w):
            print(f"[IOPaint-Client] 调整输出尺寸: {output_image.shape[1]}x{output_image.shape[0]} -> {original_w}x{original_h}")
            output_image = cv2.resize(
                output_image,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR
            )

        print(f"[IOPaint-Client] 修复完成")

        return output_image

    def inpaint_sync(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """同步版本的 inpaint 方法"""
        import asyncio
        return asyncio.run(self.inpaint(image, mask))
