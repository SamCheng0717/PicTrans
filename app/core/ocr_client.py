"""
DeepSeek-OCR 客户端
"""
import re
import base64
import httpx
from pathlib import Path
from typing import List, Optional, Union, Tuple
from PIL import Image

from ..config import config
from ..models.schemas import TextBox, TextRole


# DeepSeek-OCR 返回归一化坐标的范围 (0-999)
OCR_NORMALIZED_SIZE = 999


class OCRClient:
    """DeepSeek-OCR API 客户端"""

    def __init__(self):
        self.api_url = config.ocr.api_url
        self.model = config.ocr.model
        self.max_tokens = config.ocr.max_tokens
        self.temperature = config.ocr.temperature
        self.timeout = config.ocr.timeout

    def _get_image_size(self, image_path: Union[str, Path]) -> Tuple[int, int]:
        """获取图片原始尺寸 (width, height)"""
        with Image.open(image_path) as img:
            return img.size

    def _calculate_scale_factors(self, original_width: int, original_height: int) -> Tuple[float, float]:
        """
        计算坐标缩放比例

        DeepSeek-OCR 返回归一化坐标 (0-999)，需要转换为实际像素坐标。
        公式：实际坐标 = 归一化坐标 * (图片尺寸 / 999)

        Returns:
            (scale_x, scale_y) - x和y方向的缩放比例
        """
        scale_x = original_width / OCR_NORMALIZED_SIZE
        scale_y = original_height / OCR_NORMALIZED_SIZE
        return (scale_x, scale_y)

    def _image_to_base64(self, image_path: Union[str, Path]) -> str:
        """将图片文件转换为base64"""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        with open(path, "rb") as f:
            image_data = f.read()

        # 根据扩展名确定MIME类型
        ext = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_types.get(ext, "image/png")

        return f"data:{mime_type};base64,{base64.b64encode(image_data).decode()}"

    def _parse_ocr_response(self, response_text: str,
                             scale_x: float = 1.0, scale_y: float = 1.0,
                             image_size: Tuple[int, int] = None) -> List[TextBox]:
        """
        解析OCR响应文本

        Args:
            response_text: OCR返回的文本，格式示例: 西骑者[[34, 160, 185, 206]] WEST[[757, 309, 854, 382]]
            scale_x: x方向缩放比例 (图片宽度 / 999)
            scale_y: y方向缩放比例 (图片高度 / 999)
            image_size: 原图尺寸 (width, height)，用于边界检查

        Returns:
            文字框列表
        """
        text_boxes = []
        img_w, img_h = image_size if image_size else (99999, 99999)

        # 正则匹配: 文字[[x1, y1, x2, y2]]
        pattern = r'([^\[\]]+)\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]'
        matches = re.findall(pattern, response_text)

        for idx, match in enumerate(matches):
            text = match[0].strip()
            # 原始OCR归一化坐标 (0-999)
            raw_bbox = [int(match[1]), int(match[2]), int(match[3]), int(match[4])]

            # 转换归一化坐标到实际像素坐标
            # 公式：实际坐标 = 归一化坐标 * (图片尺寸 / 999)
            x1 = int(raw_bbox[0] * scale_x)
            y1 = int(raw_bbox[1] * scale_y)
            x2 = int(raw_bbox[2] * scale_x)
            y2 = int(raw_bbox[3] * scale_y)

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            bbox = [x1, y1, x2, y2]

            # 动态扩展边界框，确保完整覆盖文字笔画
            bbox = self._refine_bbox(bbox, img_w, img_h)

            # 自动分类角色
            role = self._classify_role(text)

            text_box = TextBox(
                id=f"t_{idx:03d}",
                text=text,
                bbox=bbox,
                role=role,
                skip=False  # 全量翻译，不跳过
            )
            text_boxes.append(text_box)

        return text_boxes

    def _classify_role(self, text: str) -> TextRole:
        """根据文字内容分类角色（全量翻译，不过滤）"""
        # 全部标记为FEATURE，不跳过任何文字
        return TextRole.FEATURE

    def _refine_bbox(self, bbox: List[int], img_w: int, img_h: int) -> List[int]:
        """
        动态扩展边界框，确保完整覆盖文字笔画

        Args:
            bbox: 原始边界框 [x1, y1, x2, y2]
            img_w: 图像宽度
            img_h: 图像高度

        Returns:
            扩展后的边界框 [x1, y1, x2, y2]
        """
        if not config.ocr.bbox_refine_enabled:
            return bbox

        x1, y1, x2, y2 = bbox
        box_h = y2 - y1
        box_w = x2 - x1

        # 动态计算扩展像素：基于高度的比例，但限制在最小和最大值之间
        expand = int(box_h * config.ocr.bbox_expand_ratio)
        expand = max(config.ocr.bbox_expand_min, min(expand, config.ocr.bbox_expand_max))

        # 扩展边界框
        new_x1 = max(0, x1 - expand)
        new_y1 = max(0, y1 - expand)
        new_x2 = min(img_w, x2 + expand)
        new_y2 = min(img_h, y2 + expand)

        # 如果有实际扩展，输出日志
        if new_x1 != x1 or new_y1 != y1 or new_x2 != x2 or new_y2 != y2:
            print(f"[OCR-BBoxRefine] 扩展 {expand}px: [{x1},{y1},{x2},{y2}] -> [{new_x1},{new_y1},{new_x2},{new_y2}]")

        return [new_x1, new_y1, new_x2, new_y2]

    async def recognize(self, image: Union[str, Path, bytes]) -> List[TextBox]:
        """
        识别图片中的文字

        Args:
            image: 图片路径、base64字符串或字节数据

        Returns:
            识别到的文字框列表
        """
        scale_x, scale_y = 1.0, 1.0
        image_path = None
        width, height = None, None  # 原图尺寸

        # 处理不同的输入格式
        if isinstance(image, bytes):
            image_url = f"data:image/png;base64,{base64.b64encode(image).decode()}"
            # 对于字节数据，尝试获取尺寸
            import io
            try:
                with Image.open(io.BytesIO(image)) as img:
                    width, height = img.size
                    scale_x, scale_y = self._calculate_scale_factors(width, height)
            except Exception:
                pass
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                image_path = path
                image_url = self._image_to_base64(path)
                # 获取原图尺寸并计算缩放比例
                width, height = self._get_image_size(path)
                scale_x, scale_y = self._calculate_scale_factors(width, height)
                print(f"[OCR] 原图尺寸: {width}x{height}")
                print(f"[OCR] 归一化坐标(0-999) -> 像素坐标: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
            elif str(image).startswith("data:image"):
                image_url = str(image)
            else:
                raise ValueError(f"无效的图片输入: {image}")
        else:
            raise ValueError(f"不支持的图片类型: {type(image)}")

        # 构建请求
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": "<image>\n<|grounding|>OCR this image."
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        # 发送请求
        async with httpx.AsyncClient(timeout=self.timeout, verify=False) as client:
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()

        # 解析响应
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # 解析OCR结果，将归一化坐标转换为像素坐标
        image_size = (width, height) if width and height else None
        text_boxes = self._parse_ocr_response(content, scale_x, scale_y, image_size)

        # 打印转换后的坐标信息
        if text_boxes:
            print(f"[OCR] 识别到 {len(text_boxes)} 个文字框")
            for box in text_boxes[:3]:  # 打印前3个作为示例
                print(f"  - {box.text}: {box.bbox}")

        return text_boxes

    def recognize_sync(self, image: Union[str, Path, bytes]) -> List[TextBox]:
        """同步版本的识别方法"""
        import asyncio
        return asyncio.run(self.recognize(image))
