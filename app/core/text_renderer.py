"""
文字渲染器 - 自适应排版 + 描边阴影
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from pathlib import Path

from ..config import config
from ..models.schemas import TextBox, TextFeatures


class TextRenderer:
    """文字渲染器"""

    def __init__(self, target_lang: str = "zh"):
        self.font_config = config.font
        self.render_config = config.render
        self.target_lang = target_lang
        self._font_cache = {}

    def _get_font(self, weight: str, size: int) -> ImageFont.FreeTypeFont:
        """获取字体对象（带缓存）"""
        cache_key = f"{self.target_lang}_{weight}_{size}"
        if cache_key not in self._font_cache:
            font_path = self.font_config.get_font_path(weight, self.target_lang)
            if not font_path.exists():
                # 回退到默认字体
                font_path = self.font_config.get_font_path("regular", self.target_lang)
            self._font_cache[cache_key] = ImageFont.truetype(str(font_path), size)
        return self._font_cache[cache_key]

    def _calculate_text_size(
        self,
        text: str,
        font: ImageFont.FreeTypeFont
    ) -> Tuple[int, int, int]:
        """
        计算文字渲染尺寸

        Returns:
            (width, height, y_offset) - y_offset是文字顶部相对于绘制位置的偏移
        """
        # 创建临时图像用于测量
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        # bbox[1]可能是负值，表示文字顶部在指定位置上方
        return (bbox[2] - bbox[0], bbox[3] - bbox[1], bbox[1])

    def _fit_text_in_box(
        self,
        text: str,
        box_width: int,
        box_height: int,
        font_weight: str = "regular"
    ) -> Tuple[int, str, int]:
        """
        自适应文字到框内

        Returns:
            (字号, 处理后的文字, 行数)
        """
        min_size = self.render_config.min_font_size
        max_size = min(self.render_config.max_font_size, box_height)
        step = self.render_config.font_size_step

        # 策略1：尝试单行
        for size in range(max_size, min_size - 1, -step):
            font = self._get_font(font_weight, size)
            text_w, text_h, _ = self._calculate_text_size(text, font)

            if text_w <= box_width and text_h <= box_height:
                return (size, text, 1)

        # 策略2：尝试两行
        if len(text) >= 4 and self.render_config.max_lines >= 2:
            mid = len(text) // 2
            # 尝试在不同位置换行
            for split_pos in [mid, mid - 1, mid + 1, mid - 2, mid + 2]:
                if 0 < split_pos < len(text):
                    wrapped = text[:split_pos] + "\n" + text[split_pos:]
                    line_height = box_height / 2 / self.render_config.line_spacing

                    for size in range(int(line_height), min_size - 1, -step):
                        font = self._get_font(font_weight, size)

                        # 计算两行的尺寸
                        lines = wrapped.split("\n")
                        max_w = 0
                        total_h = 0
                        for line in lines:
                            w, h, _ = self._calculate_text_size(line, font)
                            max_w = max(max_w, w)
                            total_h += h * self.render_config.line_spacing

                        if max_w <= box_width and total_h <= box_height:
                            return (size, wrapped, 2)

        # 策略3：使用最小字号
        font = self._get_font(font_weight, min_size)
        return (min_size, text, 1)

    def render(
        self,
        image: np.ndarray,
        text_boxes: List[TextBox]
    ) -> np.ndarray:
        """
        在图像上渲染翻译后的文字

        Args:
            image: 背景图像 (BGR, numpy array)
            text_boxes: 文字框列表（需要已翻译且有特征）

        Returns:
            渲染后的图像
        """
        # 转换为PIL Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        for box in text_boxes:
            if box.skip or not box.translated_text:
                continue

            self._render_text_box(pil_image, draw, box)

        # 转回OpenCV格式
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return result

    def _render_text_box(
        self,
        image: Image.Image,
        draw: ImageDraw.ImageDraw,
        box: TextBox
    ):
        """渲染单个文字框"""
        x1, y1, x2, y2 = box.bbox
        box_width = x2 - x1
        box_height = y2 - y1

        # 获取特征
        features = box.features or TextFeatures()

        # 计算自适应字号和文字
        font_weight = features.font_weight
        font_size, text, num_lines = self._fit_text_in_box(
            box.translated_text,
            box_width,
            box_height,
            font_weight
        )

        font = self._get_font(font_weight, font_size)

        # 调试输出
        font_path = self.font_config.get_font_path(font_weight, self.target_lang)
        print(f"[Render] 渲染文字: '{box.translated_text}'")
        print(f"[Render] 目标语言: {self.target_lang}, 字体: {font_path}")
        print(f"[Render] 字号: {font_size}, 位置: [{x1},{y1},{x2},{y2}]")
        print(f"[Render] 文字颜色: RGB{features.text_color}, 描边: {features.has_stroke}, 描边色: {features.stroke_color}")

        # 计算文字位置（居中）
        lines = text.split("\n")
        total_height = 0
        line_sizes = []
        line_offsets = []
        for line in lines:
            w, h, y_offset = self._calculate_text_size(line, font)
            line_sizes.append((w, h))
            line_offsets.append(y_offset)
            total_height += h * self.render_config.line_spacing

        # 起始Y位置（垂直居中）
        start_y = y1 + (box_height - total_height) / 2

        # 逐行渲染
        current_y = start_y
        for i, line in enumerate(lines):
            line_w, line_h = line_sizes[i]
            y_offset = line_offsets[i]

            # 水平居中
            x = x1 + (box_width - line_w) / 2
            # 修正y位置：减去y_offset来补偿字体的ascender偏移
            y = current_y - y_offset

            # 渲染阴影
            if features.has_shadow and features.shadow_color:
                shadow_offset = features.shadow_offset or self.render_config.shadow_offset
                shadow_color = features.shadow_color
                self._draw_text_with_effects(
                    draw, line, (x + shadow_offset[0], y + shadow_offset[1]),
                    font, shadow_color, None, 0
                )

            # 渲染描边
            stroke_width = 0
            stroke_color = None
            if features.has_stroke and features.stroke_color:
                stroke_width = features.stroke_width or int(font_size * self.render_config.stroke_width_ratio)
                stroke_color = features.stroke_color

            # 渲染主文字
            self._draw_text_with_effects(
                draw, line, (x, y),
                font, features.text_color, stroke_color, stroke_width
            )

            current_y += line_h * self.render_config.line_spacing

    def _draw_text_with_effects(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        position: Tuple[float, float],
        font: ImageFont.FreeTypeFont,
        fill_color: Tuple[int, int, int],
        stroke_color: Optional[Tuple[int, int, int]] = None,
        stroke_width: int = 0
    ):
        """绘制带效果的文字"""
        x, y = position

        if stroke_width > 0 and stroke_color:
            # PIL原生描边
            draw.text(
                (x, y),
                text,
                font=font,
                fill=fill_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color
            )
        else:
            draw.text((x, y), text, font=font, fill=fill_color)

    def render_single(
        self,
        image: np.ndarray,
        text: str,
        bbox: List[int],
        features: Optional[TextFeatures] = None
    ) -> np.ndarray:
        """渲染单个文字"""
        box = TextBox(
            id="single",
            text=text,
            bbox=bbox,
            translated_text=text,
            features=features or TextFeatures()
        )
        return self.render(image, [box])
