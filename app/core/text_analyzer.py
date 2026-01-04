"""
文字特征分析器 - 自动检测颜色、描边、阴影
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from collections import Counter

from ..models.schemas import TextBox, TextFeatures


class TextAnalyzer:
    """文字视觉特征分析器"""

    def __init__(self):
        pass

    def analyze(self, image: np.ndarray, text_box: TextBox) -> TextFeatures:
        """
        分析文字框的视觉特征

        Args:
            image: OpenCV格式的图片 (BGR)
            text_box: 文字框数据

        Returns:
            文字视觉特征
        """
        x1, y1, x2, y2 = text_box.bbox

        # 确保坐标在图片范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # 提取文字区域
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return TextFeatures()

        # 分析各项特征
        text_color = self._detect_text_color(roi)
        bg_color, is_gradient = self._detect_background(image, text_box.bbox)
        has_stroke, stroke_color, stroke_width = self._detect_stroke(roi, text_color)
        has_shadow, shadow_color, shadow_offset = self._detect_shadow(roi, text_color)
        font_size = self._estimate_font_size(text_box)
        font_weight = self._estimate_font_weight(roi)

        return TextFeatures(
            text_color=text_color,
            has_stroke=has_stroke,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            has_shadow=has_shadow,
            shadow_color=shadow_color,
            shadow_offset=shadow_offset,
            estimated_font_size=font_size,
            font_weight=font_weight,
            background_color=bg_color,
            background_is_gradient=is_gradient
        )

    def _detect_text_color(self, roi: np.ndarray) -> Tuple[int, int, int]:
        """检测文字主色调 - 基于颜色聚类"""
        if roi.size == 0:
            return (0, 0, 0)  # 默认黑色

        # 转换到灰度图进行二值化
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 使用OTSU自动阈值分离前景和背景
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算两个区域的平均亮度
        light_mask = binary > 127
        dark_mask = binary <= 127

        light_count = np.sum(light_mask)
        dark_count = np.sum(dark_mask)

        # 文字通常占据较小的区域，背景占据较大区域
        # 所以较小区域的颜色更可能是文字颜色
        if light_count > 0 and dark_count > 0:
            # 哪个区域更小，哪个更可能是文字
            if dark_count < light_count:
                # 深色区域较小 -> 深色文字在浅色背景上
                text_mask = dark_mask
            else:
                # 浅色区域较小 -> 浅色文字在深色背景上
                text_mask = light_mask
        elif dark_count > 0:
            text_mask = dark_mask
        else:
            text_mask = light_mask

        # 提取文字区域的颜色
        if np.sum(text_mask) > 10:
            colors = roi[text_mask]
            b = int(np.median(colors[:, 0]))
            g = int(np.median(colors[:, 1]))
            r = int(np.median(colors[:, 2]))
            return (r, g, b)

        # 默认返回黑色（大多数文字是深色的）
        return (0, 0, 0)

    def _detect_background(
        self,
        image: np.ndarray,
        bbox: List[int],
        padding: int = 15
    ) -> Tuple[Tuple[int, int, int], bool]:
        """检测文字区域周围的背景色"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # 扩展区域采样背景
        sample_regions = []

        # 上方区域
        if y1 - padding > 0:
            region = image[max(0, y1 - padding):y1, x1:x2]
            if region.size > 0:
                sample_regions.append(region)

        # 下方区域
        if y2 + padding < h:
            region = image[y2:min(h, y2 + padding), x1:x2]
            if region.size > 0:
                sample_regions.append(region)

        # 左侧区域
        if x1 - padding > 0:
            region = image[y1:y2, max(0, x1 - padding):x1]
            if region.size > 0:
                sample_regions.append(region)

        # 右侧区域
        if x2 + padding < w:
            region = image[y1:y2, x2:min(w, x2 + padding)]
            if region.size > 0:
                sample_regions.append(region)

        if not sample_regions:
            return ((128, 128, 128), False)

        # 合并所有采样区域
        all_pixels = np.vstack([r.reshape(-1, 3) for r in sample_regions])

        # 计算颜色均值
        mean_color = np.mean(all_pixels, axis=0).astype(int)
        bg_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))  # BGR->RGB

        # 检测是否是渐变背景（通过颜色方差判断）
        color_std = np.std(all_pixels, axis=0)
        is_gradient = np.mean(color_std) > 30

        return (bg_color, is_gradient)

    def _detect_stroke(
        self,
        roi: np.ndarray,
        text_color: Tuple[int, int, int]
    ) -> Tuple[bool, Optional[Tuple[int, int, int]], int]:
        """检测文字描边"""
        # 转换到灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀边缘以获取边缘区域
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # 检查边缘区域的颜色是否与文字主色明显不同
        edge_mask = dilated_edges > 0
        if np.sum(edge_mask) < 10:
            return (False, None, 0)

        edge_colors = roi[edge_mask]
        edge_mean = np.mean(edge_colors, axis=0)

        # 计算与文字颜色的差异
        text_bgr = (text_color[2], text_color[1], text_color[0])
        color_diff = np.sqrt(np.sum((edge_mean - np.array(text_bgr)) ** 2))

        # 如果差异足够大，认为有描边
        if color_diff > 50:
            stroke_color = (int(edge_mean[2]), int(edge_mean[1]), int(edge_mean[0]))

            # 估算描边宽度（基于边缘厚度）
            # 使用形态学操作估算
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 取最大轮廓
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                perimeter = cv2.arcLength(largest, True)
                if perimeter > 0:
                    # 粗略估算描边宽度
                    stroke_width = max(1, int(area / perimeter / 4))
                    return (True, stroke_color, min(stroke_width, 10))

            return (True, stroke_color, 2)

        return (False, None, 0)

    def _detect_shadow(
        self,
        roi: np.ndarray,
        text_color: Tuple[int, int, int]
    ) -> Tuple[bool, Optional[Tuple[int, int, int]], Tuple[int, int]]:
        """检测文字阴影"""
        # 转换到灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 计算文字亮度
        text_brightness = (text_color[0] + text_color[1] + text_color[2]) / 3

        # 如果文字是亮色，阴影通常是暗色
        if text_brightness > 127:
            # 查找暗色区域
            _, dark_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        else:
            # 文字是暗色，阴影可能是更暗的区域或发光效果
            _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # 检查暗色区域的分布
        dark_pixels = np.sum(dark_mask > 0)
        total_pixels = roi.shape[0] * roi.shape[1]

        # 如果暗色区域占比在合理范围内，可能有阴影
        dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

        if 0.05 < dark_ratio < 0.3:
            # 提取阴影颜色
            if np.sum(dark_mask > 0) > 0:
                shadow_colors = roi[dark_mask > 0]
                shadow_mean = np.mean(shadow_colors, axis=0)
                shadow_color = (int(shadow_mean[2]), int(shadow_mean[1]), int(shadow_mean[0]))

                # 默认阴影偏移
                shadow_offset = (2, 2)

                return (True, shadow_color, shadow_offset)

        return (False, None, (0, 0))

    def _estimate_font_size(self, text_box: TextBox) -> int:
        """估算字号"""
        # 基于文字框高度和文字数量估算
        height = text_box.height
        text_len = len(text_box.text)

        # 假设单行文字
        estimated_size = int(height * 0.8)

        # 如果文字较多，可能是多行或较小字号
        width = text_box.width
        if text_len > 0:
            char_width = width / text_len
            estimated_by_width = int(char_width * 1.2)
            estimated_size = min(estimated_size, estimated_by_width)

        return max(12, min(estimated_size, 200))

    def _estimate_font_weight(self, roi: np.ndarray) -> str:
        """估算字重"""
        # 转换到灰度并二值化
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算文字区域的填充率
        # 粗体字通常有更高的填充率
        text_pixels = np.sum(binary > 127) if np.mean(binary) > 127 else np.sum(binary <= 127)
        total_pixels = binary.size
        fill_ratio = text_pixels / total_pixels if total_pixels > 0 else 0

        # 使用形态学分析笔画粗细
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        stroke_thickness = np.sum(binary != eroded) / (np.sum(binary > 0) + 1)

        # 根据填充率和笔画粗细判断字重
        if fill_ratio > 0.5 or stroke_thickness > 0.3:
            return "heavy"
        elif fill_ratio > 0.4 or stroke_thickness > 0.25:
            return "bold"
        elif fill_ratio > 0.3 or stroke_thickness > 0.2:
            return "semibold"
        elif fill_ratio > 0.25:
            return "medium"
        elif fill_ratio > 0.15:
            return "regular"
        else:
            return "light"

    def analyze_all(self, image: np.ndarray, text_boxes: List[TextBox]) -> List[TextBox]:
        """分析所有文字框的视觉特征"""
        for box in text_boxes:
            if not box.skip:
                box.features = self.analyze(image, box)
        return text_boxes
