"""
背景修复模块 - 擦除原文字
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

from ..config import config
from ..models.schemas import TextBox


class Inpainter:
    """图像背景修复器"""

    def __init__(self, mode: str = "opencv"):
        """
        初始化修复器

        Args:
            mode: 修复模式 - "opencv" 或 "lama"
        """
        self.mode = mode
        self.opencv_radius = config.inpaint.opencv_radius
        self.opencv_method = config.inpaint.opencv_method
        self.sample_padding = config.inpaint.sample_padding
        self.blur_kernel = config.inpaint.blur_kernel

    def inpaint(self, image: np.ndarray, text_boxes: List[TextBox]) -> np.ndarray:
        """
        修复图像中的文字区域

        Args:
            image: 原始图像 (BGR)
            text_boxes: 需要修复的文字框列表

        Returns:
            修复后的图像
        """
        if self.mode == "opencv":
            return self._inpaint_opencv(image, text_boxes)
        elif self.mode == "lama":
            return self._inpaint_lama(image, text_boxes)
        else:
            return self._inpaint_opencv(image, text_boxes)

    def _create_mask(
        self,
        image_shape: Tuple[int, int],
        text_boxes: List[TextBox],
        expand: int = 3
    ) -> np.ndarray:
        """创建文字区域的mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in text_boxes:
            if box.skip:
                continue

            x1, y1, x2, y2 = box.bbox

            # 扩展区域以确保完全覆盖文字
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(w, x2 + expand)
            y2 = min(h, y2 + expand)

            mask[y1:y2, x1:x2] = 255

        return mask

    def _inpaint_opencv(self, image: np.ndarray, text_boxes: List[TextBox]) -> np.ndarray:
        """使用背景色填充 + 边缘inpaint修复文字区域"""
        if not text_boxes:
            return image.copy()

        h, w = image.shape[:2]
        result = image.copy()
        print(f"[Inpaint] 图像尺寸: {w}x{h}, 待处理文字框: {len(text_boxes)}")

        # 第一步：用背景色填充每个文字区域
        for box in text_boxes:
            if box.skip:
                continue

            x1, y1, x2, y2 = box.bbox

            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                print(f"[Inpaint] 跳过无效区域: {box.text} @ [{x1},{y1},{x2},{y2}]")
                continue

            # 采样背景色
            bg_color, is_gradient = self._sample_background(image, (x1, y1, x2, y2))
            print(f"[Inpaint] 填充 '{box.text}' @ [{x1},{y1},{x2},{y2}] -> RGB{bg_color}")

            # 用背景色填充
            result[y1:y2, x1:x2] = (bg_color[2], bg_color[1], bg_color[0])  # RGB -> BGR

        # 第二步：创建边缘mask，用inpaint处理边缘过渡
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        edge_width = 8  # 边缘宽度

        for box in text_boxes:
            if box.skip:
                continue

            x1, y1, x2, y2 = box.bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 只标记边缘区域
            # 上边缘
            edge_mask[max(0, y1-edge_width):y1+edge_width, x1:x2] = 255
            # 下边缘
            edge_mask[y2-edge_width:min(h, y2+edge_width), x1:x2] = 255
            # 左边缘
            edge_mask[y1:y2, max(0, x1-edge_width):x1+edge_width] = 255
            # 右边缘
            edge_mask[y1:y2, x2-edge_width:min(w, x2+edge_width)] = 255

        # 用inpaint处理边缘，使过渡更自然
        if np.any(edge_mask):
            method = cv2.INPAINT_TELEA if self.opencv_method == "telea" else cv2.INPAINT_NS
            result = cv2.inpaint(result, edge_mask, 5, method)

        print(f"[Inpaint] 完成修复（背景色填充 + 边缘平滑）")
        return result

    def _sample_background(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[Tuple[int, int, int], bool]:
        """采样背景颜色 - 使用更大范围和中位数"""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # 根据文字框大小动态计算采样范围
        box_h = y2 - y1
        box_w = x2 - x1
        padding = max(20, min(box_h, box_w) // 2)  # 至少20像素，最多半个框的大小

        samples = []

        # 采样四周区域（扩大范围）
        regions = [
            # 上方
            (max(0, x1 - padding//2), max(0, y1 - padding), min(w, x2 + padding//2), max(0, y1 - 2)),
            # 下方
            (max(0, x1 - padding//2), min(h, y2 + 2), min(w, x2 + padding//2), min(h, y2 + padding)),
            # 左侧
            (max(0, x1 - padding), max(0, y1 - padding//2), max(0, x1 - 2), min(h, y2 + padding//2)),
            # 右侧
            (min(w, x2 + 2), max(0, y1 - padding//2), min(w, x2 + padding), min(h, y2 + padding//2)),
        ]

        for rx1, ry1, rx2, ry2 in regions:
            if rx2 > rx1 and ry2 > ry1:
                region = image[ry1:ry2, rx1:rx2]
                if region.size > 0:
                    samples.append(region.reshape(-1, 3))

        if not samples:
            return ((128, 128, 128), False)

        all_pixels = np.vstack(samples)

        # 使用中位数代替均值，对离群值更鲁棒
        median_color = np.median(all_pixels, axis=0).astype(int)
        std_color = np.std(all_pixels, axis=0)

        # BGR -> RGB
        bg_color = (int(median_color[2]), int(median_color[1]), int(median_color[0]))

        # 判断是否是渐变
        is_gradient = np.mean(std_color) > 30

        return (bg_color, is_gradient)

    def _inpaint_lama(self, image: np.ndarray, text_boxes: List[TextBox]) -> np.ndarray:
        """
        使用LaMa模型进行修复（预留接口）

        TODO: 实现LaMa模型集成
        """
        # 暂时回退到OpenCV
        print("Warning: LaMa mode not implemented, falling back to OpenCV")
        return self._inpaint_opencv(image, text_boxes)

    def inpaint_single(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """修复单个区域"""
        dummy_box = TextBox(id="temp", text="", bbox=list(bbox))
        return self.inpaint(image, [dummy_box])
