"""
背景修复模块 - 擦除原文字
支持 OpenCV（默认）和 iopaint（AI 模式）
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional

from ..config import config
from ..models.schemas import TextBox
from .iopaint_client import IOPaintClient


class Inpainter:
    """图像背景修复器"""

    def __init__(self, mode: str = "opencv"):
        """
        初始化修复器

        Args:
            mode: 修复模式 - "opencv" 或 "iopaint"
        """
        self.mode = mode
        self.opencv_radius = config.inpaint.opencv_radius
        self.opencv_method = config.inpaint.opencv_method
        self.sample_padding = config.inpaint.sample_padding
        self.blur_kernel = config.inpaint.blur_kernel
        self.mask_expand = getattr(config.inpaint, 'mask_expand', 8)

        # IOPaint 客户端（延迟初始化）
        self._iopaint_client: Optional[IOPaintClient] = None

    def inpaint(self, image: np.ndarray, text_boxes: List[TextBox]) -> np.ndarray:
        """
        修复图像中的文字区域

        Args:
            image: 原始图像 (BGR)
            text_boxes: 需要修复的文字框列表

        Returns:
            修复后的图像
        """
        if not text_boxes:
            return image.copy()

        # 先进行 bbox 聚类
        clusters = self._cluster_boxes(text_boxes)
        print(f"[Inpaint] 聚类结果: {len(text_boxes)} 个框 -> {len(clusters)} 个聚类")

        # 根据模式选择修复方法
        if self.mode == "opencv":
            return self._inpaint_opencv(image, text_boxes, clusters)
        elif self.mode == "iopaint":
            return self._inpaint_iopaint(image, text_boxes, clusters)
        else:
            print(f"[Inpaint] 未知模式 '{self.mode}'，使用默认 opencv")
            return self._inpaint_opencv(image, text_boxes, clusters)

    def _cluster_boxes(self, text_boxes: List[TextBox], y_threshold_ratio: float = None) -> List[List[TextBox]]:
        """
        聚类相邻的文字框

        规则: x轴有重叠 + y距离 < y_threshold_ratio × 平均高度 -> 合并到同一聚类

        Args:
            text_boxes: 文字框列表
            y_threshold_ratio: y方向阈值比例（None则从config读取，越小越不容易合并，0=完全不聚类）

        Returns:
            聚类后的文字框列表，每个元素是一组相邻的文字框
        """
        if not text_boxes:
            return []

        # 从配置读取默认值
        if y_threshold_ratio is None:
            y_threshold_ratio = config.inpaint.y_threshold_ratio

        # 过滤掉 skip 的框
        active_boxes = [b for b in text_boxes if not b.skip]
        if not active_boxes:
            return []

        # 如果阈值为0，完全禁用聚类，每个框独立处理
        if y_threshold_ratio == 0:
            return [[box] for box in active_boxes]

        # 计算平均高度
        avg_height = sum(b.height for b in active_boxes) / len(active_boxes)
        y_threshold = avg_height * y_threshold_ratio

        # 按 y 坐标排序
        sorted_boxes = sorted(active_boxes, key=lambda b: b.bbox[1])

        clusters = []
        current_cluster = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            # 检查是否与当前聚类中的任何框相邻
            should_merge = False

            for cluster_box in current_cluster:
                # 检查 x 轴重叠
                x_overlap = self._check_x_overlap(box.bbox, cluster_box.bbox)
                # 检查 y 距离
                y_distance = abs(box.bbox[1] - cluster_box.bbox[3])

                if x_overlap and y_distance < y_threshold:
                    should_merge = True
                    break

            if should_merge:
                current_cluster.append(box)
            else:
                clusters.append(current_cluster)
                current_cluster = [box]

        # 添加最后一个聚类
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _check_x_overlap(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """检查两个 bbox 在 x 轴是否有重叠"""
        x1_min, _, x1_max, _ = bbox1
        x2_min, _, x2_max, _ = bbox2

        # 有重叠条件：一个框的右边界 > 另一个框的左边界
        return x1_max > x2_min and x2_max > x1_min

    def _create_mask(
        self,
        image_shape: Tuple[int, int],
        text_boxes: List[TextBox],
        expand: int = None
    ) -> np.ndarray:
        """
        创建文字区域的 binary mask

        Args:
            image_shape: 图像尺寸 (h, w)
            text_boxes: 文字框列表
            expand: 扩展像素数，默认使用配置值

        Returns:
            mask: 白色=要擦除的区域，黑色=保留
        """
        if expand is None:
            expand = self.mask_expand

        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in text_boxes:
            if box.skip:
                continue

            x1, y1, x2, y2 = box.bbox

            # 向外扩展 padding，确保完全覆盖文字边缘
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(w, x2 + expand)
            y2 = min(h, y2 + expand)

            mask[y1:y2, x1:x2] = 255

        return mask

    def _create_cluster_mask(
        self,
        image_shape: Tuple[int, int],
        clusters: List[List[TextBox]],
        expand: int = None
    ) -> np.ndarray:
        """
        为聚类后的文字框创建合并的 mask

        Args:
            image_shape: 图像尺寸
            clusters: 聚类后的文字框列表

        Returns:
            合并后的 mask
        """
        if expand is None:
            expand = self.mask_expand

        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for cluster in clusters:
            # 计算聚类的边界框
            x1 = min(b.bbox[0] for b in cluster)
            y1 = min(b.bbox[1] for b in cluster)
            x2 = max(b.bbox[2] for b in cluster)
            y2 = max(b.bbox[3] for b in cluster)

            # 扩展
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(w, x2 + expand)
            y2 = min(h, y2 + expand)

            mask[y1:y2, x1:x2] = 255

        return mask

    def _inpaint_opencv(self, image: np.ndarray, text_boxes: List[TextBox], clusters: List[List[TextBox]] = None) -> np.ndarray:
        """使用背景色填充 + 边缘inpaint修复文字区域"""
        if not text_boxes:
            return image.copy()

        h, w = image.shape[:2]
        result = image.copy()
        print(f"[Inpaint] 图像尺寸: {w}x{h}, 待处理文字框: {len(text_boxes)}")

        # 分别处理渐变和非渐变背景
        gradient_boxes = []
        solid_boxes = []

        for box in text_boxes:
            if box.skip:
                continue
            if box.features and box.features.background_is_gradient:
                gradient_boxes.append(box)
            else:
                solid_boxes.append(box)

        # 第一步：处理纯色背景的文字框（直接填充）
        for box in solid_boxes:
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
            bg_color, _ = self._sample_background(image, (x1, y1, x2, y2))
            print(f"[Inpaint] 填充 '{box.text}' @ [{x1},{y1},{x2},{y2}] -> RGB{bg_color}")

            # 用背景色填充
            result[y1:y2, x1:x2] = (bg_color[2], bg_color[1], bg_color[0])  # RGB -> BGR

        # 第二步：处理渐变背景的文字框（使用inpaint保留渐变）
        if gradient_boxes:
            print(f"[Inpaint] 检测到 {len(gradient_boxes)} 个渐变背景文字框，使用智能修复")

            # 为渐变背景文字框创建mask
            gradient_mask = np.zeros((h, w), dtype=np.uint8)

            for box in gradient_boxes:
                x1, y1, x2, y2 = box.bbox
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                print(f"[Inpaint] 渐变背景 '{box.text}' @ [{x1},{y1},{x2},{y2}]")

                # 扩展mask范围，确保覆盖文字边缘
                expand = 3
                gradient_mask[max(0, y1-expand):min(h, y2+expand),
                            max(0, x1-expand):min(w, x2+expand)] = 255

            # 使用inpaint修复渐变背景的文字区域
            if np.any(gradient_mask):
                method = cv2.INPAINT_TELEA if self.opencv_method == "telea" else cv2.INPAINT_NS
                result = cv2.inpaint(result, gradient_mask, 3, method)
                print(f"[Inpaint] 渐变背景修复完成")

        # 第三步：创建边缘mask，用inpaint处理边缘过渡
        # 只对非渐变背景的文字框处理边缘
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        edge_width = 8  # 边缘宽度

        for box in solid_boxes:
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

        print(f"[Inpaint] 完成修复（背景色填充 + 渐变保留 + 边缘平滑）")
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

    def _inpaint_iopaint(
        self,
        image: np.ndarray,
        text_boxes: List[TextBox],
        clusters: List[List[TextBox]]
    ) -> np.ndarray:
        """
        使用 IOPaint AI 模型进行背景修复

        Args:
            image: 原始图像（BGR numpy array）
            text_boxes: 文字框列表
            clusters: 聚类后的文字框

        Returns:
            修复后的图像（BGR numpy array）

        Raises:
            httpx.HTTPStatusError: IOPaint API 错误
            httpx.TimeoutException: 请求超时
        """
        import asyncio
        import concurrent.futures

        # 延迟初始化 IOPaint 客户端
        if self._iopaint_client is None:
            self._iopaint_client = IOPaintClient()

        h, w = image.shape[:2]
        print(f"[Inpaint-IOPaint] 图像尺寸: {w}x{h}, 使用 AI 模式")

        # 创建 mask（使用聚类 mask 提高效果）
        if clusters:
            mask = self._create_cluster_mask(image.shape, clusters)
        else:
            mask = self._create_mask(image.shape, text_boxes)

        # 调用 IOPaint API（在新事件循环中运行，避免冲突）
        def run_in_new_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._iopaint_client.inpaint(image, mask))
            finally:
                loop.close()

        # 执行修复
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                result = future.result()

            print(f"[Inpaint-IOPaint] AI 修复完成")
            return result

        except Exception as e:
            # 用户要求：直接报错，不回退到 OpenCV
            print(f"[Inpaint-IOPaint] AI 修复失败: {e}")
            raise RuntimeError(f"IOPaint 修复失败: {e}") from e

    def inpaint_single(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """修复单个区域"""
        dummy_box = TextBox(id="temp", text="", bbox=list(bbox))
        return self.inpaint(image, [dummy_box])
