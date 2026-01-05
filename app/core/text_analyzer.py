"""
文字特征分析器 - 自动检测颜色、描边、阴影
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from collections import Counter

from ..models.schemas import TextBox, TextFeatures
from ..config import config


class TextAnalyzer:
    """文字视觉特征分析器"""

    def __init__(self):
        pass

    # ==================== 颜色计算辅助函数 ====================

    @staticmethod
    def _calculate_brightness(color: Tuple[int, int, int]) -> float:
        """
        计算感知亮度（使用ITU-R BT.709公式）

        Args:
            color: RGB颜色元组

        Returns:
            亮度值 (0-255)
        """
        return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

    @staticmethod
    def _perceptual_color_distance(
        c1: Tuple[int, int, int],
        c2: Tuple[int, int, int]
    ) -> float:
        """
        计算感知色差（简化版CIEDE2000，使用RGB欧几里得距离）

        Args:
            c1: 第一个颜色(RGB)
            c2: 第二个颜色(RGB)

        Returns:
            色差值
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    @staticmethod
    def _rgb_to_hsv(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        RGB转HSV颜色空间

        Args:
            rgb: RGB颜色元组 (0-255)

        Returns:
            (H, S, V) 元组，H(0-180), S(0-255), V(0-255)
        """
        # OpenCV使用BGR格式，需要转换
        bgr = np.array([[rgb[::-1]]], dtype=np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0][0]
        return (float(h), float(s), float(v))

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

        # 先检测背景色
        bg_color, is_gradient = self._detect_background(image, text_box.bbox)

        # 再检测文字颜色（传入背景色用于对比）
        text_color = self._detect_text_color(roi, bg_color)
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

    # ==================== 多策略颜色检测 ====================

    def _detect_color_otsu_hsv(
        self,
        roi: np.ndarray,
        bg_color: Tuple[int, int, int] = None
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        策略1: HSV增强的OTSU检测

        Args:
            roi: 文字区域 (BGR)
            bg_color: 背景色 (RGB)

        Returns:
            (text_color, confidence) - 文字颜色和置信度(0-1)
        """
        color_config = config.color_detection

        if color_config.debug_mode:
            print(f"[ColorDetect] 策略1 (HSV+OTSU): 开始检测")

        # 转换到HSV空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 对V通道进行直方图均衡化，增强对比度
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

        # 转回BGR再转灰度
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        # 自适应阈值或OTSU
        if color_config.use_adaptive_threshold:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, color_config.adaptive_window_size, 2
            )
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 分离亮暗区域
        light_mask = binary > 127
        dark_mask = binary <= 127

        # 提取颜色
        light_color, dark_color = self._extract_colors_from_masks(roi, light_mask, dark_mask)

        # 选择颜色并计算置信度
        selected_color = (255, 255, 255)  # 默认白色
        confidence = 0.0

        if bg_color and light_color and dark_color:
            # 计算背景亮度
            bg_brightness = self._calculate_brightness(bg_color)

            # 使用背景亮度指导选择：
            # - 浅背景(>180) -> 文字应该是暗色(dark_mask)
            # - 深背景(<80) -> 文字应该是亮色(light_mask)
            # - 中等背景 -> 使用感知色差判断
            if bg_brightness > 180:
                # 浅背景，选择暗色作为文字颜色
                selected_color = dark_color
                # 验证：dark_color 应该比 bg_color 暗
                if self._calculate_brightness(dark_color) < bg_brightness:
                    confidence = 0.9
                else:
                    # 如果dark_color反而更亮，说明检测失败，回退到色差判断
                    light_diff = self._perceptual_color_distance(light_color, bg_color)
                    dark_diff = self._perceptual_color_distance(dark_color, bg_color)
                    if dark_diff > light_diff:
                        selected_color = dark_color
                    else:
                        selected_color = light_color
                    confidence = 0.5
            elif bg_brightness < 80:
                # 深背景，选择亮色作为文字颜色
                selected_color = light_color
                # 验证：light_color 应该比 bg_color 亮
                if self._calculate_brightness(light_color) > bg_brightness:
                    confidence = 0.9
                else:
                    # 回退到色差判断
                    light_diff = self._perceptual_color_distance(light_color, bg_color)
                    dark_diff = self._perceptual_color_distance(dark_color, bg_color)
                    if dark_diff > light_diff:
                        selected_color = dark_color
                    else:
                        selected_color = light_color
                    confidence = 0.5
            else:
                # 中等背景，使用感知色差判断
                light_diff = self._perceptual_color_distance(light_color, bg_color)
                dark_diff = self._perceptual_color_distance(dark_color, bg_color)

                brightness_diff_light = abs(self._calculate_brightness(light_color) - self._calculate_brightness(bg_color))
                brightness_diff_dark = abs(self._calculate_brightness(dark_color) - self._calculate_brightness(bg_color))

                # 综合评分
                light_score = 0.6 * light_diff + 0.4 * brightness_diff_light
                dark_score = 0.6 * dark_diff + 0.4 * brightness_diff_dark

                if dark_score > light_score:
                    selected_color = dark_color
                    confidence = min(dark_score / 100.0, 1.0)
                else:
                    selected_color = light_color
                    confidence = min(light_score / 100.0, 1.0)

        elif dark_color:
            selected_color = dark_color
            confidence = 0.5
        elif light_color:
            selected_color = light_color
            confidence = 0.5

        if color_config.debug_mode:
            print(f"[ColorDetect] 策略1 结果: RGB{selected_color}, confidence={confidence:.2f}")

        return (selected_color, confidence)

    def _extract_colors_from_masks(
        self,
        roi: np.ndarray,
        light_mask: np.ndarray,
        dark_mask: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """
        从mask中提取颜色

        Args:
            roi: 文字区域 (BGR)
            light_mask: 亮色mask
            dark_mask: 暗色mask

        Returns:
            (light_color, dark_color)
        """
        light_color = None
        dark_color = None

        if np.sum(light_mask) > 10:
            colors = roi[light_mask]
            light_color = (
                int(np.median(colors[:, 2])),
                int(np.median(colors[:, 1])),
                int(np.median(colors[:, 0]))
            )

        if np.sum(dark_mask) > 10:
            colors = roi[dark_mask]
            dark_color = (
                int(np.median(colors[:, 2])),
                int(np.median(colors[:, 1])),
                int(np.median(colors[:, 0]))
            )

        return (light_color, dark_color)

    def _detect_color_edge_based(
        self,
        roi: np.ndarray,
        bg_color: Tuple[int, int, int] = None
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        策略2: 基于边缘检测的颜色分割

        Args:
            roi: 文字区域 (BGR)
            bg_color: 背景色 (RGB)

        Returns:
            (text_color, confidence)
        """
        color_config = config.color_detection

        if color_config.debug_mode:
            print(f"[ColorDetect] 策略2 (Edge-based): 开始检测")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, color_config.canny_low, color_config.canny_high)

        # 膨胀边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=color_config.edge_dilate_iterations)

        # 反转得到潜在的文字区域
        text_mask = dilated > 0

        # 提取颜色
        if np.sum(text_mask) < 10:
            if color_config.debug_mode:
                print(f"[ColorDetect] 策略2: 边缘检测失败，像素不足")
            return ((0, 0, 0), 0.0)

        text_pixels = roi[text_mask]
        text_color_bgr = np.median(text_pixels, axis=0).astype(int)
        text_color = (int(text_color_bgr[2]), int(text_color_bgr[1]), int(text_color_bgr[0]))

        # 计算置信度
        confidence = 0.5
        if bg_color:
            color_diff = self._perceptual_color_distance(text_color, bg_color)
            brightness_diff = abs(self._calculate_brightness(text_color) - self._calculate_brightness(bg_color))
            confidence = min((color_diff + brightness_diff) / 150.0, 1.0)

        if color_config.debug_mode:
            print(f"[ColorDetect] 策略2 结果: RGB{text_color}, confidence={confidence:.2f}")

        return (text_color, confidence)

    def _detect_color_kmeans(
        self,
        roi: np.ndarray,
        bg_color: Tuple[int, int, int] = None
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        策略3: 改进的K-means聚类分析（考虑空间分布）

        根据研究文献，文字像素通常：
        1. 占比较小（<50%）
        2. 空间集中度高（方差小）
        3. 与背景有对比度

        Args:
            roi: 文字区域 (BGR)
            bg_color: 背景色 (RGB)

        Returns:
            (text_color, confidence)
        """
        color_config = config.color_detection

        if color_config.debug_mode:
            print(f"[ColorDetect] 策略3 (K-means): 开始检测")

        h, w = roi.shape[:2]

        # Reshape for k-means
        pixels = roi.reshape(-1, 3).astype(np.float32)

        # K-means聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    color_config.kmeans_max_iter, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, color_config.kmeans_clusters, None,
            criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # 分析每个聚类：大小、颜色、空间分布
        cluster_scores = []
        for i in range(color_config.kmeans_clusters):
            mask = labels.flatten() == i
            count = np.sum(mask)

            if count < 10:  # 忽略太小的聚类
                continue

            # 获取聚类颜色
            center_bgr = centers[i].astype(int)
            color_rgb = (int(center_bgr[2]), int(center_bgr[1]), int(center_bgr[0]))

            # 计算空间分布特征
            coordinates = np.argwhere(mask.reshape(h, w))
            if len(coordinates) > 0:
                # 计算重心
                centroid = np.mean(coordinates, axis=0)

                # 计算空间方差（越小越集中）
                spatial_variance = np.mean(np.sum((coordinates - centroid) ** 2, axis=1))

                # 计算到重心的平均距离
                avg_distance = np.mean(np.linalg.norm(coordinates - centroid, axis=1))

                # 归一化方差（0-1，越小越好）
                max_possible_variance = (h**2 + w**2) / 4.0
                normalized_variance = spatial_variance / max_possible_variance
            else:
                normalized_variance = 1.0

            # 计算聚类占比
            ratio = count / len(pixels)

            # 综合评分（越小越可能是文字）
            # 1. 聚类占比（文字通常<40%）
            size_score = min(ratio / 0.4, 1.0)

            # 2. 空间集中度（文字更集中）
            compactness_score = 1.0 - normalized_variance

            # 3. 颜色对比度（如果有背景色）
            contrast_score = 0.5
            if bg_color:
                color_diff = self._perceptual_color_distance(color_rgb, bg_color)
                contrast_score = min(color_diff / 100.0, 1.0)

            # 综合分数（加权）
            # 文字特征：小占比 + 高集中度 + 高对比度
            text_likelihood = (
                0.4 * (1.0 - size_score) +      # 小占比得分高
                0.4 * compactness_score +        # 高集中度得分高
                0.2 * contrast_score             # 高对比度得分高
            )

            cluster_scores.append({
                'color': color_rgb,
                'ratio': ratio,
                'text_likelihood': text_likelihood,
                'normalized_variance': normalized_variance,
                'count': count
            })

            if color_config.debug_mode:
                print(f"[ColorDetect]   聚类{i}: RGB{color_rgb}, 占比:{ratio:.1%}, "
                      f"方差:{normalized_variance:.3f}, 文字概率:{text_likelihood:.2f}")

        if not cluster_scores:
            if color_config.debug_mode:
                print(f"[ColorDetect] 策略3: 所有聚类都太小")
            return ((0, 0, 0), 0.0)

        # 按文字可能性排序
        cluster_scores.sort(key=lambda x: x['text_likelihood'], reverse=True)

        # 选择最可能的文字颜色
        best_cluster = cluster_scores[0]
        text_color = best_cluster['color']

        # 置信度 = 文字可能性
        confidence = best_cluster['text_likelihood']

        # 如果有背景色，额外验证
        if bg_color and len(cluster_scores) > 1:
            # 检查第二好的聚类
            second_best = cluster_scores[1]

            # 如果两个聚类分数接近，用对比度验证
            if abs(best_cluster['text_likelihood'] - second_best['text_likelihood']) < 0.1:
                best_diff = self._perceptual_color_distance(best_cluster['color'], bg_color)
                second_diff = self._perceptual_color_distance(second_best['color'], bg_color)

                if second_diff > best_diff * 1.2:  # 第二个对比度明显更好
                    text_color = second_best['color']
                    confidence = second_best['text_likelihood']
                    if color_config.debug_mode:
                        print(f"[ColorDetect] 策略3: 使用对比度验证，切换到聚类{second_best['color']}")

        if color_config.debug_mode:
            print(f"[ColorDetect] 策略3 结果: RGB{text_color}, confidence={confidence:.2f}")

        return (text_color, confidence)

    def _detect_text_color(self, roi: np.ndarray, bg_color: Tuple[int, int, int] = None) -> Tuple[int, int, int]:
        """
        多策略文字颜色检测（主方法）

        策略层级：
        1. HSV增强的OTSU检测（主策略）
        2. 边缘检测辅助（置信度<0.7时启用）
        3. K-means聚类分析（置信度<0.5时启用）

        Args:
            roi: 文字区域图像 (BGR)
            bg_color: 背景色 (RGB)，用于对比

        Returns:
            文字颜色 (RGB)
        """
        if roi.size == 0:
            return (0, 0, 0)

        color_config = config.color_detection

        # 策略1: HSV增强的OTSU
        text_color, confidence = self._detect_color_otsu_hsv(roi, bg_color)

        # 如果置信度高，直接返回
        if confidence > 0.7:
            if color_config.debug_mode:
                print(f"[ColorDetect] 使用策略1结果，置信度足够高")
            return text_color

        # 策略2: 边缘检测辅助
        if color_config.use_edge_detection:
            text_color_edge, confidence_edge = self._detect_color_edge_based(roi, bg_color)

            if confidence_edge > confidence:
                text_color = text_color_edge
                confidence = confidence_edge

        # 策略3: K-means聚类（如果启用且置信度仍低）
        if color_config.use_kmeans_fallback and confidence < 0.5:
            text_color_kmeans, confidence_kmeans = self._detect_color_kmeans(roi, bg_color)

            if confidence_kmeans > confidence:
                text_color = text_color_kmeans
                confidence = confidence_kmeans

        # 最终验证和调整
        if bg_color:
            text_color = self._validate_and_adjust_color(text_color, bg_color)

        if color_config.debug_mode:
            print(f"[ColorDetect] 最终选择: RGB{text_color}, confidence={confidence:.2f}")

        return text_color

    def _validate_and_adjust_color(
        self,
        text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        验证并调整颜色以确保足够的对比度

        Args:
            text_color: 文字颜色
            bg_color: 背景颜色

        Returns:
            调整后的文字颜色
        """
        color_config = config.color_detection

        # 计算感知色差和亮度差
        color_diff = self._perceptual_color_distance(text_color, bg_color)
        brightness_diff = abs(self._calculate_brightness(text_color) - self._calculate_brightness(bg_color))

        # 如果对比度不足，自动调整
        if color_diff < color_config.contrast_threshold and brightness_diff < color_config.min_brightness_diff:
            bg_brightness = self._calculate_brightness(bg_color)
            if bg_brightness > 127:
                adjusted = (0, 0, 0)  # 浅背景用黑字
            else:
                adjusted = (255, 255, 255)  # 深背景用白字

            if color_config.debug_mode:
                print(f"[ColorDetect] 对比度不足，调整: RGB{text_color} -> RGB{adjusted} "
                      f"(color_diff={color_diff:.1f}, brightness_diff={brightness_diff:.1f})")

            return adjusted

        return text_color

    def _sample_background_ring(
        self,
        image: np.ndarray,
        bbox: List[int],
        padding: int
    ) -> Optional[List]:
        """
        采样指定padding的背景环（用于多环采样）

        Args:
            image: 原始图像 (BGR)
            bbox: 文字框 [x1, y1, x2, y2]
            padding: 采样padding大小

        Returns:
            采样像素列表，如果采样失败返回None
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

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
            return None

        # 合并所有像素
        all_pixels = np.vstack([r.reshape(-1, 3) for r in sample_regions])
        return all_pixels.tolist()

    def _detect_background(
        self,
        image: np.ndarray,
        bbox: List[int],
        padding: int = None
    ) -> Tuple[Tuple[int, int, int], bool]:
        """
        多环采样背景检测（改进版）

        Args:
            image: 原始图像 (BGR)
            bbox: 文字框 [x1, y1, x2, y2]
            padding: 采样padding，默认使用配置值

        Returns:
            (bg_color, is_gradient) - 背景颜色和是否渐变
        """
        color_config = config.color_detection
        padding = padding or color_config.bg_sample_padding

        all_samples = []

        # 多环采样
        if color_config.bg_sample_multi_ring:
            if color_config.debug_mode:
                print(f"[ColorDetect] 使用多环采样: {color_config.bg_sample_rings}")

            for ring_padding in color_config.bg_sample_rings:
                samples = self._sample_background_ring(image, bbox, ring_padding)
                if samples is not None:
                    all_samples.extend(samples)
        else:
            # 单环采样（保持向后兼容）
            samples = self._sample_background_ring(image, bbox, padding)
            if samples is not None:
                all_samples = samples

        if not all_samples:
            if color_config.debug_mode:
                print(f"[ColorDetect] 背景采样失败，使用默认颜色")
            return ((128, 128, 128), False)

        # 计算统计信息
        all_pixels = np.array(all_samples)
        mean_color = np.mean(all_pixels, axis=0).astype(int)
        color_std = np.std(all_pixels, axis=0)
        avg_std = np.mean(color_std)

        # BGR -> RGB
        bg_color = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))

        # 判断是否渐变
        is_gradient = avg_std > color_config.bg_gradient_threshold

        if color_config.debug_mode:
            print(f"[ColorDetect] 背景采样: {len(all_samples)} px, "
                  f"RGB{bg_color}, std={avg_std:.1f}, gradient={is_gradient}")

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
