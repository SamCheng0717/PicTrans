"""
配置管理模块
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List


# 项目根目录
BASE_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class OCRConfig:
    """DeepSeek-OCR 配置"""
    api_url: str = "https://ai.beise.com:50111/app/ai/DeepSeekOcr/vllm"
    model: str = "deepseek-ai/DeepSeek-OCR"
    max_tokens: int = 2048
    temperature: float = 0
    timeout: int = 60
    # 边界框动态扩展配置（解决部分笔画未被框选的问题）
    bbox_refine_enabled: bool = True  # 是否启用边界框精炼
    bbox_expand_ratio: float = 0.15  # 扩展比例（相对于文字框高度）
    bbox_expand_min: int = 5  # 最小扩展像素数
    bbox_expand_max: int = 20  # 最大扩展像素数


@dataclass
class TranslatorConfig:
    """DeepSeek 翻译配置"""
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    api_key: str = "sk-9ff002aa634b490687c71b2b431d9d13"
    model: str = "deepseek-chat"
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout: int = 30


@dataclass
class FontConfig:
    """字体配置"""
    fonts_dir: Path = field(default_factory=lambda: BASE_DIR / "fonts")

    # 语言字体映射: 语言代码 -> (字体目录, 字重映射)
    language_fonts: dict = field(default_factory=lambda: {
        "ko": {
            "dir": "AlibabaSansKR",
            "weights": {
                "light": "AlibabaSansKR-Regular/AlibabaSansKR-Regular.ttf",
                "regular": "AlibabaSansKR-Regular/AlibabaSansKR-Regular.ttf",
                "medium": "AlibabaSansKR-Medium/AlibabaSansKR-Medium.ttf",
                "bold": "AlibabaSansKR-Bold/AlibabaSansKR-Bold.ttf",
                "heavy": "AlibabaSansKR-Bold/AlibabaSansKR-Bold.ttf",
            },
            "default": "AlibabaSansKR-Regular/AlibabaSansKR-Regular.ttf"
        },
        "zh": {
            "dir": "MiSans/ttf",
            "weights": {
                "thin": "MiSans-Thin.ttf",
                "extralight": "MiSans-ExtraLight.ttf",
                "light": "MiSans-Light.ttf",
                "normal": "MiSans-Normal.ttf",
                "regular": "MiSans-Regular.ttf",
                "medium": "MiSans-Medium.ttf",
                "demibold": "MiSans-Demibold.ttf",
                "semibold": "MiSans-Semibold.ttf",
                "bold": "MiSans-Bold.ttf",
                "heavy": "MiSans-Heavy.ttf",
            },
            "default": "MiSans-Regular.ttf"
        },
        "default": {
            "dir": "MiSans/ttf",
            "weights": {
                "regular": "MiSans-Regular.ttf",
                "bold": "MiSans-Bold.ttf",
            },
            "default": "MiSans-Regular.ttf"
        }
    })

    def get_font_path(self, weight: str = "regular", lang: str = "zh") -> Path:
        """获取字体文件路径"""
        # 获取语言配置，如果没有则使用默认
        lang_config = self.language_fonts.get(lang, self.language_fonts["default"])

        # 获取字重对应的字体文件
        weights = lang_config["weights"]
        font_file = weights.get(weight.lower(), lang_config["default"])

        return self.fonts_dir / lang_config["dir"] / font_file


@dataclass
class InpaintConfig:
    """图像修复配置"""
    mode: str = "opencv"  # opencv / iopaint
    # OpenCV inpaint 参数
    opencv_radius: int = 5
    opencv_method: str = "telea"  # telea / ns (Navier-Stokes)
    # 背景采样参数
    sample_padding: int = 10
    blur_kernel: int = 15
    # Mask 扩展像素（防止字边残留）
    mask_expand: int = 5
    # 文字框聚类阈值（越小越不容易合并，0=完全不聚类）
    y_threshold_ratio: float = 0.5

    # IOPaint AI 修复配置
    iopaint_api_url: str = "http://192.168.103.43:8080"  # IOPaint 服务地址
    iopaint_timeout: int = 60  # 超时时间（秒），AI 修复较慢
    iopaint_model: str = "lama"  # 修复模型：lama（推荐）/ ldm / fcf
    iopaint_ldm_steps: int = 20  # LDM 推理步数
    iopaint_no_half: bool = False  # 禁用半精度（FP16）
    iopaint_size_limit: str = "2000"  # 最大边长限制（像素）


@dataclass
class RenderConfig:
    """文字渲染配置"""
    min_font_size: int = 12
    max_font_size: int = 200
    font_size_step: int = 2
    # 描边参数
    stroke_width_ratio: float = 0.08  # 描边宽度 = 字号 * ratio
    # 阴影参数
    shadow_offset: tuple = (2, 2)
    shadow_blur: int = 3
    shadow_opacity: float = 0.5
    # 行间距
    line_spacing: float = 1.2
    # 最大允许行数
    max_lines: int = 2


@dataclass
class ColorDetectionConfig:
    """颜色检测配置"""
    # 背景采样参数
    bg_sample_padding: int = 30  # 增加到30px
    bg_sample_multi_ring: bool = True  # 多环采样
    bg_sample_rings: List[int] = field(default_factory=lambda: [5, 15, 30, 50])
    bg_gradient_threshold: float = 25.0  # 渐变检测阈值（color std）

    # OTSU 增强
    use_hsv_preprocessing: bool = True  # 使用HSV预处理
    use_adaptive_threshold: bool = True  # 使用自适应阈值
    adaptive_window_size: int = 11  # 自适应阈值窗口大小

    # 边缘检测辅助
    use_edge_detection: bool = True  # 使用边缘检测辅助
    canny_low: int = 50  # Canny边缘检测低阈值
    canny_high: int = 150  # Canny边缘检测高阈值
    edge_dilate_iterations: int = 2  # 边缘膨胀迭代次数

    # K-means 聚类
    use_kmeans_fallback: bool = True  # 使用K-means作为兜底策略
    kmeans_clusters: int = 3  # K-means聚类数量
    kmeans_max_iter: int = 20  # K-means最大迭代次数

    # 对比度判断
    contrast_threshold: int = 70  # 对比度阈值（从100降低到70）
    use_perceptual_distance: bool = True  # 使用感知色差
    min_brightness_diff: int = 30  # 最小亮度差

    # 调试模式
    debug_mode: bool = True  # 启用调试模式（详细日志）
    save_intermediate: bool = False  # 保存中间结果图片


@dataclass
class AppConfig:
    """应用总配置"""
    ocr: OCRConfig = field(default_factory=OCRConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    font: FontConfig = field(default_factory=FontConfig)
    inpaint: InpaintConfig = field(default_factory=InpaintConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    color_detection: ColorDetectionConfig = field(default_factory=ColorDetectionConfig)  # 新增颜色检测配置

    # 输出目录
    output_dir: Path = field(default_factory=lambda: BASE_DIR / "output")

    # 需要过滤的文字角色（不翻译）
    skip_roles: list = field(default_factory=lambda: ["price", "promo", "brand"])

    # 支持的语言
    supported_languages: dict = field(default_factory=lambda: {
        "ko": "韩文",
        "ja": "日文",
        "en": "英文",
        "th": "泰文",
        "zh-TW": "繁体中文",
    })

    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = AppConfig()
