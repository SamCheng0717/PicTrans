"""
数据模型定义
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum


class TextRole(Enum):
    """文字角色类型"""
    FEATURE = "feature"      # 产品特性
    SLOGAN = "slogan"        # 标语口号
    BRAND = "brand"          # 品牌名
    PRICE = "price"          # 价格
    PROMO = "promo"          # 促销信息
    UNKNOWN = "unknown"      # 未知


@dataclass
class TextFeatures:
    """文字视觉特征"""
    # 颜色 (RGB)
    text_color: Tuple[int, int, int] = (255, 255, 255)

    # 描边
    has_stroke: bool = False
    stroke_color: Optional[Tuple[int, int, int]] = None
    stroke_width: int = 0

    # 阴影
    has_shadow: bool = False
    shadow_color: Optional[Tuple[int, int, int]] = None
    shadow_offset: Tuple[int, int] = (0, 0)

    # 字体特征
    estimated_font_size: int = 24
    font_weight: str = "regular"  # thin/light/regular/medium/bold/heavy

    # 背景颜色（用于inpaint）
    background_color: Optional[Tuple[int, int, int]] = None
    background_is_gradient: bool = False


@dataclass
class TextBox:
    """文字框数据"""
    id: str
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]

    # 翻译后的文字
    translated_text: Optional[str] = None

    # 角色分类
    role: TextRole = TextRole.UNKNOWN

    # 视觉特征
    features: Optional[TextFeatures] = None

    # 处理标记
    skip: bool = False  # 是否跳过处理
    is_english: bool = False  # 是否为英文文字

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )


@dataclass
class TranslationTask:
    """翻译任务"""
    image_path: str
    source_lang: str = "zh"
    target_lang: str = "ko"

    # 可选参数
    font_name: Optional[str] = None  # 指定字体
    inpaint_mode: str = "opencv"     # opencv / lama
    quality: str = "high"            # high / medium / fast

    # 过滤选项
    skip_price: bool = True
    skip_promo: bool = True
    skip_brand: bool = False
    skip_english: bool = False  # 跳过英文文字翻译

    # 自定义跳过的文字
    skip_texts: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """处理结果"""
    success: bool
    output_path: Optional[str] = None

    # 检测到的文字
    detected_texts: List[TextBox] = field(default_factory=list)

    # 处理统计
    total_texts: int = 0
    translated_texts: int = 0
    skipped_texts: int = 0

    # 错误信息
    error_message: Optional[str] = None

    # 耗时统计（毫秒）
    ocr_time: int = 0
    translate_time: int = 0
    render_time: int = 0
    total_time: int = 0


@dataclass
class APIRequest:
    """API请求模型"""
    # 图片（base64或URL）
    image: str

    # 语言设置
    source_lang: str = "zh"
    target_lang: str = "ko"

    # 字体设置
    font_mode: str = "auto"          # auto / manual
    font_name: Optional[str] = None  # 手动指定时使用
    font_weight: str = "auto"        # auto / thin / regular / bold 等

    # 修复模式
    inpaint_mode: str = "opencv"     # opencv / lama

    # 质量设置
    quality: str = "high"            # high / medium / fast

    # 过滤设置
    skip_price: bool = True
    skip_promo: bool = True
    skip_brand: bool = False
    skip_english: bool = False  # 跳过英文文字翻译
    skip_texts: List[str] = field(default_factory=list)


@dataclass
class APIResponse:
    """API响应模型"""
    success: bool
    message: str

    # 输出
    output_path: Optional[str] = None
    output_base64: Optional[str] = None  # 可选，返回base64图片

    # 详情
    detected_texts: List[dict] = field(default_factory=list)
    translated_texts: List[dict] = field(default_factory=list)

    # 统计
    stats: dict = field(default_factory=dict)
