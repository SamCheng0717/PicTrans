# PicTrans - 图片多语言翻译系统

一个基于 AI 的智能图片翻译系统，专注于电商场景的产品图片本地化。通过 OCR 识别、语义翻译、背景修复和文字渲染，实现高质量的图片多语言翻译。

## 📋 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [核心模块说明](#核心模块说明)
- [文字过滤策略](#文字过滤策略)
- [边界框处理机制](#边界框处理机制)
- [支持的语言](#支持的语言)
- [目录结构](#目录结构)
- [常见问题](#常见问题-faq)
- [技术细节](#技术细节)
- [更新日志](#更新日志)

## ✨ 功能特性

- ✅ **智能 OCR 识别** - 基于 DeepSeek-OCR 的高精度文字识别
- ✅ **动态边界框扩展** - 自动扩展边界框，确保完整覆盖文字笔画（包括"专"字头部、"利"字立刀旁等细节）
- ✅ **语义优化翻译** - 针对电商场景优化的 DeepSeek 翻译，保留专业术语
- ✅ **文字特征分析** - 自动检测颜色、描边、阴影、字体大小和字重
- ✅ **智能背景修复** - 支持 OpenCV 快速修复和 Qwen AI 智能修复双模式
- ✅ **自适应文字渲染** - 根据文字框大小自动调整字号和排版
- ✅ **灵活的过滤策略** - 可配置跳过价格、促销、品牌等特定类型文字
- ✅ **多语言支持** - 支持韩语、日语、英语、泰语、繁体中文等

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        1. 图片输入                                │
│              (文件上传 / base64 数据)                            │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   2. OCR 文字识别                                 │
│  • 调用 DeepSeek-OCR API                                         │
│  • 返回归一化坐标 (0-999)                                         │
│  • 转换为实际像素坐标                                             │
│  • ⭐ 动态扩展边界框                                              │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   3. 文字过滤                                     │
│  • 根据 TextRole 分类                                            │
│  • 应用 API 过滤选项:                                             │
│    - skip_price: 跳过价格                                         │
│    - skip_promo: 跳过促销信息                                     │
│    - skip_brand: 跳过品牌名                                       │
│    - skip_texts: 自定义跳过列表                                   │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   4. 特征分析                                     │
│  • 检测文字颜色 (与背景对比度)                                     │
│  • 检测描边 (边缘颜色和宽度)                                       │
│  • 检测阴影 (颜色和偏移)                                          │
│  • 估算字体大小和字重                                             │
│  • 检测背景色 (是否渐变)                                          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   5. 文字翻译                                     │
│  • 调用 DeepSeek 翻译 API                                         │
│  • 电商语义优化 (保留专业术语)                                     │
│  • 批量翻译提升效率                                               │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   6. 背景修复 (Inpainting)                        │
│  • 文字框聚类 (合并相邻区域)                                       │
│  • 创建 mask (扩展 8px 防止残留)                                  │
│  • 两种模式:                                                      │
│    - OpenCV: 背景色填充 + 边缘平滑                                │
│    - Qwen AI: 智能修复                                           │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   7. 文字渲染                                     │
│  • 自适应字号 (根据框大小)                                        │
│  • 自动换行 (最多2行)                                             │
│  • 居中对齐                                                       │
│  • 应用视觉特征 (颜色、描边、阴影)                                 │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   8. 输出结果                                     │
│  • 保存到 output 目录                                             │
│  • 返回处理统计 (OCR时间、翻译时间、渲染时间)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd PicTrans

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API 密钥（编辑 app/config.py）
# - DeepSeek-OCR API
# - DeepSeek 翻译 API

# 4. 启动服务
python run.py
```

服务将在 `http://localhost:5000` 启动。

### 快速测试

```bash
# 基础翻译测试
curl -X POST http://localhost:5000/api/translate \
  -F "image=@test.jpg" \
  -F "target_lang=ko"

# 查看输出结果
ls output/
```

## 📚 API 文档

### 端点列表

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/translate` | POST | 图片翻译（核心端点） |
| `/api/output/<filename>` | GET | 获取输出文件 |
| `/api/languages` | GET | 支持的语言列表 |

### 核心端点：`/api/translate`

#### 请求方式

支持两种输入方式：

1. **multipart/form-data** (文件上传)
2. **application/json** (base64 数据)

#### 请求参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `image` | File/String | - | 图片文件（上传）或 base64 数据 |
| `source_lang` | String | `zh` | 源语言代码 |
| `target_lang` | String | `ko` | 目标语言代码 |
| `inpaint_mode` | String | `opencv` | 背景修复模式：`opencv` 或 `qwen` |
| `skip_price` | Boolean | `true` | 是否跳过价格翻译 |
| `skip_promo` | Boolean | `true` | 是否跳过促销信息翻译 |
| `skip_brand` | Boolean | `false` | 是否跳过品牌名翻译 |
| `skip_texts` | Array | `[]` | 自定义跳过的文字列表 |

#### 响应格式

```json
{
  "success": true,
  "message": "处理成功",
  "output_path": "E:\\PicTrans\\output\\image_ko_20240105_123456.jpg",
  "detected_texts": [
    {
      "id": "t_000",
      "text": "入会享专属福利",
      "bbox": [100, 150, 400, 200],
      "role": "feature"
    },
    {
      "id": "t_001",
      "text": "¥299",
      "bbox": [300, 250, 400, 280],
      "role": "price"
    }
  ],
  "translated_texts": [
    {
      "id": "t_000",
      "original": "入会享专属福利",
      "translated": "가입 시 전용 혜택"
    }
  ],
  "stats": {
    "total_texts": 2,
    "translated_count": 1,
    "skipped_count": 1,
    "ocr_time_ms": 3200,
    "translate_time_ms": 1500,
    "render_time_ms": 800,
    "total_time_ms": 5500
  }
}
```

### 使用示例

#### 示例 1：基础翻译（将中文翻译为韩语）

```bash
curl -X POST http://localhost:5000/api/translate \
  -F "image=@product.jpg" \
  -F "target_lang=ko"
```

#### 示例 2：翻译品牌名（默认跳过品牌，这里强制翻译）

```bash
curl -X POST http://localhost:5000/api/translate \
  -F "image=@product.jpg" \
  -F "target_lang=ko" \
  -F "skip_brand=false"
```

#### 示例 3：使用 AI 修复模式（适合复杂背景）

```bash
curl -X POST http://localhost:5000/api/translate \
  -F "image=@product.jpg" \
  -F "target_lang=ja" \
  -F "inpaint_mode=qwen"
```

#### 示例 4：自定义跳过特定文字

```bash
curl -X POST http://localhost:5000/api/translate \
  -F "image=@product.jpg" \
  -F "target_lang=en" \
  -F 'skip_texts=["二维码","扫码","关注我们"]'
```

#### 示例 5：使用 Python 客户端

```python
import requests
import json

# 读取图片
with open('product.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/translate',
        files={'image': f},
        data={
            'target_lang': 'ko',
            'skip_price': 'true',
            'skip_promo': 'true',
            'skip_brand': 'false'
        }
    )

# 解析响应
result = response.json()

if result['success']:
    print(f"✅ 翻译成功!")
    print(f"📁 输出路径: {result['output_path']}")
    print(f"📊 统计信息:")
    print(f"   - 识别文字数: {result['stats']['total_texts']}")
    print(f"   - 翻译文字数: {result['stats']['translated_count']}")
    print(f"   - 跳过文字数: {result['stats']['skipped_count']}")
    print(f"   - 总耗时: {result['stats']['total_time_ms']}ms")

    # 打印翻译结果
    for item in result['translated_texts']:
        print(f"   {item['original']} → {item['translated']}")
else:
    print(f"❌ 翻译失败: {result['message']}")
```

#### 示例 6：使用 base64 数据

```python
import base64
import requests

# 读取并编码图片
with open('product.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# 发送请求
response = requests.post(
    'http://localhost:5000/api/translate',
    json={
        'image': f'data:image/jpeg;base64,{image_data}',
        'target_lang': 'ko',
        'inpaint_mode': 'opencv'
    }
)

result = response.json()
print(result['output_path'])
```

## ⚙️ 配置说明

所有配置位于 `app/config.py` 文件中。

### OCR 配置

```python
@dataclass
class OCRConfig:
    # DeepSeek-OCR API 配置
    api_url: str = "https://ai.beise.com:50111/app/ai/DeepSeekOcr/vllm"
    model: str = "deepseek-ai/DeepSeek-OCR"
    max_tokens: int = 2048
    temperature: float = 0
    timeout: int = 60

    # 边界框动态扩展配置（解决部分笔画未被框选的问题）
    bbox_refine_enabled: bool = True    # 是否启用边界框精炼
    bbox_expand_ratio: float = 0.15     # 扩展比例（相对于文字框高度）
    bbox_expand_min: int = 5            # 最小扩展像素数
    bbox_expand_max: int = 20           # 最大扩展像素数
```

**参数说明**：
- `bbox_refine_enabled`: 启用后自动扩展 OCR 返回的边界框，确保覆盖完整文字
- `bbox_expand_ratio`: 扩展比例，相对于文字框高度（默认 15%）
- `bbox_expand_min`: 最小扩展像素，即使文字很小也至少扩展这么多（默认 5px）
- `bbox_expand_max`: 最大扩展像素，即使文字很大也最多扩展这么多（默认 20px）

**扩展计算公式**：
```python
expand = int(文字框高度 × bbox_expand_ratio)
expand = max(bbox_expand_min, min(expand, bbox_expand_max))
```

### 翻译配置

```python
@dataclass
class TranslatorConfig:
    api_url: str = "https://api.deepseek.com/v1/chat/completions"
    api_key: str = "sk-your-api-key"  # 需要替换为实际的 API 密钥
    model: str = "deepseek-chat"
    max_tokens: int = 1024
    temperature: float = 0.3           # 翻译随机度，越低越确定
    timeout: int = 30
```

### 背景修复配置

```python
@dataclass
class InpaintConfig:
    mode: str = "opencv"               # 修复模式：opencv / qwen
    opencv_radius: int = 5             # OpenCV 修复半径
    opencv_method: str = "telea"       # 修复算法：telea / ns
    mask_expand: int = 8               # Mask 扩展像素（防止字边残留）

    # Qwen-Image-Edit-2511 参数（需要单独部署）
    qwen_api_url: str = "http://localhost:8765"
    qwen_prompt: str = "Remove all text and symbols in the masked area. Reconstruct the background naturally."
```

**模式对比**：
| 模式 | 速度 | 效果 | 适用场景 |
|------|------|------|----------|
| OpenCV | 快 | 良好 | 纯色背景、简单背景 |
| Qwen AI | 慢 | 优秀 | 复杂背景、纹理背景 |

### 渲染配置

```python
@dataclass
class RenderConfig:
    min_font_size: int = 12            # 最小字号
    max_font_size: int = 200           # 最大字号
    font_size_step: int = 2            # 字号调整步长
    line_spacing: float = 1.2          # 行间距
    max_lines: int = 2                 # 最大允许行数
```

### 字体配置

```python
@dataclass
class FontConfig:
    fonts_dir: Path = field(default_factory=lambda: BASE_DIR / "fonts")

    language_fonts: dict = field(default_factory=lambda: {
        "ko": {  # 韩语
            "dir": "AlibabaSansKR",
            "weights": {
                "regular": "AlibabaSansKR-Regular/AlibabaSansKR-Regular.ttf",
                "bold": "AlibabaSansKR-Bold/AlibabaSansKR-Bold.ttf",
            }
        },
        "zh": {  # 中文
            "dir": "MiSans/ttf",
            "weights": {
                "regular": "MiSans-Regular.ttf",
                "bold": "MiSans-Bold.ttf",
            }
        },
        # ... 更多语言配置
    })
```

## 🔧 核心模块说明

### 1. OCR 模块 (`app/core/ocr_client.py`)

**职责**：使用 DeepSeek-OCR API 识别图片中的文字

**关键功能**：
- 调用 DeepSeek-OCR API 进行异步识别
- 归一化坐标转换 (0-999 → 实际像素)
- 动态边界框扩展（确保完整覆盖文字笔画）
- 文字角色分类

**关键方法**：
```python
async def recognize(image) -> List[TextBox]:
    """异步 OCR 识别"""

def _parse_ocr_response(response_text, scale_x, scale_y, image_size) -> List[TextBox]:
    """解析 OCR 响应，转换坐标"""

def _refine_bbox(bbox, img_w, img_h) -> List[int]:
    """动态扩展边界框"""
```

**坐标转换公式**：
```python
# DeepSeek-OCR 返回归一化坐标 (0-999)
scale_x = 图片宽度 / 999
scale_y = 图片高度 / 999

# 转换为实际像素坐标
x1 = int(raw_x1 * scale_x)
y1 = int(raw_y1 * scale_y)
x2 = int(raw_x2 * scale_x)
y2 = int(raw_y2 * scale_y)
```

### 2. 文字分析模块 (`app/core/text_analyzer.py`)

**职责**：分析文字的视觉特征

**检测内容**：
- 文字颜色（与背景对比度分析）
- 描边（边缘颜色和宽度）
- 阴影（颜色和偏移）
- 字体大小（基于文字框估算）
- 字重（light/regular/medium/bold/heavy）
- 背景色（是否渐变）

**关键方法**：
```python
def analyze(image, text_box) -> TextFeatures:
    """分析单个文字框的视觉特征"""

def _detect_text_color(roi, bg_color) -> Tuple[int, int, int]:
    """检测文字主色调"""

def _detect_stroke(roi, text_color) -> Tuple[bool, Optional[Tuple], int]:
    """检测文字描边"""

def _detect_shadow(roi, text_color) -> Tuple[bool, Optional[Tuple], Tuple]:
    """检测文字阴影"""
```

### 3. 翻译模块 (`app/core/translator.py`)

**职责**：使用 DeepSeek API 翻译文字

**特点**：
- 电商语义优化翻译
- 批量翻译提升效率
- 保留专业术语和品牌名

**关键方法**：
```python
async def translate_boxes(text_boxes, target_lang) -> List[TextBox]:
    """批量翻译文字框"""

def _build_prompt(texts, target_lang) -> str:
    """构建电商翻译提示词"""
```

**翻译提示词示例**：
```
你是专业的电商翻译专家。请将以下产品特性文字翻译为{target_lang}。
要求：
1. 保持简洁，符合电商产品描述风格
2. 专业术语保留原文（如材质、工艺等）
3. 突出产品卖点

文字列表：
1. 透气防水
2. 轻便舒适
...
```

### 4. 背景修复模块 (`app/core/inpainter.py`)

**职责**：擦除原文字并修复背景

**两种模式**：

#### OpenCV 模式（默认）
- 背景色填充
- 边缘平滑处理
- 适合纯色背景

#### Qwen AI 模式
- AI 智能修复
- 自然重建背景
- 适合复杂背景

**关键功能**：
- 文字框聚类（合并相邻区域）
- Mask 扩展（8px 防止残留）
- 背景色采样（中位数方法）

**关键方法**：
```python
def inpaint(image, text_boxes) -> np.ndarray:
    """修复图像中的文字区域"""

def _cluster_boxes(text_boxes, y_threshold_ratio) -> List[List[TextBox]]:
    """聚类相邻的文字框"""

def _create_mask(image_shape, text_boxes, expand) -> np.ndarray:
    """创建文字区域的 binary mask"""

def _sample_background(image, bbox) -> Tuple[Tuple[int, int, int], bool]:
    """采样背景颜色"""
```

### 5. 文字渲染模块 (`app/core/text_renderer.py`)

**职责**：在背景图片上渲染翻译后的文字

**特点**：
- 自适应字号（根据文字框大小）
- 自动换行（最多2行）
- 居中对齐（水平和垂直）
- 应用视觉特征（颜色、描边、阴影）

**关键方法**：
```python
def render(image, text_boxes) -> np.ndarray:
    """在图像上渲染翻译后的文字"""

def _fit_text_in_box(text, box_width, box_height, font_weight) -> Tuple[int, str, int]:
    """自适应文字到框内"""

def _render_text_box(image, draw, box):
    """渲染单个文字框"""
```

### 6. 处理流程 (`app/api/pipeline.py`)

**职责**：协调整个翻译流程

**完整流程**：
```python
def process(image_path, task) -> ProcessingResult:
    # 1. 加载图片
    image = cv2.imread(image_path)

    # 2. OCR 识别
    text_boxes = await ocr_client.recognize(image)

    # 3. 过滤文字框
    text_boxes = _filter_boxes(text_boxes, task)

    # 4. 分析特征
    text_boxes = text_analyzer.analyze_all(image, text_boxes)

    # 5. 翻译
    text_boxes = await translator.translate_boxes(text_boxes, task.target_lang)

    # 6. 擦除原文字
    inpainted = inpainter.inpaint(image, text_boxes)

    # 7. 渲染新文字
    result = renderer.render(inpainted, text_boxes)

    # 8. 保存结果
    output_path = save_result(result, task.target_lang)

    return ProcessingResult(...)
```

## 🎯 文字过滤策略

### TextRole 枚举

```python
class TextRole(Enum):
    FEATURE = "feature"      # 产品特性（如 "透气"、"防水"）
    SLOGAN = "slogan"        # 标语口号（如 "Just Do It"）
    BRAND = "brand"          # 品牌名（如 "Nike"）
    PRICE = "price"          # 价格（如 "¥299"）
    PROMO = "promo"          # 促销信息（如 "限时5折"）
    UNKNOWN = "unknown"      # 未知类型
```

### 过滤规则

| 角色 | API 参数 | 默认跳过 | 说明 |
|------|---------|---------|------|
| PRICE | `skip_price` | ✅ True | 跳过价格翻译（价格不需要翻译） |
| PROMO | `skip_promo` | ✅ True | 跳过促销信息（如"限时折扣"通常不需要翻译） |
| BRAND | `skip_brand` | ❌ False | 保留品牌名翻译（品牌名通常需要音译） |
| FEATURE | - | ❌ False | 保留产品特性翻译 |
| SLOGAN | - | ❌ False | 保留标语翻译 |

### 过滤逻辑

```python
def _filter_boxes(text_boxes, task):
    for box in text_boxes:
        # 根据角色过滤
        if task.skip_price and box.role == TextRole.PRICE:
            box.skip = True
        elif task.skip_promo and box.role == TextRole.PROMO:
            box.skip = True
        elif task.skip_brand and box.role == TextRole.BRAND:
            box.skip = True

        # 根据自定义文字过滤
        if box.text in task.skip_texts:
            box.skip = True

    return text_boxes
```

### 自定义过滤

通过 `skip_texts` 参数指定要跳过的文字列表：

```bash
curl -X POST http://localhost:5000/api/translate \
  -F "image=@product.jpg" \
  -F "target_lang=ko" \
  -F 'skip_texts=["二维码","扫码","关注我们","客服热线"]'
```

这些文字将：
- ❌ 不会被翻译
- ❌ 不会被擦除
- ✅ 保持原样

## 🔍 边界框处理机制

### 问题说明

DeepSeek-OCR 返回的边界框可能不完整，部分笔画没有被覆盖。例如：
- **"专"字的头部**没有被框选
- **"利"字的立刀旁**没有被框选
- 其他笔画复杂的文字也可能存在类似问题

这会导致：
- 背景修复时文字边缘残留
- 影响翻译效果和视觉质量

### 解决方案：动态扩展

#### 扩展算法

```python
def _refine_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    box_h = y2 - y1

    # 动态计算扩展像素
    expand = int(box_h × 0.15)  # 基于高度的比例
    expand = max(5, min(expand, 20))  # 限制在 5-20px 之间

    # 四个方向扩展
    new_x1 = max(0, x1 - expand)
    new_y1 = max(0, y1 - expand)
    new_x2 = min(img_w, x2 + expand)
    new_y2 = min(img_h, y2 + expand)

    return [new_x1, new_y1, new_x2, new_y2]
```

#### 扩展效果示例

```
原始边界框：[100, 150, 400, 200]
文字框高度：50px
扩展计算：
  - expand = 50 × 0.15 = 7.5
  - expand = max(5, min(7.5, 20)) = 7px
扩展后：[93, 143, 407, 207]
```

#### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bbox_refine_enabled` | `True` | 是否启用边界框精炼 |
| `bbox_expand_ratio` | `0.15` | 扩展比例（相对于文字框高度） |
| `bbox_expand_min` | `5` | 最小扩展像素数 |
| `bbox_expand_max` | `20` | 最大扩展像素数 |

#### 调整扩展范围

如果发现某些文字仍然没有被完全覆盖，可以调整配置：

```python
# app/config.py

# 更激进的扩展
bbox_expand_ratio: float = 0.20   # 提高到 20%
bbox_expand_min: int = 8          # 最小 8px
bbox_expand_max: int = 30         # 最大 30px
```

如果扩展太多影响效果，可以降低：

```python
# 更保守的扩展
bbox_expand_ratio: float = 0.10   # 降低到 10%
bbox_expand_min: int = 3          # 最小 3px
bbox_expand_max: int = 15         # 最大 15px
```

#### 禁用扩展

如果不需要这个功能：

```python
bbox_refine_enabled: bool = False
```

#### 日志输出

启用后，系统会输出扩展日志：

```
[OCR-BBoxRefine] 扩展 15px: [100,200,250,280] -> [85,185,265,295]
[OCR-BBoxRefine] 扩展 8px: [300,400,450,430] -> [292,392,458,438]
```

## 🌍 支持的语言

| 语言代码 | 语言名称 | 字体 | 支持状态 |
|---------|---------|------|----------|
| `ko` | 韩语 | AlibabaSansKR | ✅ 完整支持 |
| `ja` | 日语 | Noto Sans JP | ✅ 完整支持 |
| `en` | 英语 | MiSans | ✅ 完整支持 |
| `th` | 泰语 | Noto Sans Thai | ✅ 完整支持 |
| `zh-TW` | 繁体中文 | MiSans | ✅ 完整支持 |
| `zh` | 简体中文 | MiSans | ✅ 完整支持 |

### 添加新语言

如果需要添加新的语言支持：

1. **下载字体文件**
   ```bash
   # 将字体文件放入 fonts/ 目录
   fonts/
   └── YourFont/
       └── YourFont-Regular.ttf
   ```

2. **修改配置文件** (`app/config.py`)
   ```python
   supported_languages: dict = field(default_factory=lambda: {
       # ... 现有语言
       "fr": "法语",  # 添加新语言
   })

   language_fonts: dict = field(default_factory=lambda: {
       # ... 现有语言配置
       "fr": {
           "dir": "YourFont",
           "weights": {
               "regular": "YourFont-Regular.ttf",
               "bold": "YourFont-Bold.ttf",
           },
           "default": "YourFont-Regular.ttf"
       }
   })
   ```

3. **测试翻译**
   ```bash
   curl -X POST http://localhost:5000/api/translate \
     -F "image=@test.jpg" \
     -F "target_lang=fr"
   ```

## 📁 目录结构

```
PicTrans/
├── app/
│   ├── api/
│   │   ├── routes.py              # API 路由定义
│   │   └── pipeline.py            # 核心处理流程
│   ├── core/
│   │   ├── ocr_client.py          # OCR 客户端
│   │   ├── translator.py          # 翻译客户端
│   │   ├── text_analyzer.py       # 文字特征分析器
│   │   ├── inpainter.py           # 背景修复器
│   │   └── text_renderer.py       # 文字渲染器
│   ├── models/
│   │   └── schemas.py             # 数据模型定义
│   ├── config.py                  # 全局配置文件
│   └── __init__.py
│
├── fonts/                         # 字体目录
│   ├── AlibabaSansKR/             # 韩语字体
│   ├── MiSans/                    # 中文字体
│   └── ...                        # 其他语言字体
│
├── input/                         # 输入图片目录（可选）
├── output/                        # 输出图片目录（自动创建）
│   └── image_ko_20240105_123456.jpg
│
├── tests/                         # 测试目录
│   └── test_ocr_visual.py         # OCR 可视化测试
│
├── run.py                         # 应用入口文件
├── requirements.txt               # Python 依赖列表
├── README.md                      # 项目文档
└── .gitignore
```

### 关键文件说明

| 文件 | 行数 | 说明 |
|------|------|------|
| `run.py` | ~50 | Flask 应用启动入口 |
| `app/api/routes.py` | ~150 | API 端点定义和请求处理 |
| `app/api/pipeline.py` | ~200 | 核心翻译流程协调 |
| `app/core/ocr_client.py` | ~250 | OCR 识别和边界框处理 |
| `app/core/translator.py` | ~150 | 翻译 API 调用和提示词构建 |
| `app/core/text_analyzer.py` | ~340 | 文字视觉特征分析 |
| `app/core/inpainter.py` | ~380 | 背景修复（OpenCV + Qwen） |
| `app/core/text_renderer.py` | ~220 | 文字渲染和排版 |
| `app/config.py` | ~160 | 全局配置管理 |

## ❓ 常见问题 (FAQ)

### Q1: 如何调整边界框扩展范围？

**问题**：某些文字仍然没有被完全覆盖

**解决方案**：修改 `app/config.py` 中的配置

```python
# 更激进的扩展（确保覆盖所有笔画）
bbox_expand_ratio: float = 0.20   # 提高到 20%
bbox_expand_min: int = 8          # 最小 8px
bbox_expand_max: int = 30         # 最大 30px
```

### Q2: 如何禁用边界框扩展？

**问题**：扩展太多影响了效果

**解决方案**：

```python
# 方案 1：完全禁用
bbox_refine_enabled: bool = False

# 方案 2：更保守的扩展
bbox_expand_ratio: float = 0.10   # 降低到 10%
bbox_expand_min: int = 3          # 最小 3px
bbox_expand_max: int = 15         # 最大 15px
```

### Q3: OpenCV 和 Qwen 修复模式的区别？

| 特性 | OpenCV | Qwen AI |
|------|--------|---------|
| **速度** | 快（~1秒） | 慢（~5-10秒） |
| **效果** | 良好 | 优秀 |
| **适用场景** | 纯色背景、简单背景 | 复杂背景、纹理背景 |
| **资源消耗** | 低 | 高（需要 GPU） |
| **依赖** | 无需额外部署 | 需要部署 Qwen 服务 |

**推荐**：
- 默认使用 OpenCV 模式（快速）
- 复杂背景使用 Qwen AI 模式（高质量）

### Q4: 如何添加新的语言支持？

**步骤**：

1. **下载字体文件**
   ```bash
   mkdir -p fonts/NewLanguage
   # 将字体文件复制到该目录
   ```

2. **修改配置** (`app/config.py`)
   ```python
   supported_languages: dict = field(default_factory=lambda: {
       "new_lang": "新语言名称",
   })

   language_fonts: dict = field(default_factory=lambda: {
       "new_lang": {
           "dir": "NewLanguage",
           "weights": {
               "regular": "NewLanguage-Regular.ttf",
           },
           "default": "NewLanguage-Regular.ttf"
       }
   })
   ```

3. **测试**
   ```bash
   curl -X POST http://localhost:5000/api/translate \
     -F "image=@test.jpg" \
     -F "target_lang=new_lang"
   ```

### Q5: 为什么某些文字没有被翻译？

**可能原因**：

1. **文字被标记为 skip=True**
   ```
   [OCR] 识别到 5 个文字框
   [Filter] 跳过文字: ¥299 (role=price)
   [Filter] 跳过文字: 限时5折 (role=promo)
   ```
   **解决方案**：修改 API 参数，设置 `skip_price=false` 或 `skip_promo=false`

2. **OCR 没有识别到该文字**
   ```
   [OCR] 识别到 3 个文字框
   # 实际有 4 个文字，但漏掉了 1 个
   ```
   **解决方案**：
   - 检查图片质量
   - 提高图片分辨率
   - 调整 OCR 参数

3. **文字在自定义过滤列表中**
   ```
   [Filter] 跳过文字: 二维码 (在 skip_texts 中)
   ```
   **解决方案**：从 `skip_texts` 列表中移除该文字

### Q6: 如何查看调试信息？

**方法 1：查看控制台日志**

启动时会输出详细日志：

```
[OCR] 原图尺寸: 1000x800
[OCR] 归一化坐标(0-999) -> 像素坐标: scale_x=1.0010, scale_y=0.8008
[OCR] 识别到 5 个文字框
[OCR-BBoxRefine] 扩展 15px: [100,200,250,280] -> [85,185,265,295]
[Filter] 跳过文字: ¥299 (role=price)
[Analyzer] 检测到文字颜色: RGB(0,0,0)
[Inpaint] 聚类结果: 5 个框 -> 3 个聚类
[Render] '入会享专属福利' @ [100,150,400,200] 字号:18 颜色:RGB(0,0,0)
```

**方法 2：使用 API 响应中的统计信息**

```json
{
  "stats": {
    "ocr_time_ms": 3200,        # OCR 耗时
    "translate_time_ms": 1500,  # 翻译耗时
    "render_time_ms": 800,      # 渲染耗时
    "total_time_ms": 5500       # 总耗时
  }
}
```

**方法 3：使用 OCR 可视化测试**

```bash
# 生成 OCR 识别结果的可视化图片
python tests/test_ocr_visual.py input/test.jpg

# 输出：input/test_ocr_boxes.jpg
# 包含识别的文字框和文字内容
```

### Q7: 如何处理透明背景的图片？

**问题**：透明背景图片修复后变成黑色或白色

**解决方案**：
1. 将透明背景转换为白色背景
2. 或者使用 Qwen AI 模式（更好处理透明背景）

```python
from PIL import Image

# 转换透明背景为白色
img = Image.open('input.png')
if img.mode in ('RGBA', 'LA'):
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img.save('input.jpg', 'JPEG')
```

### Q8: 翻译速度太慢怎么办？

**可能原因**：
1. OCR API 响应慢
2. 翻译 API 响应慢
3. 使用了 Qwen AI 修复模式

**解决方案**：

1. **使用 OpenCV 模式**（默认）
   ```bash
   curl -X POST http://localhost:5000/api/translate \
     -F "image=@product.jpg" \
     -F "inpaint_mode=opencv"  # 使用快速模式
   ```

2. **优化 API 超时设置**
   ```python
   # app/config.py
   timeout: int = 30  # 降低超时时间
   ```

3. **批量处理时使用异步**
   ```python
   import asyncio
   import aiohttp

   async def batch_translate(images):
       tasks = [translate_image(img) for img in images]
       results = await asyncio.gather(*tasks)
       return results
   ```

### Q9: 如何提高翻译质量？

**建议**：

1. **提供高质量的输入图片**
   - 分辨率至少 1000px
   - 文字清晰可读
   - 背景不要太复杂

2. **调整翻译参数**
   ```python
   # app/config.py
   temperature: float = 0.1  # 降低随机度，提高确定性
   ```

3. **优化提示词**
   ```python
   # app/core/translator.py
   prompt = f"""
   你是专业的电商翻译专家。
   请将以下文字翻译为{target_lang}。

   要求：
   1. 保持简洁，符合电商风格
   2. 专业术语保留原文
   3. 突出产品卖点

   文字：{texts}
   """
   ```

### Q10: 如何部署到生产环境？

**建议**：

1. **使用 Gunicorn 或 uWSGI**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 run:app
   ```

2. **使用 Nginx 反向代理**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **配置日志和监控**
   ```python
   import logging

   logging.basicConfig(
       filename='app.log',
       level=logging.INFO,
       format='%(asctime)s %(levelname)s %(message)s'
   )
   ```

4. **设置环境变量**
   ```bash
   export DEEPSEEK_API_KEY="sk-..."
   export QWEN_API_URL="http://qwen-server:8765"
   ```

## 🔬 技术细节

### 坐标系统

**DeepSeek-OCR 坐标系统**：
- 使用归一化坐标（0-999）
- 无论图片实际尺寸如何，坐标都在这个范围内
- 需要根据图片尺寸进行转换

**坐标转换公式**：
```python
# 计算缩放比例
scale_x = 图片宽度 / 999
scale_y = 图片高度 / 999

# 转换归一化坐标到实际像素坐标
实际x = OCR_x × scale_x
实际y = OCR_y × scale_y
```

**示例**：
```
图片尺寸：1000 × 800
OCR 返回：专[[100, 200, 150, 250]]

scale_x = 1000 / 999 ≈ 1.001
scale_y = 800 / 999 ≈ 0.801

转换后：
  x1 = 100 × 1.001 ≈ 100
  y1 = 200 × 0.801 ≈ 160
  x2 = 150 × 1.001 ≈ 150
  y2 = 250 × 0.801 ≈ 200

实际边界框：[100, 160, 150, 200]
```

### TextBox 数据结构

```python
@dataclass
class TextBox:
    # 基础信息
    id: str                          # "t_000", "t_001", ...
    text: str                        # 识别的文字

    # 边界框
    bbox: List[int]                  # [x1, y1, x2, y2]
    width: int = field(init=False)   # x2 - x1
    height: int = field(init=False)  # y2 - y1

    # 角色和状态
    role: TextRole                  # 文字角色
    skip: bool = False              # 是否跳过

    # 翻译结果
    translated_text: Optional[str] = None

    # 视觉特征
    features: Optional[TextFeatures] = None
```

**数据流转过程**：

```
1. OCR 识别
   TextBox {id, text, bbox, role}

   ↓

2. 特征分析
   TextBox {..., features}

   ↓

3. 过滤标记
   TextBox {..., skip=True/False}

   ↓

4. 翻译
   TextBox {..., translated_text}

   ↓

5. 渲染
   使用完整的 TextBox 对象
```

### 性能统计

API 响应包含详细的性能统计：

```json
{
  "stats": {
    "total_texts": 5,           # 识别到的文字总数
    "translated_count": 3,      # 翻译的文字数
    "skipped_count": 2,         # 跳过的文字数

    "ocr_time_ms": 3200,        # OCR 耗时
    "translate_time_ms": 1500,  # 翻译耗时
    "render_time_ms": 800,      # 渲染耗时（修复 + 渲染）
    "total_time_ms": 5500       # 总耗时
  }
}
```

**性能基准**（基于测试）：
- OCR 识别：~3 秒（取决于图片大小和文字数量）
- 翻译：~1.5 秒（批量翻译）
- 背景修复（OpenCV）：~0.5 秒
- 文字渲染：~0.3 秒
- **总计**：~5-6 秒（单张图片）

### 文字框聚类算法

**目的**：将相邻的文字框合并，提高修复效果

**聚类规则**：
```
条件 1: x 轴有重叠
条件 2: y 距离 < 1.5 × 平均高度

满足以上两个条件的框会被合并到同一聚类
```

**示例**：
```
原始文字框：
  [100, 100, 200, 150]  "入会享"
  [210, 100, 300, 150]  "专属"    ← 与"入会享" x 轴重叠，y 距离 < 1.5 × 50 = 75px
  [100, 160, 250, 200]  "福利"    ← 与上面两个 y 距离 10px，合并

聚类结果：
  聚类 1: [100, 100, 300, 200]    包含 3 个框
```

**优势**：
- 减少修复次数
- 避免文字框之间的接缝
- 更自然的背景修复效果

### 颜色检测算法

**文字颜色检测**：
```python
# 1. 转换到灰度图
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# 2. OTSU 自动阈值分割
thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 3. 分离亮色和暗色区域
light_mask = binary > 127
dark_mask = binary <= 127

# 4. 计算每个区域的颜色中位数
light_color = median(roi[light_mask])
dark_color = median(roi[dark_mask])

# 5. 选择与背景对比度更大的颜色
if 与背景对比度(dark_color) > 与背景对比度(light_color):
    文字颜色 = dark_color
else:
    文字颜色 = light_color
```

**背景色检测**：
```python
# 1. 扩展采样范围（padding = 15px）
# 2. 采样四周区域
regions = [
    上方区域,  # image[y1-15:y1, x1:x2]
    下方区域,  # image[y2:y2+15, x1:x2]
    左侧区域,  # image[y1:y2, x1-15:x1]
    右侧区域,  # image[y1:y2, x2:x2+15]
]

# 3. 计算所有采样区域的中位数颜色
bg_color = median(all_regions_pixels)

# 4. 检测是否渐变
color_std = std(all_regions_pixels)
is_gradient = mean(color_std) > 30
```

## 📝 更新日志

### 2025-01-05
**新功能**：
- ✨ 新增边界框动态扩展功能
  - 解决部分笔画未被框选的问题（如"专"字头部、"利"字立刀旁）
  - 支持动态扩展比例（5-20px）
  - 可配置开启/关闭

**优化**：
- 📝 完善文档和注释
- 🐛 修复边界框扩展时的边界检查问题

### 2025-01-04
**新功能**：
- ✨ 新增 Qwen AI 修复模式
  - 支持智能背景修复
  - 适合复杂背景场景

**优化**：
- 📈 优化文字框聚类算法
- 🚀 提升渲染性能

### 2025-01-03
**初始版本**：
- 🎉 实现核心翻译流程
  - OCR 识别
  - 文字翻译
  - 背景修复（OpenCV）
  - 文字渲染
- 🌍 支持多语言翻译
- 🎨 自适应文字排版

## 👥 团队贡献

- **核心开发**：PicTrans 团队
- **技术栈**：
  - FastAPI - Web 框架
  - DeepSeek-OCR - 文字识别
  - DeepSeek 翻译 - 文字翻译
  - OpenCV - 图像处理
  - PIL/Pillow - 图像操作
  - Flask - API 服务

## 📄 许可证

[内部使用] - Internal Use Only

---

## 📧 联系方式

如有问题或建议，请联系开发团队。

**最后更新**：2025-01-05
