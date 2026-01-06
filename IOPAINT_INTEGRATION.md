# IOPaint 集成文档

**版本**: 1.0
**更新日期**: 2026-01-06
**服务地址**: http://192.168.103.43:8080

---

## 目录

1. [概述](#概述)
2. [IOPaint API 规范](#iopaint-api-规范)
3. [PicTrans 配置](#pictrans-配置)
4. [使用方法](#使用方法)
5. [实现细节](#实现细节)
6. [故障排除](#故障排除)
7. [性能对比](#性能对比)
8. [参考资源](#参考资源)

---

## 概述

### 什么是 IOPaint？

IOPaint 是一个基于 AI 的图像修复服务，使用 LAMA（Large Mask Inpainting）等先进模型进行智能背景重建。它能够高质量地擦除图像中的指定区域（如文字、水印），并自动修复背景。

### 在 PicTrans 中的作用

在 PicTrans 图片翻译流程中，IOPaint 负责**背景修复**步骤：

```
OCR 识别文字 → 创建文字 Mask → IOPaint 修复背景 → 翻译文字 → 渲染新文字
```

IOPaint 擦除原始文字区域并智能修复背景，为后续的翻译文字渲染提供干净的画布。

### OpenCV vs IOPaint 对比

| 特性 | OpenCV 模式 | IOPaint 模式 |
|------|------------|-------------|
| **处理速度** | ~0.5秒 | ~5-10秒 |
| **质量** | 中等 | 高质量 |
| **适用场景** | 纯色/简单渐变背景 | 复杂纹理/照片背景 |
| **依赖** | 本地 OpenCV 库 | 远程 IOPaint 服务 |
| **网络要求** | 无 | 需要访问服务器 |
| **算法** | Telea/NS 算法 | AI 深度学习模型 |

**推荐使用场景**:
- **OpenCV**: 电商产品图（纯色背景）、批量快速处理
- **IOPaint**: 风景照、人物照、复杂纹理背景、追求最高质量

---

## IOPaint API 规范

### 服务信息

- **基础 URL**: `http://192.168.103.43:8080`
- **API 文档**: http://192.168.103.43:8080/docs
- **OpenAPI 规范**: http://192.168.103.43:8080/openapi.json
- **框架**: FastAPI
- **OpenAPI 版本**: 3.1.0

### 核心端点: POST /api/v1/inpaint

#### 请求格式

```http
POST /api/v1/inpaint HTTP/1.1
Host: http://192.168.103.43:8080
Content-Type: application/json

{
  "image": "<base64_encoded_image>",
  "mask": "<base64_encoded_mask>",
  "ldm_steps": 20,
  "ldm_sampler": "plms",
  "hd_strategy": "Crop",
  "hd_strategy_crop_trigger_size": 800,
  "hd_strategy_crop_margin": 128,
  "hd_strategy_resize_limit": 2000
}
```

#### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image` | string | ✅ | - | Base64 编码的原始图像（PNG 格式） |
| `mask` | string | ✅ | - | Base64 编码的 mask 图像（255=擦除，0=保留） |
| `ldm_steps` | integer | ❌ | 20 | LDM 模型推理步数（越大越慢但质量越高） |
| `ldm_sampler` | string | ❌ | "plms" | 采样器（plms/ddim/pndm） |
| `hd_strategy` | string | ❌ | "Crop" | 高分辨率策略（Crop/Resize/Original） |
| `hd_strategy_crop_trigger_size` | integer | ❌ | 800 | 触发 Crop 的最小尺寸 |
| `hd_strategy_crop_margin` | integer | ❌ | 128 | Crop 边距 |
| `hd_strategy_resize_limit` | integer | ❌ | 1280 | Resize 策略的最大尺寸 |

#### 响应格式

- **Content-Type**: `image/png` 或 `application/json`
- **成功响应**: 直接返回修复后的图像二进制数据，或包含 base64 图像的 JSON
- **错误响应**: HTTP 状态码 4xx/5xx，JSON 格式错误信息

#### cURL 示例

```bash
# 使用 Python 生成 base64 编码
python -c "
import base64
with open('image.png', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
with open('mask.png', 'rb') as f:
    mask_b64 = base64.b64encode(f.read()).decode()
print(img_b64[:50], mask_b64[:50])
"

# 发送请求
curl -X POST http://192.168.103.43:8080/api/v1/inpaint \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<image_base64>",
    "mask": "<mask_base64>",
    "ldm_steps": 20
  }' \
  --output result.png
```

### 其他可用端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/server-config` | GET | 获取服务器配置（模型列表、插件等） |
| `/api/v1/model` | GET | 获取当前加载的模型信息 |
| `/api/v1/model` | POST | 切换使用的修复模型 |
| `/api/v1/samplers` | GET | 获取可用的采样器列表 |
| `/api/v1/adjust_mask` | POST | 调整 mask（扩展/收缩/反转） |
| `/api/v1/gen-info` | POST | 提取图像生成信息（元数据） |

---

## PicTrans 配置

### 配置文件位置

`app/config.py` → `InpaintConfig` 类

### 配置参数

```python
@dataclass
class InpaintConfig:
    """图像修复配置"""
    # ========== 模式选择 ==========
    mode: str = "opencv"  # "opencv" 或 "iopaint"

    # ========== OpenCV 参数 ==========
    opencv_radius: int = 5
    opencv_method: str = "telea"  # telea / ns
    sample_padding: int = 10
    blur_kernel: int = 15
    mask_expand: int = 8  # Mask 扩展像素

    # ========== IOPaint AI 修复配置 ==========
    iopaint_api_url: str = "http://192.168.103.43:8080"  # 服务地址
    iopaint_timeout: int = 60  # 超时时间（秒）
    iopaint_model: str = "lama"  # 修复模型：lama/ldm/fcf
    iopaint_ldm_steps: int = 20  # LDM 推理步数
    iopaint_no_half: bool = False  # 禁用半精度（FP16）
    iopaint_size_limit: str = "2000"  # 最大边长限制（像素）
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| **iopaint_api_url** | str | http://192.168.103.43:8080 | IOPaint 服务的完整 URL（不包含 /api/v1） |
| **iopaint_timeout** | int | 60 | HTTP 请求超时时间（秒），AI 处理较慢 |
| **iopaint_model** | str | lama | 使用的修复模型：<br>• `lama` - 推荐，快速且高质量<br>• `ldm` - 潜在扩散模型<br>• `fcf` - 自由形式修复 |
| **iopaint_ldm_steps** | int | 20 | LDM 模型的推理步数（10-100），越大越慢但质量越高 |
| **iopaint_no_half** | bool | False | 是否禁用半精度浮点数（FP16），True 则使用 FP32（更慢但更精确） |
| **iopaint_size_limit** | str | "2000" | 图像最大边长限制（像素），超过则自动缩放 |

### 修改配置

**方式 1: 直接修改 config.py**

```python
# E:\PicTrans\app\config.py
iopaint_api_url: str = "http://your-iopaint-server:8080"
iopaint_timeout: int = 90  # 增加到 90 秒
iopaint_ldm_steps: int = 30  # 更高质量
```

**方式 2: 运行时覆盖**

```python
from app.config import config

# 临时修改配置
config.inpaint.iopaint_api_url = "http://localhost:8080"
config.inpaint.iopaint_timeout = 120
```

---

## 使用方法

### CLI 命令行使用

#### 基本用法

```bash
# 使用 IOPaint 模式处理单张图片
python cli.py input/test5.jpg -t ko --inpaint iopaint

# 使用 OpenCV 模式（默认）
python cli.py input/test5.jpg -t ko --inpaint opencv
```

#### 批量处理

```bash
# 批量处理整个文件夹
python cli.py input/ -t ko --inpaint iopaint -c 3

# 多语言输出
python cli.py input/test5.jpg -t ko -t ja -t en --inpaint iopaint
```

#### CLI 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-t, --target-lang` | 目标语言（可多次指定） | `-t ko -t ja` |
| `-s, --source-lang` | 源语言（默认 zh） | `-s zh` |
| `--inpaint` | Inpaint 模式 | `--inpaint iopaint` |
| `-c, --concurrent` | 并发处理数（默认 3） | `-c 5` |
| `-o, --output-dir` | 输出目录 | `-o ./translated` |

### API 调用

#### Flask API 端点

```bash
curl -X POST http://localhost:5000/api/translate \
  -F "image=@input/test5.jpg" \
  -F "target_lang=ko" \
  -F "source_lang=zh" \
  -F "inpaint_mode=iopaint"
```

#### 响应格式

```json
{
  "success": true,
  "message": "处理成功",
  "output_path": "E:\\PicTrans\\output\\test5_ko_20260106_105119.jpg",
  "detected_texts": [
    {
      "id": "text_001",
      "text": "十月稻田",
      "bbox": [489, 3, 789, 85],
      "role": "feature",
      "skip": false
    }
  ],
  "translated_texts": [
    {
      "id": "text_001",
      "original": "十月稻田",
      "translated": "시월 다오톈"
    }
  ],
  "stats": {
    "total_texts": 12,
    "translated_count": 12,
    "skipped_count": 0,
    "ocr_time_ms": 1568,
    "translate_time_ms": 5425,
    "render_time_ms": 253,
    "total_time_ms": 7723
  }
}
```

### Python 代码调用

#### 使用 Pipeline

```python
from app.core.pipeline import Pipeline
from app.models.schemas import TranslationTask

# 创建 IOPaint 模式的 pipeline
pipeline = Pipeline(inpaint_mode="iopaint")

# 创建翻译任务
task = TranslationTask(
    image_path="input/test5.jpg",
    source_lang="zh",
    target_lang="ko",
    inpaint_mode="iopaint"
)

# 同步执行
result = pipeline.process_sync(task)

# 或异步执行
import asyncio
result = asyncio.run(pipeline.process(task))

# 检查结果
if result.success:
    print(f"✓ 成功: {result.output_path}")
    print(f"识别: {result.total_texts} 个文字")
    print(f"翻译: {result.translated_texts} 个")
    print(f"耗时: {result.total_time}ms")
else:
    print(f"✗ 失败: {result.error_message}")
```

#### 直接使用 Inpainter

```python
import cv2
from app.core.inpainter import Inpainter
from app.models.schemas import TextBox

# 创建 Inpainter（IOPaint 模式）
inpainter = Inpainter(mode="iopaint")

# 读取图像
image = cv2.imread("input/test.jpg")

# 创建文字框列表
text_boxes = [
    TextBox(
        id="text_001",
        text="示例文字",
        bbox=[100, 100, 300, 150]
    )
]

# 执行修复
try:
    result_image = inpainter.inpaint(image, text_boxes)
    cv2.imwrite("output/inpainted.jpg", result_image)
    print("✓ 修复成功")
except Exception as e:
    print(f"✗ 修复失败: {e}")
```

---

## 实现细节

### 架构概览

```
┌─────────────────────────────────────────────────────────┐
│ Pipeline (app/core/pipeline.py)                         │
│                                                           │
│  ┌─────────┐   ┌─────────────┐   ┌────────────────┐    │
│  │ OCR     │──▶│ Inpainter   │──▶│ Translator     │    │
│  │ Client  │   │             │   │                 │    │
│  └─────────┘   └──────┬──────┘   └────────────────┘    │
│                       │                                  │
│                       ▼                                  │
│             ┌──────────────────┐                        │
│             │ IOPaintClient    │                        │
│             │ (iopaint_client  │                        │
│             │  .py)            │                        │
│             └─────────┬────────┘                        │
│                       │                                  │
│                       ▼                                  │
│             ┌──────────────────┐                        │
│             │ IOPaint Service  │                        │
│             │ (FastAPI)        │                        │
│             │ 192.168.103.43:  │                        │
│             │ 8080             │                        │
│             └──────────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. IOPaintClient (`app/core/iopaint_client.py`)

**职责**: 封装 IOPaint HTTP API 调用

**关键方法**:
```python
class IOPaintClient:
    def __init__(self):
        """从 config.inpaint 读取配置"""

    def _image_to_base64(self, image: np.ndarray) -> str:
        """BGR numpy → Base64 string"""

    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """Mask numpy → Base64 string"""

    def _base64_to_image(self, base64_str: str) -> np.ndarray:
        """Base64 string → BGR numpy"""

    async def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """异步调用 IOPaint API"""

    def inpaint_sync(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """同步包装方法"""
```

**特点**:
- 配置驱动：所有参数从 `config.inpaint` 读取
- 格式转换：自动处理 BGR ↔ RGB ↔ Base64
- 异步优先：主接口为 async，提供 sync 包装
- 错误传播：直接抛出 HTTP 异常

#### 2. Inpainter 集成 (`app/core/inpainter.py`)

**新增方法**: `_inpaint_iopaint()`

```python
def _inpaint_iopaint(
    self,
    image: np.ndarray,
    text_boxes: List[TextBox],
    clusters: List[List[TextBox]]
) -> np.ndarray:
    """使用 IOPaint AI 模型进行背景修复"""

    # 1. 延迟初始化 IOPaint 客户端
    if self._iopaint_client is None:
        self._iopaint_client = IOPaintClient()

    # 2. 创建 mask（优先使用聚类 mask）
    if clusters:
        mask = self._create_cluster_mask(image.shape, clusters)
    else:
        mask = self._create_mask(image.shape, text_boxes)

    # 3. 在新事件循环中调用 IOPaint（避免 asyncio 冲突）
    def run_in_new_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._iopaint_client.inpaint(image, mask)
            )
        finally:
            loop.close()

    # 4. 使用 ThreadPoolExecutor 执行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_new_loop)
        result = future.result()

    return result
```

**设计要点**:
- **延迟初始化**: 仅在使用 iopaint 模式时才创建客户端
- **新事件循环**: 避免与现有 asyncio 循环冲突
- **ThreadPoolExecutor**: 在单独线程中运行异步代码
- **错误直接抛出**: 不回退到 OpenCV（用户要求）

### 数据流程

```
┌──────────────────────────────────────────────────────────┐
│ 1. OCR 识别                                               │
│    Image → TextBox[] (bbox, text, role, ...)             │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│ 2. 创建 Mask                                              │
│    TextBox[] → Binary Mask (255=擦除, 0=保留)            │
│    • 文字框聚类（合并相邻框）                             │
│    • Mask 扩展 8px（防止边缘残留）                        │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│ 3. 格式转换                                               │
│    Image (BGR numpy) → PNG bytes → Base64 string          │
│    Mask (grayscale numpy) → PNG bytes → Base64 string     │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│ 4. IOPaint API 调用                                       │
│    POST /api/v1/inpaint                                   │
│    {                                                      │
│      "image": "<base64>",                                 │
│      "mask": "<base64>",                                  │
│      "ldm_steps": 20,                                     │
│      ...                                                  │
│    }                                                      │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│ 5. 响应解析                                               │
│    • 尝试 JSON（包含 base64）                             │
│    • 尝试纯 base64 字符串                                 │
│    • 尝试二进制图像数据                                   │
│    Base64/Binary → PNG bytes → numpy → BGR array         │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│ 6. 尺寸校验                                               │
│    if output.shape != input.shape:                        │
│        output = cv2.resize(output, input.shape)           │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│ 7. 后续处理                                               │
│    Cleaned Image → Translator → Text Renderer            │
└──────────────────────────────────────────────────────────┘
```

### 关键技术点

#### 1. 颜色空间转换

```python
# BGR (OpenCV) → RGB (PIL) → PNG
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(rgb_image)

# PNG → RGB (PIL) → BGR (OpenCV)
rgb_array = np.array(pil_image)
bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
```

**注意事项**:
- OpenCV 默认使用 BGR 顺序
- PIL/网络传输使用 RGB 顺序
- 必须正确转换，否则颜色会错乱

#### 2. Base64 编码

```python
import base64
import io
from PIL import Image

# 编码
buffer = io.BytesIO()
pil_image.save(buffer, format='PNG')
base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

# 解码
image_bytes = base64.b64decode(base64_str)
pil_image = Image.open(io.BytesIO(image_bytes))
```

**为什么用 Base64？**
- JSON 兼容：可以在 JSON 中传输二进制数据
- 无需文件IO：直接在内存中处理
- 跨平台兼容：标准编码格式

#### 3. 异步处理

**问题**: Pipeline 中可能已有 asyncio 事件循环

**解决**: 在新线程中创建新事件循环

```python
import asyncio
import concurrent.futures

def run_in_new_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_function())
    finally:
        loop.close()

with concurrent.futures.ThreadPoolExecutor() as executor:
    future = executor.submit(run_in_new_loop)
    result = future.result()  # 阻塞等待结果
```

#### 4. 超时控制

```python
# httpx 客户端超时配置
async with httpx.AsyncClient(timeout=60) as client:
    response = await client.post(url, json=payload)
```

**超时设置**:
- 默认 60 秒（AI 处理较慢）
- 可通过 `iopaint_timeout` 配置调整
- 超时会抛出 `httpx.TimeoutException`

---

## 故障排除

### 常见错误

#### 1. 405 Method Not Allowed

**错误信息**:
```
httpx.HTTPStatusError: Client error '405 Method Not Allowed'
for url 'http://192.168.103.43:8080/inpaint'
```

**原因**: 使用了错误的 API 端点

**解决**:
- ✅ 正确: `/api/v1/inpaint`
- ❌ 错误: `/inpaint`

**检查代码**:
```python
# app/core/iopaint_client.py 第 154 行
f"{self.api_url}/api/v1/inpaint"  # 应该包含 /api/v1
```

---

#### 2. 请求超时

**错误信息**:
```
httpx.TimeoutException: Request exceeded the configured timeout.
```

**原因**:
- IOPaint AI 处理较慢（5-10秒）
- 服务器负载过高
- 图像尺寸过大

**解决方法**:

**方法 1: 增加超时时间**
```python
# app/config.py
iopaint_timeout: int = 90  # 增加到 90 秒
```

**方法 2: 减小图像尺寸**
```python
iopaint_size_limit: str = "1500"  # 降低到 1500px
```

**方法 3: 减少推理步数**
```python
iopaint_ldm_steps: int = 10  # 降低到 10 步（更快但质量稍低）
```

---

#### 3. 连接拒绝

**错误信息**:
```
httpx.ConnectError: [Errno 111] Connection refused
```

**原因**: IOPaint 服务未启动或地址错误

**解决步骤**:

1. **检查服务状态**:
```bash
curl http://192.168.103.43:8080/docs
# 应该返回 HTML 文档页面
```

2. **检查配置地址**:
```python
# app/config.py
iopaint_api_url: str = "http://192.168.103.43:8080"  # 确认地址正确
```

3. **检查网络连通性**:
```bash
ping 192.168.103.43
```

4. **检查防火墙**:
```bash
# Linux
sudo ufw status
sudo ufw allow 8080

# Windows
# 检查 Windows 防火墙设置
```

---

#### 4. 图像尺寸过大

**错误信息**:
```
RuntimeError: IOPaint 修复失败: Image size exceeds limit
```

**原因**: 图像超过 `hd_strategy_resize_limit`

**解决**:

**方法 1: 调整 size_limit**
```python
# app/config.py
iopaint_size_limit: str = "3000"  # 增加限制
```

**方法 2: 使用 Crop 策略**
- Crop 策略会自动分块处理大图
- 默认已启用：`hd_strategy="Crop"`

**方法 3: 预处理缩小图像**
```python
# 在发送前检查尺寸
if max(image.shape[:2]) > 2000:
    scale = 2000 / max(image.shape[:2])
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    image = cv2.resize(image, new_size)
```

---

#### 5. Base64 解码错误

**错误信息**:
```
binascii.Error: Invalid base64-encoded string
```

**原因**: 响应格式不符合预期

**解决**: 检查 IOPaint 服务版本和响应格式

**调试方法**:
```python
# 打印响应内容类型和前 100 个字符
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Response preview: {response.text[:100]}")
```

---

### 调试方法

#### 1. 启用详细日志

IOPaintClient 会自动打印日志：

```
[IOPaint-Client] 初始化完成
  API 地址: http://192.168.103.43:8080
  模型: lama
  超时: 60s
[IOPaint-Client] 开始修复: 1280x1280
[IOPaint-Client] 修复完成
```

**如果看不到日志**，检查 Python 日志级别。

#### 2. 直接测试 IOPaint API

使用 curl 测试 API 是否可用：

```bash
# 测试健康检查
curl http://192.168.103.43:8080/api/v1/server-config

# 查看可用模型
curl http://192.168.103.43:8080/api/v1/model

# 测试 inpaint 端点（需要 base64 数据）
python tests/test_iopaint_integration.py
```

#### 3. 检查 OpenAPI 文档

访问 Swagger UI 查看最新 API 规范：

http://192.168.103.43:8080/docs

- 查看所有可用端点
- 查看参数说明
- 在线测试 API

#### 4. 使用集成测试

运行完整的集成测试：

```bash
python tests/test_iopaint_integration.py
```

测试流程：
1. OCR 识别文字框
2. 加载测试图像
3. 分析文字特征
4. 调用 IOPaint 修复
5. 保存结果到 `tests/output/`

#### 5. 对比 OpenCV 模式

如果 IOPaint 失败，尝试 OpenCV 模式确认问题范围：

```bash
# OpenCV 模式
python cli.py input/test5.jpg -t ko --inpaint opencv

# IOPaint 模式
python cli.py input/test5.jpg -t ko --inpaint iopaint
```

如果 OpenCV 成功但 IOPaint 失败，说明问题在 IOPaint 服务或网络。

---

## 性能对比

### 处理速度

| 模式 | 单张图片 | 批量处理(10张) | 备注 |
|------|---------|----------------|------|
| **OpenCV** | ~0.5s | ~5s | 纯 CPU 计算 |
| **IOPaint** | ~5-10s | ~60-80s | 包含网络传输 |

**影响因素**:
- 图像尺寸（越大越慢）
- 网络延迟
- IOPaint 服务器负载
- `ldm_steps` 参数（越大越慢）

### 质量对比

| 场景 | OpenCV | IOPaint | 推荐 |
|------|--------|---------|------|
| **纯色背景** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OpenCV（更快） |
| **简单渐变** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | IOPaint（更自然） |
| **复杂纹理** | ⭐⭐ | ⭐⭐⭐⭐⭐ | IOPaint（唯一选择） |
| **照片背景** | ⭐ | ⭐⭐⭐⭐⭐ | IOPaint（唯一选择） |
| **边缘处理** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | IOPaint（更平滑） |

### 成本对比

| 项目 | OpenCV | IOPaint |
|------|--------|---------|
| **硬件要求** | 无（CPU 即可） | GPU 服务器 |
| **部署复杂度** | 简单（已集成） | 中等（需单独部署） |
| **维护成本** | 低 | 中 |
| **并发能力** | 高（无瓶颈） | 中（受服务器限制） |
| **网络依赖** | 无 | 有 |

### 推荐策略

**OpenCV 优先场景**:
- 电商产品图（纯色/简单背景）
- 批量快速处理
- 无 GPU 服务器
- 对质量要求不高

**IOPaint 优先场景**:
- 人物照片、风景照
- 复杂纹理/渐变背景
- 追求最高质量
- 有 GPU 服务器资源

**混合策略**:
```python
# 根据背景复杂度自动选择
if is_simple_background:
    mode = "opencv"  # 快速处理
else:
    mode = "iopaint"  # 高质量处理
```

---

## 参考资源

### IOPaint 官方资源

- **GitHub**: https://github.com/Sanster/IOPaint
- **官网**: https://www.iopaint.com/
- **在线 Demo**: https://www.iopaint.com/
- **文档**: https://github.com/Sanster/IOPaint/wiki

### 本地部署

**服务地址**: http://192.168.103.43:8080
- **Swagger UI**: http://192.168.103.43:8080/docs
- **OpenAPI JSON**: http://192.168.103.43:8080/openapi.json
- **Redoc**: http://192.168.103.43:8080/redoc

### PicTrans 相关文件

| 文件 | 说明 |
|------|------|
| `app/core/iopaint_client.py` | IOPaint HTTP 客户端（~205 行） |
| `app/core/inpainter.py` | Inpainter 集成（第 360-418 行） |
| `app/config.py` | InpaintConfig 配置（第 96-115 行） |
| `tests/test_iopaint_integration.py` | 集成测试脚本 |
| `CLAUDE.md` | 开发文档（第 187-214 行） |

### 模型和算法

**LAMA (Large Mask Inpainting)**:
- 论文: Resolution-robust Large Mask Inpainting with Fourier Convolutions
- arXiv: https://arxiv.org/abs/2109.07161
- 特点: 快速、高质量、适合大区域修复

**LDM (Latent Diffusion Model)**:
- 基于 Stable Diffusion
- 特点: 更高质量但更慢

### 技术栈

- **httpx**: 现代异步 HTTP 客户端
- **PIL/Pillow**: Python 图像处理库
- **OpenCV**: 计算机视觉库
- **NumPy**: 数值计算库
- **asyncio**: Python 异步IO框架

---

## 更新日志

### v1.0 (2026-01-06)

**初始版本**:
- ✅ IOPaint API 集成完成
- ✅ 支持 CLI 和 API 调用
- ✅ Base64 编码格式
- ✅ 异步 HTTP 请求
- ✅ 错误处理和故障排除
- ✅ 完整文档编写

**已知问题**:
- IOPaint 失败时不回退到 OpenCV（设计决策）
- 大图像处理较慢（AI 模型限制）

**未来计划**:
- 添加连接池优化
- 支持批量请求
- 添加缓存机制
- 支持更多 IOPaint 模型

---

**文档维护者**: Claude Code
**最后更新**: 2026-01-06
**反馈**: E:\PicTrans\CLAUDE.md
