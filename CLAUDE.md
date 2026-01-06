# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PicTrans is an AI-powered image translation system specialized for e-commerce product localization. It translates text in product images through an 8-step pipeline: OCR recognition → text filtering → feature analysis → translation → background restoration → text rendering → output.

**Key Technology Stack:**
- Flask web framework with Blueprint architecture
- DeepSeek-OCR API for text recognition
- DeepSeek Translation API for text translation
- OpenCV for image processing and background restoration
- PIL/Pillow for image manipulation

## Development Commands

### Running the Application

**Option 1: API Server Mode**

```bash
# Start the development server
python run.py
```

The server runs on `http://localhost:5000` with debug mode enabled.

**Option 2: CLI Mode (Recommended for Batch Processing)**

```bash
# Process single image
python cli.py input.jpg -t ko

# Process directory
python cli.py ./images/ -t ko

# Multi-language output
python cli.py input.jpg -t ko -t ja -t en
```

CLI mode is faster and more efficient for batch processing as it bypasses HTTP overhead.

### Testing the API

```bash
# Basic translation test
curl -X POST http://localhost:5000/api/translate \
  -F "image=@test.jpg" \
  -F "target_lang=ko"

# Health check
curl http://localhost:5000/api/health

# Get supported languages
curl http://localhost:5000/api/languages
```

### CLI Mode Usage

```bash
# Basic usage
python cli.py <input_path> [options]

# Common options
  -t, --target-lang    Target language (can specify multiple)
  -s, --source-lang    Source language (default: zh)
  -c, --concurrent     Concurrent processing count (default: 3)
  -o, --output-dir     Output directory (default: ./output)
  --inpaint            Inpaint mode: opencv or iopaint (default: opencv)

# Examples
python cli.py product.jpg -t ko
python cli.py ./images/ -t ko -t ja -c 5
python cli.py product.jpg -t ko -o ./translated
```

### OCR Visualization Test

```bash
# Generate OCR visualization with bounding boxes
python tests/test_ocr_visual.py input/test.jpg
# Output: input/test_ocr_boxes.jpg
```

## Architecture

### Core Processing Pipeline

The entire translation flow is orchestrated by `app/core/pipeline.py`. The `Pipeline.process()` method executes these sequential steps:

1. **Image Loading** - Validates and reads image using OpenCV
2. **OCR Recognition** - Calls `OCRClient.recognize()` to extract text with bounding boxes
3. **Text Filtering** - Applies skip rules based on TextRole (price/promo/brand)
4. **Feature Analysis** - `TextAnalyzer.analyze_all()` detects color, stroke, shadow, font
5. **Translation** - `Translator.translate_boxes()` translates text in batch
6. **Background Restoration** - `Inpainter.inpaint()` erases original text and repairs background
7. **Text Rendering** - `TextRenderer.render()` renders translated text with detected features
8. **Save Result** - Writes output to `output/` directory with timestamp

### Data Flow: TextBox Object

The `TextBox` dataclass (defined in `app/models/schemas.py`) is the central data structure that flows through all pipeline stages:

```python
@dataclass
class TextBox:
    id: str                    # Unique identifier
    text: str                  # Original text from OCR
    bbox: List[int]            # [x1, y1, x2, y2] pixel coordinates
    translated_text: Optional[str]  # Translated text
    role: TextRole            # FEATURE/SLOGAN/BRAND/PRICE/PROMO/UNKNOWN
    features: Optional[TextFeatures]  # Visual features (color, stroke, shadow)
    skip: bool               # Whether to skip translation
```

**Property Methods:**
- `width`: bbox[2] - bbox[0]
- `height`: bbox[3] - bbox[1]
- `center`: center point coordinates

### Module Organization

**app/core/** - Core processing modules:
- `ocr_client.py` - OCR API integration with dynamic bbox refinement
- `translator.py` - Translation API with e-commerce optimized prompts
- `text_analyzer.py` - Visual feature detection (color, stroke, shadow, font)
- `inpainter.py` - Background restoration with clustering and mask expansion
- `text_renderer.py` - Adaptive text rendering with font fitting
- `pipeline.py` - Orchestrates all modules in sequence

**app/api/** - Web API layer:
- `routes.py` - Flask Blueprint with `/api/translate` endpoint
- Handles both multipart file upload and base64 JSON input

**cli.py** - Command-line interface:
- Direct execution without Flask server overhead
- Batch processing with concurrency control
- Multi-language output support
- Uses the same pipeline as API mode

**app/models/schemas.py** - Data models:
- `TextRole` enum, `TextBox`, `TextFeatures`, `TranslationTask`, `ProcessingResult`

## Critical Implementation Details

### OCR Coordinate System

DeepSeek-OCR returns normalized coordinates (0-999) that must be converted to pixel coordinates:

```python
scale_x = image_width / 999
scale_y = image_height / 999
pixel_x = normalized_x × scale_x
pixel_y = normalized_y × scale_y
```

### Dynamic Bbox Refinement

OCR bboxes are automatically expanded to cover partial strokes (added 2025-01-05):

```python
# In ocr_client.py::_refine_bbox()
expand = int(box_height × 0.15)  # 15% of height
expand = max(5, min(expand, 20))  # Clamp to 5-20px range
```

Configure in `app/config.py`:
- `bbox_refine_enabled: bool = True`
- `bbox_expand_ratio: float = 0.15`
- `bbox_expand_min: int = 5`
- `bbox_expand_max: int = 20`

### Text Filtering Strategy

Text boxes are filtered using fixed rules based on `TextRole` enum:
- PRICE → always skipped
- PROMO → always skipped
- BRAND → always translated
- FEATURE/SLOGAN → always translated
- UNKNOWN → always translated

These rules are implemented in `pipeline.py::_filter_boxes()` and cannot be customized via CLI or API parameters.

### Background Restoration Modes

**OpenCV Mode** (default and currently only supported):
- Fast (~0.5s)
- Background color sampling with median
- Mask expansion of 8px to prevent edge artifacts
- Clustering algorithm merges adjacent text boxes
- Best for solid color backgrounds
- Gradient background support via intelligent detection

**iopaint Mode** (planned):
- Will be integrated in the future
- AI-powered background reconstruction for complex backgrounds
- Currently falls back to OpenCV if specified

Configure via API parameter `inpaint_mode` or `InpaintConfig.mode`.

### Text Rendering Adaptive Algorithm

The `_fit_text_in_box()` method in `text_renderer.py` uses a three-strategy approach:
1. Try single-line with decreasing font size
2. Try two-line split if length ≥ 4 chars
3. Fall back to minimum font size

Configurable in `app/config.py`:
- `min_font_size: int = 12`
- `max_font_size: int = 200`
- `max_lines: int = 2`
- `line_spacing: float = 1.2`

### Font Configuration

Fonts are organized by language in `app/config.py`:

```python
language_fonts: dict = {
    "ko": {"dir": "AlibabaSansKR", "weights": {...}},
    "zh": {"dir": "MiSans/ttf", "weights": {...}},
    # ... other languages
}
```

To add a new language:
1. Add font files to `fonts/` directory
2. Update `supported_languages` dict
3. Add entry to `FontConfig.language_fonts`

## Configuration File Structure

All configuration is centralized in `app/config.py` using dataclasses:

- `OCRConfig` - OCR API and bbox refinement settings
- `TranslatorConfig` - Translation API and prompts
- `InpaintConfig` - Background restoration modes and parameters
- `RenderConfig` - Text rendering constraints
- `FontConfig` - Font paths per language and weight
- `AppConfig` - Output directory, supported languages, skip roles

The global `config` instance is imported throughout the codebase.

## API Endpoints

- `POST /api/translate` - Main translation endpoint
- `GET /api/health` - Health check
- `GET /api/languages` - Supported languages list
- `GET /api/output/<filename>` - Retrieve output file

## Output Conventions

Output files are saved to `output/` directory with naming pattern:
```
{original_name}_{target_lang}_{timestamp}.jpg
```

Example: `product_ko_20250105_143022.jpg`

## Common Modifications

### Adjusting Bbox Expansion

If OCR bboxes don't fully cover text strokes, adjust in `app/config.py`:

```python
bbox_expand_ratio: float = 0.20  # More aggressive (20%)
bbox_expand_min: int = 8
bbox_expand_max: int = 30
```

### Adding New Text Roles

1. Add enum value to `TextRole` in `app/models/schemas.py`
2. Update filtering logic in `pipeline.py::_filter_boxes()` to define the skip behavior for the new role

### Changing Translation Prompts

Edit `_build_prompt()` in `app/core/translator.py`. The current prompt is optimized for e-commerce with emphasis on:
- Concise, product-focused style
- Preserving technical terms
- Highlighting product features

### Adding New Inpainting Modes

1. Implement new method in `Inpainter` class following `_inpaint_opencv()` pattern
2. Add mode parameter to `__init__()` method
3. Update `inpaint()` method to route to new mode
4. Update `InpaintConfig` in `app/config.py`
5. Update the choices in `cli.py` `--inpaint` argument

## CLI Mode Architecture

### Execution Flow

```
cli.py::main()
  ↓
find_images() - Discover image files
  ↓
For each target_lang:
  ↓
  process_images() - Async batch processing
    ↓
    Pipeline.process_batch() - Concurrent processing
      ↓
      For each task:
        Pipeline.process() - Single image processing
          ↓
          [8-step pipeline as described above]
```

### Key Components

**find_images(input_path)**:
- Accepts file or directory path
- Supports: .jpg, .jpeg, .png, .webp, .gif
- Returns sorted list of Path objects

**process_images(images, target_lang, ...)**:
- Creates `TranslationTask` objects for each image
- Calls `Pipeline.process_batch()` with concurrency control
- Prints real-time progress and statistics
- Returns success/fail counts

**argparse Configuration**:
- `input` (positional): Image or directory path
- `-t, --target-lang`: Can be specified multiple times for multi-language output
- `-s, --source-lang`: Default "zh"
- `-c, --concurrent`: Default 3, controls parallel processing
- `-o, --output-dir`: Overrides default output directory
- `--inpaint`: Mode selection (opencv/iopaint, default: opencv)

### Concurrency Model

CLI mode uses `asyncio.gather()` for concurrent processing:

```python
# In pipeline.py::process_batch()
semaphore = asyncio.Semaphore(max_concurrent)
tasks = [process_with_semaphore(task, semaphore) for task in tasks]
results = await asyncio.gather(*tasks)
```

Benefits:
- API calls are parallelized (OCR + translation)
- Limits concurrent requests to avoid rate limiting
- Maintains order of input files

### Multi-Language Output

When multiple `-t` flags are provided:

```bash
python cli.py product.jpg -t ko -t ja -t en
```

Execution:
1. Process all images for first language (ko)
2. Process all images for second language (ja)
3. Process all images for third language (en)
4. Output separate files for each language

Output naming:
```
product_ko_20250105_143022.jpg
product_ja_20250105_143025.jpg
product_en_20250105_143028.jpg
```

### Output Format

**Success**:
```
✓ [1/10] product.jpg
  输出: E:\PicTrans\output\product_ko_20250105_143022.jpg
  识别: 4 个文字, 翻译: 4, 跳过: 0
  耗时: 5200ms (OCR:3200ms, 翻译:1500ms, 渲染:500ms)
```

**Failure**:
```
✗ [2/10] corrupted.jpg
  错误: Failed to decode image
```

**Summary**:
```
处理完成: 成功 9, 失败 1
```

### Advantages Over API Mode

1. **No HTTP overhead** - Direct function calls
2. **Built-in batching** - Process entire directories at once
3. **Multi-language support** - Specify multiple target languages in single command
4. **Progress tracking** - Real-time terminal output
5. **Simpler integration** - Easy to use in shell scripts and automation
