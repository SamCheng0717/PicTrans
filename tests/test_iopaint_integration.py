"""
测试 IOPaint 集成功能
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.core.ocr_client import OCRClient
from app.core.inpainter import Inpainter
from app.core.text_analyzer import TextAnalyzer
from app.core.translator import Translator
from app.core.text_renderer import TextRenderer


async def test_iopaint():
    """测试 IOPaint 模式的完整流程"""

    test_image = "E:\\PicTrans\\input\\demo_group\\1.jpg"

    if not Path(test_image).exists():
        print(f"测试图片不存在: {test_image}")
        return False

    print("=" * 60)
    print("IOPaint 集成测试")
    print("=" * 60)

    # 1. OCR 识别
    print("[1/7] 正在识别文字...")
    ocr_client = OCRClient()
    text_boxes = await ocr_client.recognize(test_image)
    print(f"✓ 识别到 {len(text_boxes)} 个文字框")

    if not text_boxes:
        print("✗ 未识别到文字，测试终止")
        return False

    # 1.5. 智能过滤：标记不需要翻译的文本
    print("[1.5/7] 智能过滤...")
    from app.core.pipeline import Pipeline
    skip_count = 0
    for box in text_boxes:
        if Pipeline._should_skip_translation(box.text):
            box.skip = True
            skip_count += 1
            print(f"  跳过翻译（保留原文）: '{box.text}'")
    print(f"✓ 智能过滤完成: {len(text_boxes)} 个文本, {skip_count} 个无需翻译")

    # 2. 读取图像
    print("[2/7] 读取图像...")
    image = cv2.imread(test_image)
    if image is None:
        print(f"✗ 无法读取图片: {test_image}")
        return False

    h, w = image.shape[:2]
    print(f"✓ 图片尺寸: {w}x{h}")

    # 3. 分析特征
    print("[3/7] 分析文字特征...")
    analyzer = TextAnalyzer()
    text_boxes = analyzer.analyze_all(image, text_boxes)
    print(f"✓ 特征分析完成")

    # 4. 翻译文字
    print("[4/7] 翻译文字...")
    translator = Translator()
    source_lang = "zh"  # 原文是中文
    target_lang = "ko"  # 翻译为韩文（使用AlibabaSansKR字体）
    text_boxes = await translator.translate_boxes(
        text_boxes,
        target_lang=target_lang,  # 使用关键字参数，避免混淆
        source_lang=source_lang
    )
    print(f"✓ 翻译完成")
    for box in text_boxes:
        print(f"  '{box.text}' → '{box.translated_text}'")

    # 5. IOPaint 修复背景
    print("[5/7] 调用 IOPaint 进行背景修复...")
    inpainter = Inpainter(mode="iopaint")

    try:
        inpainted_bg = inpainter.inpaint(image, text_boxes)
        print("✓ IOPaint 修复成功")

        # 6. 渲染翻译文字
        print("[6/7] 渲染翻译文字...")
        print(f"[DEBUG] target_lang = '{target_lang}'")
        renderer = TextRenderer(target_lang=target_lang)
        print(f"[DEBUG] renderer.target_lang = '{renderer.target_lang}'")

        # 测试字体路径
        from app.config import config
        test_font_path = config.font.get_font_path("regular", target_lang)
        print(f"[DEBUG] 字体路径: {test_font_path}")
        print(f"[DEBUG] 字体是否存在: {test_font_path.exists()}")

        final_result = renderer.render(inpainted_bg, text_boxes)
        print("✓ 文字渲染完成")

        # 7. 生成对比图
        print("[7/7] 生成对比可视化...")

        # 创建输出目录
        output_dir = Path("tests/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ========== 可视化输出开始 ==========

        # 1. OCR框选可视化
        print("\n[可视化] 生成OCR框选图...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 加载中文字体
        font_path = (
            Path(__file__).parent.parent
            / "fonts"
            / "MiSans"
            / "ttf"
            / "MiSans-Regular.ttf"
        )
        font = ImageFont.truetype(str(font_path), 20)

        # 定义颜色列表（RGB格式）
        colors = [
            (255, 0, 0),  # 红
            (0, 255, 0),  # 绿
            (0, 0, 255),  # 蓝
            (255, 165, 0),  # 橙
            (128, 0, 128),  # 紫
            (0, 255, 255),  # 青
        ]

        # 绘制每个文字框
        for i, box in enumerate(text_boxes):
            x1, y1, x2, y2 = box.bbox
            color = colors[i % len(colors)]

            # 绘制矩形框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 绘制标签（编号 + 文字）
            label = f"{i + 1}: {box.text}"
            label_y = y1 - 25 if y1 >= 25 else y2 + 5

            # 标签背景
            bbox_coords = draw.textbbox((x1, label_y), label, font=font)
            draw.rectangle(
                [
                    bbox_coords[0] - 2,
                    bbox_coords[1] - 2,
                    bbox_coords[2] + 2,
                    bbox_coords[3] + 2,
                ],
                fill=color,
            )

            # 标签文字（白色）
            draw.text((x1, label_y), label, font=font, fill=(255, 255, 255))

        # 转回OpenCV格式并保存
        ocr_boxes_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        ocr_boxes_path = output_dir / "test_iopaint_ocr_boxes.jpg"
        cv2.imwrite(str(ocr_boxes_path), ocr_boxes_image)
        print(f"✓ OCR框选: {ocr_boxes_path}")

        # 2. 遮罩层输出
        print("[可视化] 生成遮罩层...")

        # 重新生成遮罩（与IOPaint使用的相同）
        clusters = inpainter._cluster_boxes(text_boxes)
        if clusters:
            mask = inpainter._create_cluster_mask(image.shape, clusters)
        else:
            mask = inpainter._create_mask(image.shape, text_boxes)

        # 保存二值mask
        mask_path = output_dir / "test_iopaint_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"✓ 遮罩层: {mask_path}")

        # 创建遮罩叠加可视化（红色半透明）
        mask_overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 2] = mask  # 填充红色通道（BGR格式）

        # 混合：70%原图 + 30%红色mask
        mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)

        # 保存遮罩叠加图
        mask_overlay_path = output_dir / "test_iopaint_mask_overlay.jpg"
        cv2.imwrite(str(mask_overlay_path), mask_overlay)
        print(f"✓ 遮罩叠加: {mask_overlay_path}")

        # ========== 可视化输出结束 ==========

        # 保存各阶段结果
        print("\n[保存结果] 输出各阶段图片...")

        # 保存原图
        original_path = output_dir / "test_iopaint_original.jpg"
        cv2.imwrite(str(original_path), image)
        print(f"✓ 原图: {original_path}")

        # 保存背景修复结果
        inpainted_path = output_dir / "test_iopaint_inpainted.jpg"
        cv2.imwrite(str(inpainted_path), inpainted_bg)
        print(f"✓ 背景修复: {inpainted_path}")

        # 保存最终翻译结果
        final_path = output_dir / "test_iopaint_translated.jpg"
        cv2.imwrite(str(final_path), final_result)
        print(f"✓ 翻译结果: {final_path}")

        # 创建对比图（原图 vs 翻译结果）
        comparison = np.hstack([image, final_result])
        comparison_path = output_dir / "test_iopaint_comparison.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"✓ 原图vs翻译对比: {comparison_path}")

        # 创建完整流程对比（原图 + 修复背景 + 翻译结果）
        full_comparison = np.hstack([image, inpainted_bg, final_result])
        full_comparison_path = output_dir / "test_iopaint_full_comparison.jpg"
        cv2.imwrite(str(full_comparison_path), full_comparison)
        print(f"✓ 完整流程对比: {full_comparison_path}")

        print()
        print("=" * 60)
        print("测试完成！")
        print(f"\n输出文件：")
        print(f"  1. OCR识别框选: {(output_dir / 'test_iopaint_ocr_boxes.jpg').absolute()}")
        print(f"  2. 遮罩层: {(output_dir / 'test_iopaint_mask.png').absolute()}")
        print(f"  3. 遮罩叠加: {(output_dir / 'test_iopaint_mask_overlay.jpg').absolute()}")
        print(f"  4. 原图: {(output_dir / 'test_iopaint_original.jpg').absolute()}")
        print(f"  5. 背景修复: {(output_dir / 'test_iopaint_inpainted.jpg').absolute()}")
        print(f"  6. 翻译结果: {(output_dir / 'test_iopaint_translated.jpg').absolute()}")
        print(f"  7. 原图vs翻译对比: {(output_dir / 'test_iopaint_comparison.jpg').absolute()}")
        print(f"  8. 完整流程对比: {(output_dir / 'test_iopaint_full_comparison.jpg').absolute()}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"✗ IOPaint 修复失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    try:
        success = asyncio.run(test_iopaint())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
