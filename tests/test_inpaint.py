"""
测试文字抹除（Inpainting）功能
测试 OpenCV 背景修复模式（iopaint 待实现）
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from app.core.ocr_client import OCRClient
from app.core.inpainter import Inpainter
from app.core.text_analyzer import TextAnalyzer


async def test_inpaint(test_image: str, mode: str = "opencv"):
    """
    测试文字抹除功能

    Args:
        test_image: 测试图片路径
        mode: 抹除模式 "opencv" 或 "iopaint"

    测试流程：
    1. OCR 识别图片中的文字
    2. 分析文字特征（检测渐变背景）
    3. 创建文字区域 mask
    4. 执行文字抹除
    5. 保存对比结果
    """

    if not Path(test_image).exists():
        print(f"✗ 测试图片不存在: {test_image}")
        return False

    print("=" * 60)
    print(f"文字抹除测试 - {mode.upper()} 模式")
    print("=" * 60)
    print(f"测试图片: {test_image}")
    print()

    # Step 1: OCR 识别
    print("[1/5] 正在识别文字...")
    ocr_client = OCRClient()
    text_boxes = await ocr_client.recognize(test_image)

    print(f"✓ 识别到 {len(text_boxes)} 个文字框")
    for i, box in enumerate(text_boxes[:5]):
        print(f"  {i+1}. '{box.text}' @ [{box.bbox[0]},{box.bbox[1]},{box.bbox[2]},{box.bbox[3]}]")
    if len(text_boxes) > 5:
        print(f"  ... 还有 {len(text_boxes) - 5} 个")
    print()

    if not text_boxes:
        print("✗ 未识别到文字，测试终止")
        return False

    # Step 2: 读取原图
    print("[2/5] 读取原图...")
    image = cv2.imread(test_image)
    if image is None:
        print(f"✗ 无法读取图片: {test_image}")
        return False

    h, w = image.shape[:2]
    print(f"✓ 图片尺寸: {w}x{h}")
    print()

    # Step 3: 分析文字特征（检测渐变背景）
    print("[3/5] 分析文字特征...")
    analyzer = TextAnalyzer()
    analyzer.analyze_all(image, text_boxes)

    gradient_count = sum(1 for box in text_boxes
                        if box.features and box.features.background_is_gradient)
    print(f"✓ 特征分析完成")
    print(f"  渐变背景: {gradient_count} 个")
    print(f"  纯色背景: {len(text_boxes) - gradient_count} 个")
    print()

    # Step 4: 执行文字抹除
    print(f"[4/5] 执行文字抹除 ({mode} 模式)...")
    inpainter = Inpainter(mode=mode)

    try:
        result = inpainter.inpaint(image, text_boxes)
        print(f"✓ 文字抹除完成")
        print()
    except Exception as e:
        print(f"✗ 文字抹除失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: 保存结果
    print("[5/5] 保存结果...")

    # 创建输出目录
    output_dir = Path("tests/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取测试图片名称（不含扩展名）
    image_name = Path(test_image).stem

    # 保存原图
    original_path = output_dir / f"{image_name}_original.jpg"
    cv2.imwrite(str(original_path), image)
    print(f"✓ 原图: {original_path}")

    # 保存处理结果
    result_path = output_dir / f"{image_name}_{mode}_result.jpg"
    cv2.imwrite(str(result_path), result)
    print(f"✓ 结果: {result_path}")

    # 创建并保存 mask 可视化
    mask = inpainter._create_mask(image.shape, text_boxes)
    mask_path = output_dir / f"{image_name}_{mode}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"✓ Mask: {mask_path}")

    # 创建 mask 叠加可视化（红色半透明）
    mask_overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 2] = mask  # 红色通道
    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
    mask_overlay_path = output_dir / f"{image_name}_{mode}_mask_overlay.jpg"
    cv2.imwrite(str(mask_overlay_path), mask_overlay)
    print(f"✓ Mask叠加: {mask_overlay_path}")

    # 创建对比图（原图 | 结果）
    if image.shape != result.shape:
        print(f"  注意: 结果尺寸不一致，调整中... {result.shape} -> {image.shape}")
        result = cv2.resize(result, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    comparison = np.hstack([image, result])
    comparison_path = output_dir / f"{image_name}_{mode}_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"✓ 对比图: {comparison_path}")

    # 创建三联图（原图 | mask叠加 | 结果）
    triple = np.hstack([image, mask_overlay, result])
    triple_path = output_dir / f"{image_name}_{mode}_triple.jpg"
    cv2.imwrite(str(triple_path), triple)
    print(f"✓ 三联图: {triple_path}")

    print()
    print("=" * 60)
    print("测试完成！")
    print(f"请查看 {output_dir} 目录下的结果图片：")
    print(f"  - {result_path.name} (抹除结果)")
    print(f"  - {comparison_path.name} (原图对比)")
    print(f"  - {triple_path.name} (原图|mask|结果)")
    print("=" * 60)

    return True


def main():
    """运行测试"""
    import argparse

    parser = argparse.ArgumentParser(description="测试文字抹除功能")
    parser.add_argument("image", nargs="?", default="input/test5.jpg",
                       help="测试图片路径 (默认: input/test5.jpg)")
    parser.add_argument("-m", "--mode",
                       choices=["opencv", "iopaint"],
                       default="opencv",
                       help="测试模式: opencv 或 iopaint (默认: opencv)")

    args = parser.parse_args()

    try:
        success = asyncio.run(test_inpaint(args.image, mode=args.mode))

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
