"""
OCR 识别结果可视化测试
在原图上画框，显示识别到的文字
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.core.ocr_client import OCRClient


async def test_ocr_visual(image_path: str, output_path: str = None):
    """
    可视化 OCR 识别结果
    """
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return

    # 读取图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # OCR 识别
    ocr = OCRClient()
    text_boxes = await ocr.recognize(image_path)

    print(f"\n识别到 {len(text_boxes)} 个文字框:")
    print("-" * 60)

    # 转换为 PIL 图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # 加载中文字体
    font_path = Path(__file__).parent.parent / "fonts" / "MiSans" / "ttf" / "MiSans-Regular.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 20)
    except:
        font = ImageFont.load_default()

    # 颜色列表
    colors = [
        (255, 0, 0),    # 红
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 蓝
        (255, 165, 0),  # 橙
        (128, 0, 128),  # 紫
        (0, 255, 255),  # 青
        (255, 0, 255),  # 粉
        (255, 255, 0),  # 黄
    ]

    for i, box in enumerate(text_boxes):
        x1, y1, x2, y2 = box.bbox
        text = box.text
        color = colors[i % len(colors)]

        # 画矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 标签
        label = f"{i+1}: {text}"

        # 标签位置（框上方）
        label_y = y1 - 25
        if label_y < 5:
            label_y = y2 + 5

        # 画标签背景
        bbox = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)

        # 画标签文字（白色）
        draw.text((x1, label_y), label, font=font, fill=(255, 255, 255))

        # 打印信息
        print(f"[{i+1}] '{text}'")
        print(f"    bbox: [{x1}, {y1}, {x2}, {y2}]  size: {x2-x1}x{y2-y1}")

    print("-" * 60)

    # 保存结果到 output 目录
    if output_path is None:
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_ocr_boxes.jpg"

    # 转回 OpenCV 格式保存
    result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result)
    print(f"\n结果已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OCR 识别结果可视化")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("-o", "--output", help="输出图片路径")
    args = parser.parse_args()

    asyncio.run(test_ocr_visual(args.image, args.output))


if __name__ == "__main__":
    main()
