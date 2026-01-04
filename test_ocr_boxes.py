"""
OCR坐标可视化测试脚本
在原图上画框标记识别到的文字位置，用于调试坐标是否正确
"""
import asyncio
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.core.ocr_client import OCRClient
from app.config import config


# 用于画框的颜色列表 (BGR格式)
COLORS = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 0, 255),    # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 黄色
    (128, 255, 0),  # 浅绿
    (255, 128, 0),  # 浅蓝
    (0, 128, 255),  # 橙色
    (128, 0, 255),  # 粉色
]


def draw_boxes_on_image(image_path: str, output_path: str = None):
    """
    执行OCR并在图片上画框标记文字位置

    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径，默认为 output/debug_ocr.jpg
    """
    # 设置输出路径
    if output_path is None:
        output_path = config.output_dir / "debug_ocr.jpg"

    # 加载图片
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"错误: 无法读取图片 {image_path}")
        return

    h, w = image.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 执行OCR
    print("\n正在执行OCR识别...")
    ocr = OCRClient()
    text_boxes = asyncio.run(ocr.recognize(image_path))

    print(f"\n识别到 {len(text_boxes)} 个文字区域:\n")

    # 在图片上画框
    result = image.copy()

    for i, box in enumerate(text_boxes):
        x1, y1, x2, y2 = box.bbox
        color = COLORS[i % len(COLORS)]

        # 画矩形框 (线宽3)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

        # 计算框的尺寸
        box_w = x2 - x1
        box_h = y2 - y1

        # 打印信息
        print(f"[{i+1:2d}] {box.text}")
        print(f"     坐标: [{x1}, {y1}, {x2}, {y2}]")
        print(f"     尺寸: {box_w} x {box_h}")
        print()

        # 在框上方标注序号
        label = f"{i+1}"
        cv2.putText(result, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # 保存结果
    cv2.imwrite(str(output_path), result)
    print(f"调试图片已保存到: {output_path}")

    # 同时生成一个带中文标注的版本 (使用PIL)
    output_path_cn = Path(output_path).with_name("debug_ocr_labeled.jpg")
    draw_boxes_with_chinese_labels(image, text_boxes, output_path_cn)


def draw_boxes_with_chinese_labels(image: np.ndarray, text_boxes: list, output_path: Path):
    """
    使用PIL在图片上画框并标注中文
    """
    # 转换为PIL图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # 尝试加载中文字体
    try:
        font_path = config.font.get_font_path("regular", "zh")
        font = ImageFont.truetype(str(font_path), 24)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(text_boxes):
        x1, y1, x2, y2 = box.bbox
        color = COLORS[i % len(COLORS)]
        # PIL颜色是RGB格式
        color_rgb = (color[2], color[1], color[0])

        # 画矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

        # 标注文字内容
        label = f"{i+1}. {box.text}"
        # 在框上方绘制标签背景
        bbox = draw.textbbox((x1, y1 - 30), label, font=font)
        draw.rectangle(bbox, fill=(255, 255, 255))
        draw.text((x1, y1 - 30), label, fill=color_rgb, font=font)

    # 保存
    pil_image.save(str(output_path), quality=95)
    print(f"带中文标注的调试图片已保存到: {output_path}")


if __name__ == "__main__":
    import sys

    # 默认测试图片
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "input/test.jpg"

    print("=" * 50)
    print("OCR坐标可视化测试")
    print("=" * 50)

    draw_boxes_on_image(image_path)
