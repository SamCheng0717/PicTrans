"""
测试韩文字体渲染
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 字体路径
font_path = Path("E:/PicTrans/fonts/AlibabaSansKR/AlibabaSansKR-Regular/AlibabaSansKR-Regular.ttf")

print(f"字体文件存在: {font_path.exists()}")
print(f"字体路径: {font_path}")

# 测试文字
test_texts = [
    "안녕하세요",  # 韩文
    "Hello",       # 英文
    "你好",        # 中文
]

# 创建测试图像
img = Image.new("RGB", (400, 200), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype(str(font_path), 32)
    print(f"字体加载成功: {font.getname()}")

    y = 20
    for text in test_texts:
        draw.text((20, y), text, font=font, fill=(0, 0, 0))
        print(f"渲染: {text}")
        y += 50

    # 保存
    output_path = "output/test_font.png"
    img.save(output_path)
    print(f"\n测试图片已保存到: {output_path}")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
