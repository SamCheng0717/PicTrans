"""
命令行批量处理工具
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from app.config import config
from app.models.schemas import TranslationTask
from app.core.pipeline import Pipeline


def find_images(input_path: str) -> List[Path]:
    """查找所有图片文件"""
    path = Path(input_path)
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

    if path.is_file():
        if path.suffix.lower() in extensions:
            return [path]
        else:
            print(f"不支持的文件格式: {path.suffix}")
            return []

    elif path.is_dir():
        images = []
        for ext in extensions:
            images.extend(path.glob(f"*{ext}"))
            images.extend(path.glob(f"*{ext.upper()}"))
        return sorted(images)

    else:
        print(f"路径不存在: {input_path}")
        return []


async def process_images(
    images: List[Path],
    target_lang: str,
    source_lang: str,
    skip_price: bool,
    skip_promo: bool,
    skip_brand: bool,
    skip_english: bool,
    max_concurrent: int,
    inpaint_mode: str = "opencv"
):
    """处理图片列表"""
    pipeline = Pipeline(inpaint_mode=inpaint_mode)

    # 创建任务列表
    tasks = [
        TranslationTask(
            image_path=str(img),
            source_lang=source_lang,
            target_lang=target_lang,
            skip_price=skip_price,
            skip_promo=skip_promo,
            skip_brand=skip_brand,
            skip_english=skip_english
        )
        for img in images
    ]

    print(f"\n开始处理 {len(tasks)} 张图片...")
    print(f"目标语言: {target_lang}")
    print(f"Inpaint 模式: {inpaint_mode}")
    print(f"输出目录: {config.output_dir}")
    print("-" * 50)

    # 批量处理
    results = await pipeline.process_batch(tasks, max_concurrent)

    # 统计结果
    success_count = 0
    fail_count = 0

    for i, (task, result) in enumerate(zip(tasks, results)):
        image_name = Path(task.image_path).name
        if result.success:
            success_count += 1
            print(f"✓ [{i+1}/{len(tasks)}] {image_name}")
            print(f"  输出: {result.output_path}")
            print(f"  识别: {result.total_texts} 个文字, 翻译: {result.translated_texts}, 跳过: {result.skipped_texts}")
            print(f"  耗时: {result.total_time}ms (OCR:{result.ocr_time}ms, 翻译:{result.translate_time}ms, 渲染:{result.render_time}ms)")
        else:
            fail_count += 1
            print(f"✗ [{i+1}/{len(tasks)}] {image_name}")
            print(f"  错误: {result.error_message}")

    print("-" * 50)
    print(f"处理完成: 成功 {success_count}, 失败 {fail_count}")

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="PicTrans - 电商图片多语言翻译工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单张图片（默认使用 OpenCV）
  python cli.py input.jpg -t ko

  # 处理整个文件夹
  python cli.py ./images/ -t ko

  # 翻译成日文，保留品牌
  python cli.py ./images/ -t ja --keep-brand

  # 多语言输出
  python cli.py input.jpg -t ko -t ja -t en

  # 使用 AI inpaint（需要 GPU 服务器）
  python cli.py input.jpg -t ko --inpaint qwen --qwen-url http://192.168.1.100:8765
        """
    )

    parser.add_argument(
        "input",
        help="输入图片或目录路径"
    )

    parser.add_argument(
        "-t", "--target-lang",
        action="append",
        default=[],
        help="目标语言代码，可多次指定 (ko/ja/en/th/zh-TW)"
    )

    parser.add_argument(
        "-s", "--source-lang",
        default="zh",
        help="源语言代码 (默认: zh)"
    )

    parser.add_argument(
        "--skip-price",
        action="store_true",
        default=True,
        help="跳过价格文字 (默认: 是)"
    )

    parser.add_argument(
        "--keep-price",
        action="store_true",
        help="保留价格文字"
    )

    parser.add_argument(
        "--skip-promo",
        action="store_true",
        default=True,
        help="跳过促销文字 (默认: 是)"
    )

    parser.add_argument(
        "--keep-promo",
        action="store_true",
        help="保留促销文字"
    )

    parser.add_argument(
        "--skip-brand",
        action="store_true",
        help="跳过品牌名"
    )

    parser.add_argument(
        "--keep-brand",
        action="store_true",
        default=True,
        help="保留品牌名 (默认: 是)"
    )

    parser.add_argument(
        "--skip-english",
        action="store_true",
        help="跳过英文文字翻译"
    )

    parser.add_argument(
        "--keep-english",
        action="store_true",
        default=True,
        help="保留英文文字翻译 (默认: 是)"
    )

    parser.add_argument(
        "-c", "--concurrent",
        type=int,
        default=3,
        help="并发处理数 (默认: 3)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        help="输出目录 (默认: ./output)"
    )

    parser.add_argument(
        "--inpaint",
        choices=["opencv", "qwen"],
        default="opencv",
        help="Inpaint 模式: opencv(默认,快速) 或 qwen(AI,需要GPU服务器)"
    )

    parser.add_argument(
        "--qwen-url",
        default=None,
        help="Qwen inpaint 服务器地址 (默认: http://localhost:8765)"
    )

    args = parser.parse_args()

    # 处理语言参数
    target_langs = args.target_lang if args.target_lang else ["ko"]

    # 处理过滤参数
    skip_price = not args.keep_price
    skip_promo = not args.keep_promo
    skip_brand = args.skip_brand
    skip_english = not args.keep_english

    # 设置输出目录
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # 设置 Qwen API URL
    if args.qwen_url:
        config.inpaint.qwen_api_url = args.qwen_url

    # Inpaint 模式
    inpaint_mode = args.inpaint

    # 查找图片
    images = find_images(args.input)
    if not images:
        print("未找到可处理的图片")
        sys.exit(1)

    print(f"找到 {len(images)} 张图片")

    # 对每种目标语言处理
    total_success = 0
    total_fail = 0

    for target_lang in target_langs:
        print(f"\n{'='*50}")
        print(f"目标语言: {target_lang}")
        print(f"{'='*50}")

        success, fail = asyncio.run(process_images(
            images=images,
            target_lang=target_lang,
            source_lang=args.source_lang,
            skip_price=skip_price,
            skip_promo=skip_promo,
            skip_brand=skip_brand,
            skip_english=skip_english,
            max_concurrent=args.concurrent,
            inpaint_mode=inpaint_mode
        ))

        total_success += success
        total_fail += fail

    if len(target_langs) > 1:
        print(f"\n{'='*50}")
        print(f"总计: 成功 {total_success}, 失败 {total_fail}")


if __name__ == "__main__":
    main()
