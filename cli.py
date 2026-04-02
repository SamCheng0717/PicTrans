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
        images = set()  # 使用 set 去重（Windows 文件系统不区分大小写会导致重复）
        for ext in extensions:
            images.update(path.glob(f"*{ext}"))
            images.update(path.glob(f"*{ext.upper()}"))
        return sorted(images)

    else:
        print(f"路径不存在: {input_path}")
        return []


async def process_images(
    images: List[Path],
    target_lang: str,
    source_lang: str,
    max_concurrent: int,
    inpaint_mode: str = "opencv",
    generate_compare: bool = False
):
    """处理图片列表"""
    # 创建任务列表
    tasks = [
        TranslationTask(
            image_path=str(img),
            source_lang=source_lang,
            target_lang=target_lang,
            inpaint_mode=inpaint_mode
        )
        for img in images
    ]

    print(f"\n开始处理 {len(tasks)} 张图片...")
    print(f"目标语言: {target_lang}")
    print(f"Inpaint 模式: {inpaint_mode}")
    print(f"输出目录: {config.output_dir}")
    print("-" * 50)

    # 为每个任务创建独立的 Pipeline 实例，避免并发时的状态竞争
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single(task):
        async with semaphore:
            # 每个任务使用独立的 Pipeline 实例
            pipeline = Pipeline(inpaint_mode=inpaint_mode)
            return await pipeline.process(task)

    # 批量处理
    results = await asyncio.gather(
        *[process_single(task) for task in tasks],
        return_exceptions=True
    )

    # 处理异常
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            from app.models.schemas import ProcessingResult
            processed_results.append(ProcessingResult(
                success=False,
                error_message=str(result)
            ))
        else:
            processed_results.append(result)

    # 统计结果
    success_count = 0
    fail_count = 0

    for i, (task, result) in enumerate(zip(tasks, processed_results)):
        image_name = Path(task.image_path).name
        if result.success:
            success_count += 1
            print(f"✓ [{i+1}/{len(tasks)}] {image_name}")
            print(f"  输出: {result.output_path}")
            print(f"  识别: {result.total_texts} 个文字, 翻译: {result.translated_texts}, 跳过: {result.skipped_texts}")
            print(f"  耗时: {result.total_time}ms (OCR:{result.ocr_time}ms, 翻译:{result.translate_time}ms, 渲染:{result.render_time}ms)")

            # 生成对比图（仅生成对比图，删除单独的翻译图）
            if generate_compare:
                try:
                    import cv2
                    import numpy as np
                    import os

                    # 读取原图和翻译后的图
                    original = cv2.imread(task.image_path)
                    translated = cv2.imread(result.output_path)

                    if original is not None and translated is not None:
                        # 水平拼接
                        comparison = np.hstack([original, translated])

                        # 生成对比图文件名（使用原翻译图的文件名，添加_compare后缀）
                        output_path = Path(result.output_path)
                        compare_name = output_path.stem + "_compare" + output_path.suffix
                        compare_path = output_path.parent / compare_name

                        # 保存对比图
                        cv2.imwrite(str(compare_path), comparison)

                        # 删除单独的翻译图
                        if os.path.exists(result.output_path):
                            os.remove(result.output_path)

                        # 更新输出路径为对比图路径
                        result.output_path = str(compare_path)
                        print(f"  对比图: {compare_path}")
                    else:
                        print(f"  对比图生成失败: 无法读取图片")
                except Exception as e:
                    print(f"  对比图生成失败: {e}")
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
  # 处理单张图片
  python cli.py input.jpg -t ko

  # 处理整个文件夹
  python cli.py ./images/ -t ko

  # 多语言输出
  python cli.py input.jpg -t ko -t ja -t en

  # 指定并发数
  python cli.py ./images/ -t ko -c 5

  # 自定义输出目录
  python cli.py input.jpg -t ko -o ./translated

  # 仅生成对比图（不保存单独的翻译图）
  python cli.py input.jpg -t ko --compare

  # 使用 IOPaint 模式并仅生成对比图
  python cli.py ./images/ -t ko --inpaint iopaint --compare
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
        choices=["opencv", "iopaint"],
        default="opencv",
        help="Inpaint 模式: opencv(默认) 或 iopaint(待实现)"
    )

    parser.add_argument(
        "--translator",
        choices=["hunyuan", "deepseek"],
        default=None,
        help="翻译后端: hunyuan(默认) 或 deepseek"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="仅生成原图与翻译图的对比图（水平拼接），不保存单独的翻译图"
    )

    args = parser.parse_args()

    # 处理语言参数
    target_langs = args.target_lang if args.target_lang else ["ko"]

    # 设置输出目录
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Inpaint 模式
    inpaint_mode = args.inpaint

    # 翻译后端切换
    if args.translator:
        config.translator.backend = args.translator

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
            max_concurrent=args.concurrent,
            inpaint_mode=inpaint_mode,
            generate_compare=args.compare
        ))

        total_success += success
        total_fail += fail

    if len(target_langs) > 1:
        print(f"\n{'='*50}")
        print(f"总计: 成功 {total_success}, 失败 {total_fail}")


if __name__ == "__main__":
    main()
