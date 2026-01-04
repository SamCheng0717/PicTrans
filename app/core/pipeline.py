"""
主处理流水线 - 串联所有模块
"""
import time
import asyncio
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from ..config import config
from ..models.schemas import TextBox, TranslationTask, ProcessingResult, TextRole
from .ocr_client import OCRClient
from .translator import Translator
from .text_analyzer import TextAnalyzer
from .inpainter import Inpainter
from .text_renderer import TextRenderer


class Pipeline:
    """图片翻译处理流水线"""

    def __init__(self):
        self.ocr = OCRClient()
        self.translator = Translator()
        self.analyzer = TextAnalyzer()
        self.inpainter = Inpainter()

    async def process(self, task: TranslationTask) -> ProcessingResult:
        """
        处理单张图片

        Args:
            task: 翻译任务

        Returns:
            处理结果
        """
        start_time = time.time()
        result = ProcessingResult(success=False)

        try:
            # 1. 加载图片
            image_path = Path(task.image_path)
            if not image_path.exists():
                result.error_message = f"图片文件不存在: {task.image_path}"
                return result

            image = cv2.imread(str(image_path))
            if image is None:
                result.error_message = f"无法读取图片: {task.image_path}"
                return result

            # 2. OCR识别
            ocr_start = time.time()
            text_boxes = await self.ocr.recognize(image_path)
            result.ocr_time = int((time.time() - ocr_start) * 1000)
            result.total_texts = len(text_boxes)

            if not text_boxes:
                result.error_message = "未识别到文字"
                return result

            # 3. 过滤文字框
            text_boxes = self._filter_boxes(text_boxes, task)

            # 4. 分析文字特征
            text_boxes = self.analyzer.analyze_all(image, text_boxes)

            # 5. 翻译
            translate_start = time.time()
            text_boxes = await self.translator.translate_boxes(
                text_boxes,
                target_lang=task.target_lang,
                source_lang=task.source_lang
            )
            result.translate_time = int((time.time() - translate_start) * 1000)

            # 6. 擦除原文字
            render_start = time.time()
            inpaint_boxes = [b for b in text_boxes if not b.skip and b.translated_text]

            # 调试日志
            print(f"[Pipeline] 需要inpaint的文字框: {len(inpaint_boxes)}")
            for box in inpaint_boxes:
                print(f"  - {box.text} -> {box.translated_text} @ {box.bbox}")

            inpainted_image = self.inpainter.inpaint(image, inpaint_boxes)

            # 7. 渲染新文字（使用目标语言对应的字体）
            renderer = TextRenderer(target_lang=task.target_lang)
            final_image = renderer.render(inpainted_image, text_boxes)
            result.render_time = int((time.time() - render_start) * 1000)

            # 8. 保存结果
            output_path = self._generate_output_path(image_path, task.target_lang)
            cv2.imwrite(str(output_path), final_image)

            # 填充结果
            result.success = True
            result.output_path = str(output_path)
            result.detected_texts = text_boxes
            result.translated_texts = len([b for b in text_boxes if b.translated_text])
            result.skipped_texts = len([b for b in text_boxes if b.skip])
            result.total_time = int((time.time() - start_time) * 1000)

        except Exception as e:
            result.error_message = str(e)
            import traceback
            traceback.print_exc()

        return result

    def _filter_boxes(self, text_boxes: list, task: TranslationTask) -> list:
        """根据任务配置过滤文字框"""
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

    def _generate_output_path(self, input_path: Path, target_lang: str) -> Path:
        """生成输出文件路径"""
        output_dir = config.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{input_path.stem}_{target_lang}_{timestamp}{input_path.suffix}"
        return output_dir / output_name

    def process_sync(self, task: TranslationTask) -> ProcessingResult:
        """同步版本的处理方法"""
        return asyncio.run(self.process(task))

    async def process_batch(
        self,
        tasks: list,
        max_concurrent: int = 3
    ) -> list:
        """
        批量处理图片

        Args:
            tasks: 任务列表
            max_concurrent: 最大并发数

        Returns:
            结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(task):
            async with semaphore:
                return await self.process(task)

        results = await asyncio.gather(
            *[process_with_limit(task) for task in tasks],
            return_exceptions=True
        )

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    def process_batch_sync(self, tasks: list, max_concurrent: int = 3) -> list:
        """同步版本的批量处理"""
        return asyncio.run(self.process_batch(tasks, max_concurrent))
