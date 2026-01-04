"""
DeepSeek 翻译客户端 - 电商语义翻译
"""
import httpx
from typing import List, Dict, Optional

from ..config import config
from ..models.schemas import TextBox


class Translator:
    """DeepSeek 翻译客户端"""

    # 语言名称映射
    LANGUAGE_NAMES = {
        "zh": "中文",
        "ko": "韩文",
        "ja": "日文",
        "en": "英文",
        "th": "泰文",
        "zh-TW": "繁体中文",
        "vi": "越南文",
        "id": "印尼文",
        "ms": "马来文",
    }

    def __init__(self):
        self.api_url = config.translator.api_url
        self.api_key = config.translator.api_key
        self.model = config.translator.model
        self.max_tokens = config.translator.max_tokens
        self.temperature = config.translator.temperature
        self.timeout = config.translator.timeout

    def _build_prompt(self, texts: List[str], target_lang: str) -> str:
        """构建电商语义翻译Prompt"""
        lang_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        prompt = f"""你是跨境电商商品文案翻译专家。请将以下中文商品卖点翻译为【{lang_name}】。

要求：
1. 保持简短精炼，适合商品图展示
2. 使用目标市场电商常见表达方式
3. 翻译后长度不要超过原文的1.5倍
4. 禁止解释性长句，保持卖点风格
5. 保留数字、单位、型号等不需要翻译的内容
6. 每行翻译一个文案，保持原有顺序

待翻译文案：
"""
        for i, text in enumerate(texts, 1):
            prompt += f"{i}. {text}\n"

        prompt += f"""
请直接输出翻译结果，每行一个，格式如下：
1. [翻译结果]
2. [翻译结果]
..."""

        return prompt

    def _parse_response(self, response_text: str, original_texts: List[str]) -> Dict[str, str]:
        """解析翻译响应"""
        translations = {}
        lines = response_text.strip().split("\n")

        for i, original in enumerate(original_texts):
            # 尝试匹配带编号的行
            for line in lines:
                line = line.strip()
                # 匹配格式: "1. xxx" 或 "1、xxx" 或 "1.xxx"
                if line.startswith(f"{i+1}.") or line.startswith(f"{i+1}、") or line.startswith(f"{i+1}:"):
                    # 移除编号前缀
                    translated = line.split(".", 1)[-1].split("、", 1)[-1].split(":", 1)[-1].strip()
                    translations[original] = translated
                    break
            else:
                # 如果没找到匹配，尝试按顺序取
                if i < len(lines):
                    line = lines[i].strip()
                    # 移除可能的编号前缀
                    for prefix in [f"{i+1}.", f"{i+1}、", f"{i+1}:"]:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    translations[original] = line
                else:
                    # 保留原文
                    translations[original] = original

        return translations

    async def translate(
        self,
        texts: List[str],
        target_lang: str = "ko",
        source_lang: str = "zh"
    ) -> Dict[str, str]:
        """
        批量翻译文字

        Args:
            texts: 待翻译的文字列表
            target_lang: 目标语言代码
            source_lang: 源语言代码

        Returns:
            翻译结果字典 {原文: 译文}
        """
        if not texts:
            return {}

        prompt = self._build_prompt(texts, target_lang)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        return self._parse_response(content, texts)

    async def translate_boxes(
        self,
        text_boxes: List[TextBox],
        target_lang: str = "ko",
        source_lang: str = "zh"
    ) -> List[TextBox]:
        """
        翻译文字框列表

        Args:
            text_boxes: 文字框列表
            target_lang: 目标语言
            source_lang: 源语言

        Returns:
            更新了translated_text的文字框列表
        """
        # 过滤出需要翻译的文字
        to_translate = [box for box in text_boxes if not box.skip]
        texts = [box.text for box in to_translate]

        if not texts:
            return text_boxes

        # 批量翻译
        translations = await self.translate(texts, target_lang, source_lang)

        # 更新翻译结果
        for box in text_boxes:
            if not box.skip and box.text in translations:
                box.translated_text = translations[box.text]
            elif box.skip:
                box.translated_text = None  # 跳过的不设置翻译

        return text_boxes

    def translate_sync(
        self,
        texts: List[str],
        target_lang: str = "ko",
        source_lang: str = "zh"
    ) -> Dict[str, str]:
        """同步版本的翻译方法"""
        import asyncio
        return asyncio.run(self.translate(texts, target_lang, source_lang))

    def translate_boxes_sync(
        self,
        text_boxes: List[TextBox],
        target_lang: str = "ko",
        source_lang: str = "zh"
    ) -> List[TextBox]:
        """同步版本的文字框翻译方法"""
        import asyncio
        return asyncio.run(self.translate_boxes(text_boxes, target_lang, source_lang))
