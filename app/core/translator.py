"""
DeepSeek 翻译客户端 - 电商语义翻译
"""

import httpx
from typing import List, Dict, Optional

from ..config import config
from ..models.schemas import TextBox


class Translator:
    """翻译客户端 - 支持混元/DeepSeek 后端"""

    # 语言名称映射
    LANGUAGE_NAMES = {
        "zh": "中文", "en": "英语", "fr": "法语", "pt": "葡萄牙语",
        "es": "西班牙语", "ja": "日语", "tr": "土耳其语", "ru": "俄语",
        "ar": "阿拉伯语", "ko": "韩语", "th": "泰语", "it": "意大利语",
        "de": "德语", "vi": "越南语", "ms": "马来语", "id": "印尼语",
        "tl": "菲律宾语", "hi": "印地语", "zh-Hant": "繁体中文",
        "pl": "波兰语", "cs": "捷克语", "nl": "荷兰语", "km": "高棉语",
        "my": "缅甸语", "fa": "波斯语", "gu": "古吉拉特语", "ur": "乌尔都语",
        "te": "泰卢固语", "mr": "马拉地语", "he": "希伯来语", "bn": "孟加拉语",
        "ta": "泰米尔语", "uk": "乌克兰语", "bo": "藏语", "kk": "哈萨克语",
        "mn": "蒙古语", "ug": "维吾尔语", "yue": "粤语",
    }

    def __init__(self, backend: str = None):
        cfg = config.translator
        self.backend = backend or cfg.backend

        if self.backend == "hunyuan":
            self.api_url = cfg.hunyuan_api_url
            self.api_key = cfg.hunyuan_api_key
            self.model = cfg.hunyuan_model
        else:
            self.api_url = cfg.deepseek_api_url
            self.api_key = cfg.deepseek_api_key
            self.model = cfg.deepseek_model

        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.timeout = cfg.timeout

    def _build_prompt(self, texts: List[str], target_lang: str) -> str:
        """构建翻译Prompt（根据后端选择不同风格）"""
        lang_name = self.LANGUAGE_NAMES.get(target_lang, target_lang)

        if self.backend == "hunyuan":
            source_text = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
            prompt = f"将以下文本翻译为{lang_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}\n\n请按编号输出翻译结果："
        else:
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
            prompt += "\n请直接输出翻译结果，每行一个，格式：\n1. [翻译结果]\n2. [翻译结果]\n..."

        return prompt

    def _parse_response(
        self, response_text: str, original_texts: List[str]
    ) -> Dict[str, str]:
        """解析翻译响应"""
        # print(response_text)
        translations = {}
        lines = response_text.strip().split("\n")

        for i, original in enumerate(original_texts):
            translated = None

            # 尝试匹配带编号的行
            for line in lines:
                line = line.strip()
                # 匹配格式: "1. xxx" 或 "1、xxx" 或 "1: xxx"
                prefix_dot = f"{i + 1}."
                prefix_chinese = f"{i + 1}、"
                prefix_colon = f"{i + 1}:"

                if line.startswith(prefix_dot):
                    translated = line[len(prefix_dot) :].strip()
                    break
                elif line.startswith(prefix_chinese):
                    translated = line[len(prefix_chinese) :].strip()
                    break
                elif line.startswith(prefix_colon):
                    translated = line[len(prefix_colon) :].strip()
                    break
            else:
                # 如果没找到匹配，尝试按顺序取
                if i < len(lines):
                    line = lines[i].strip()
                    # 移除可能的编号前缀
                    for prefix in [f"{i + 1}.", f"{i + 1}、", f"{i + 1}:"]:
                        if line.startswith(prefix):
                            line = line[len(prefix) :].strip()
                            break
                    translated = line

            # 兜底：如果翻译为空或None，使用原文
            if not translated:
                translated = original
                print(f"[Translator] 警告: '{original}' 翻译为空，使用原文")

            translations[original] = translated

        return translations

    async def translate(
        self, texts: List[str], target_lang: str = "ko", source_lang: str = "zh"
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
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        return self._parse_response(content, texts)

    async def translate_boxes(
        self,
        text_boxes: List[TextBox],
        target_lang: str = "ko",
        source_lang: str = "zh",
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
        self, texts: List[str], target_lang: str = "ko", source_lang: str = "zh"
    ) -> Dict[str, str]:
        """同步版本的翻译方法"""
        import asyncio

        return asyncio.run(self.translate(texts, target_lang, source_lang))

    def translate_boxes_sync(
        self,
        text_boxes: List[TextBox],
        target_lang: str = "ko",
        source_lang: str = "zh",
    ) -> List[TextBox]:
        """同步版本的文字框翻译方法"""
        import asyncio

        return asyncio.run(self.translate_boxes(text_boxes, target_lang, source_lang))
