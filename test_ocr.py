import asyncio
from app.core.ocr_client import OCRClient

async def test():
    ocr = OCRClient()
    boxes = await ocr.recognize('input/test.jpg')
    for b in boxes[:5]:
        print(f'{b.text}: {b.bbox} (w={b.width}, h={b.height})')

asyncio.run(test())