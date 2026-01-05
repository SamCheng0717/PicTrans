"""
Flask API 路由
"""
import os
import base64
import asyncio
import tempfile
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from ..config import config
from ..models.schemas import TranslationTask, APIRequest, APIResponse
from ..core.pipeline import Pipeline


api_bp = Blueprint("api", __name__, url_prefix="/api")

# 全局Pipeline实例
pipeline = Pipeline()

# 允许的图片扩展名
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@api_bp.route("/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "ok", "version": "1.0.0"})


@api_bp.route("/translate", methods=["POST"])
def translate_image():
    """
    翻译图片中的文字

    支持两种方式：
    1. multipart/form-data 上传文件
    2. application/json 传入base64
    """
    try:
        # 解析请求参数
        if request.content_type and "multipart/form-data" in request.content_type:
            # 文件上传方式
            if "image" not in request.files:
                return jsonify(APIResponse(
                    success=False,
                    message="缺少图片文件"
                ).__dict__), 400

            file = request.files["image"]
            if file.filename == "":
                return jsonify(APIResponse(
                    success=False,
                    message="未选择文件"
                ).__dict__), 400

            if not allowed_file(file.filename):
                return jsonify(APIResponse(
                    success=False,
                    message=f"不支持的文件格式，允许: {ALLOWED_EXTENSIONS}"
                ).__dict__), 400

            # 保存临时文件
            filename = secure_filename(file.filename)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            image_path = temp_path

            # 获取其他参数
            source_lang = request.form.get("source_lang", "zh")
            target_lang = request.form.get("target_lang", "ko")
            inpaint_mode = request.form.get("inpaint_mode", "opencv")
            skip_price = request.form.get("skip_price", "true").lower() == "true"
            skip_promo = request.form.get("skip_promo", "true").lower() == "true"
            skip_brand = request.form.get("skip_brand", "false").lower() == "true"
            skip_english = request.form.get("skip_english", "false").lower() == "true"
            skip_texts = request.form.get("skip_texts", "").split(",")
            skip_texts = [t.strip() for t in skip_texts if t.strip()]

        else:
            # JSON方式
            data = request.get_json()
            if not data:
                return jsonify(APIResponse(
                    success=False,
                    message="缺少请求数据"
                ).__dict__), 400

            image_data = data.get("image")
            if not image_data:
                return jsonify(APIResponse(
                    success=False,
                    message="缺少图片数据"
                ).__dict__), 400

            # 处理base64图片
            if image_data.startswith("data:image"):
                # 移除data URL前缀
                image_data = image_data.split(",", 1)[1]

            # 解码并保存临时文件
            try:
                image_bytes = base64.b64decode(image_data)
            except Exception:
                return jsonify(APIResponse(
                    success=False,
                    message="无效的base64图片数据"
                ).__dict__), 400

            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "input.png")
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            image_path = temp_path

            source_lang = data.get("source_lang", "zh")
            target_lang = data.get("target_lang", "ko")
            inpaint_mode = data.get("inpaint_mode", "opencv")
            skip_price = data.get("skip_price", True)
            skip_promo = data.get("skip_promo", True)
            skip_brand = data.get("skip_brand", False)
            skip_english = data.get("skip_english", False)
            skip_texts = data.get("skip_texts", [])

        # 创建任务
        task = TranslationTask(
            image_path=image_path,
            source_lang=source_lang,
            target_lang=target_lang,
            inpaint_mode=inpaint_mode,
            skip_price=skip_price,
            skip_promo=skip_promo,
            skip_brand=skip_brand,
            skip_english=skip_english,
            skip_texts=skip_texts
        )

        # 执行处理
        result = asyncio.run(pipeline.process(task))

        # 构建响应
        if result.success:
            # 转换文字框为字典
            detected = []
            translated = []
            for box in result.detected_texts:
                box_dict = {
                    "id": box.id,
                    "text": box.text,
                    "bbox": box.bbox,
                    "role": box.role.value,
                    "skip": box.skip
                }
                detected.append(box_dict)

                if box.translated_text:
                    translated.append({
                        "id": box.id,
                        "original": box.text,
                        "translated": box.translated_text
                    })

            response = APIResponse(
                success=True,
                message="处理成功",
                output_path=result.output_path,
                detected_texts=detected,
                translated_texts=translated,
                stats={
                    "total_texts": result.total_texts,
                    "translated_count": result.translated_texts,
                    "skipped_count": result.skipped_texts,
                    "ocr_time_ms": result.ocr_time,
                    "translate_time_ms": result.translate_time,
                    "render_time_ms": result.render_time,
                    "total_time_ms": result.total_time
                }
            )
        else:
            response = APIResponse(
                success=False,
                message=result.error_message or "处理失败"
            )

        return jsonify(response.__dict__)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(APIResponse(
            success=False,
            message=f"服务器错误: {str(e)}"
        ).__dict__), 500


@api_bp.route("/output/<filename>", methods=["GET"])
def get_output_file(filename: str):
    """获取输出文件"""
    file_path = config.output_dir / filename
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(file_path)


@api_bp.route("/languages", methods=["GET"])
def get_languages():
    """获取支持的语言列表"""
    return jsonify({
        "languages": config.supported_languages
    })
