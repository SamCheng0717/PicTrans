"""
Flask 应用启动入口
"""
from flask import Flask
from flask_cors import CORS

from app.api import api_bp
from app.config import config


def create_app():
    """创建Flask应用"""
    app = Flask(__name__)

    # 启用CORS
    CORS(app)

    # 配置
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

    # 注册蓝图
    app.register_blueprint(api_bp)

    # 强制关闭连接，避免反向代理复用已关闭连接
    @app.after_request
    def add_connection_close(response):
        response.headers["Connection"] = "close"
        return response

    # 根路由
    @app.route("/")
    def index():
        return {
            "name": "PicTrans API",
            "version": "1.0.0",
            "description": "电商图片多语言翻译系统",
            "endpoints": {
                "health": "/api/health",
                "translate": "/api/translate",
                "languages": "/api/languages",
                "output": "/api/output/<filename>"
            }
        }

    return app


# 供 uWSGI 直接引用的顶层变量
app = create_app()

if __name__ == "__main__":
    print(f"输出目录: {config.output_dir}")
    print("启动服务: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
