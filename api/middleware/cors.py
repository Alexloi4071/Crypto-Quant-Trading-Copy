"""
CORS Middleware
跨域资源共享中间件
处理跨域请求和预检请求
"""

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.config import api_config
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

class CustomCORSMiddleware:
    """自定义CORS中间件"""

    def __init__(self, app):
        self.app = app
        self.allowed_origins = api_config.CORS_ORIGINS
        self.allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-User-ID",
            "Cache-Control"
        ]
        self.expose_headers = [
            "X-User-ID",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset"
        ]

        logger.info("CORS中间件初始化完成")
        logger.info(f"允许的源: {self.allowed_origins}")

    async def __call__(self, request: Request, call_next):
        """处理CORS请求"""

        # 获取请求来源
        origin = request.headers.get("Origin")

        # 预检请求处理
        if request.method == "OPTIONS":
            return self.handle_preflight_request(origin)

        # 处理正常请求
        response = await call_next(request)

        # 添加CORS头
        return self.add_cors_headers(response, origin)

    def handle_preflight_request(self, origin: str) -> Response:
        """处理预检请求"""

        # 检查origin是否被允许
        if not self.is_origin_allowed(origin):
            return JSONResponse(
                status_code=403,
                content={"error": "Origin not allowed"},
                headers={"Vary": "Origin"}
            )

        # 创建预检响应
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods": ", ".join(self.allowed_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allowed_headers),
            "Access-Control-Max-Age": "86400",  # 24小时
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin"
        }

        return Response(status_code=200, headers=headers)

    def add_cors_headers(self, response: Response, origin: str) -> Response:
        """添加CORS响应头"""

        if self.is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
            response.headers["Vary"] = "Origin"

        return response

    def is_origin_allowed(self, origin: str) -> bool:
        """检查origin是否被允许"""
        if not origin:
            return True  # 同源请求

        # 检查通配符
        if "*" in self.allowed_origins:
            return True

        # 检查精确匹配
        if origin in self.allowed_origins:
            return True

        # 检查模式匹配（例如 *.example.com）
        for allowed_origin in self.allowed_origins:
            if allowed_origin.startswith("*."):
                domain_pattern = allowed_origin[2:]  # 移除 "*."
                if origin.endswith(f".{domain_pattern}") or origin == domain_pattern:
                    return True

        return False

class DynamicCORSConfig:
    """动态CORS配置"""

    @staticmethod

    def get_development_origins():
        """获取开发环境允许的源"""
        return [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
            "http://localhost:5173",  # Vite dev server
            "http://localhost:4200",  # Angular dev server
        ]

    @staticmethod

    def get_production_origins():
        """获取生产环境允许的源"""
        return [
            "https://yourdomain.com",
            "https://api.yourdomain.com",
            "https://dashboard.yourdomain.com"
        ]

    @classmethod

    def get_cors_config(cls):
        """获取CORS配置"""
        if api_config.is_development():
            origins = cls.get_development_origins()
        else:
            origins = cls.get_production_origins()

        # 合并用户配置的源
        if hasattr(api_config, 'CORS_ORIGINS') and api_config.CORS_ORIGINS:
            if isinstance(api_config.CORS_ORIGINS, list):
                origins.extend(api_config.CORS_ORIGINS)
            elif api_config.CORS_ORIGINS == "*":
                origins = ["*"]

        return {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": [
                "Accept",
                "Accept-Language",
                "Content-Language",
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-User-ID",
                "Cache-Control",
                "X-CSRF-Token"
            ],
            "expose_headers": [
                "X-User-ID",
                "X-Rate-Limit-Remaining",
                "X-Rate-Limit-Reset",
                "X-Request-ID"
            ]
        }

# 辅助函数

def setup_cors_middleware(app):
    """设置CORS中间件"""

    cors_config = DynamicCORSConfig.get_cors_config()

    logger.info("设置CORS中间件")
    logger.info(f"允许的源: {cors_config['allow_origins']}")

    app.add_middleware(
        CORSMiddleware,
        **cors_config
    )

    return app

def create_cors_headers(origin: str = None) -> dict:
    """创建CORS响应头"""
    headers = {}

    cors_config = DynamicCORSConfig.get_cors_config()

    if origin and (origin in cors_config["allow_origins"] or "*" in cors_config["allow_origins"]):
        headers.update({
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Expose-Headers": ", ".join(cors_config["expose_headers"]),
            "Vary": "Origin"
        })

    return headers

def handle_cors_preflight(request: Request) -> Response:
    """处理CORS预检请求"""
    origin = request.headers.get("Origin")

    cors_config = DynamicCORSConfig.get_cors_config()

    if not origin or (origin not in cors_config["allow_origins"] and "*" not in cors_config["allow_origins"]):
        return JSONResponse(
            status_code=403,
            content={"error": "Origin not allowed"}
        )

    headers = {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": ", ".join(cors_config["allow_methods"]),
        "Access-Control-Allow-Headers": ", ".join(cors_config["allow_headers"]),
        "Access-Control-Max-Age": "86400",
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin"
    }

    return Response(status_code=200, headers=headers)

# 装饰器

def cors_enabled(func):
    """CORS启用装饰器"""

    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)

        # 如果response是JSONResponse，添加CORS头
        if hasattr(response, 'headers'):
            cors_headers = create_cors_headers()
            for key, value in cors_headers.items():
                response.headers[key] = value

        return response

    return wrapper
