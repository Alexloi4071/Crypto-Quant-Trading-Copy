"""
Authentication Middleware
认证中间件，提供JWT认证和权限控制
集成系统安全认证功能
"""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import jwt
from datetime import datetime, timedelta
from typing import Optional
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入配置
from api.config import api_config
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

class AuthMiddleware:
    """认证中间件类"""

    def __init__(self, app):
        self.app = app
        self.secret_key = api_config.SECRET_KEY
        self.algorithm = "HS256"
        self.token_expire_minutes = api_config.ACCESS_TOKEN_EXPIRE_MINUTES

        # 免认证路径
        self.exempt_paths = [
            "/",
            "/health",
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static",
            "/favicon.ico"
        ]

        # 只读权限路径（不需要write权限）
        self.readonly_paths = [
            "/api/v1/portfolio/summary",
            "/api/v1/portfolio/positions",
            "/api/v1/portfolio/performance",
            "/api/v1/portfolio/risk-assessment",
            "/api/v1/portfolio/history",
            "/api/v1/trading/status",
            "/api/v1/signals",
            "/api/v1/data",
            "/api/v1/monitoring"
        ]

        logger.info("认证中间件初始化完成")

    async def __call__(self, request: Request, call_next):
        """中间件处理函数"""
        try:
            # 检查是否需要认证
            if not api_config.ENABLE_AUTH:
                response = await call_next(request)
                return response

            # 检查是否为免认证路径
            request_path = str(request.url.path)
            if any(request_path.startswith(path) for path in self.exempt_paths):
                response = await call_next(request)
                return response

            # 提取和验证token
            token = self._extract_token(request)
            if not token:
                return self._create_auth_error_response("Missing authentication token")

            # 验证token
            user_info = self._verify_token(token)
            if not user_info:
                return self._create_auth_error_response("Invalid or expired token")

            # 检查权限
            required_permission = self._get_required_permission(request)
            if required_permission and not self._check_permission(user_info, required_permission):
                return self._create_permission_error_response(f"Insufficient permissions: {required_permission}")

            # 将用户信息添加到request state
            request.state.current_user = user_info

            # 继续处理请求
            response = await call_next(request)

            # 添加认证相关的响应头
            response.headers["X-User-ID"] = user_info.get("user_id", "unknown")

            return response

        except Exception as e:
            logger.error(f"认证中间件错误: {e}")
            return self._create_auth_error_response("Authentication error")

    def _extract_token(self, request: Request) -> Optional[str]:
        """从请求中提取token"""
        # 从Authorization header提取
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            return authorization.split(" ")[1]

        # 从query参数提取（用于WebSocket等场景）
        token = request.query_params.get("token")
        if token:
            return token

        return None

    def _verify_token(self, token: str) -> Optional[dict]:
        """验证JWT token"""
        try:
            # 解码token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )

            # 检查过期时间
            exp_timestamp = payload.get("exp")
            if exp_timestamp and datetime.utcnow().timestamp() > exp_timestamp:
                logger.warning("Token已过期")
                return None

            # 返回用户信息
            return {
                "user_id": payload.get("user_id"),
                "username": payload.get("username"),
                "permissions": payload.get("permissions", []),
                "exp": exp_timestamp
            }

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"JWT token无效: {e}")
            return None
        except Exception as e:
            logger.error(f"验证token失败: {e}")
            return None

    def _get_required_permission(self, request: Request) -> Optional[str]:
        """根据请求路径和方法确定所需权限"""
        request_path = str(request.url.path)
        method = request.method.upper()

        # GET请求一般只需要read权限
        if method == "GET":
            return "read"

        # POST, PUT, DELETE请求需要write权限
        if method in ["POST", "PUT", "DELETE"]:
            # 检查是否为交易相关操作
            if "/trading/" in request_path or "/positions/" in request_path:
                return "trade"  # 交易权限
            else:
                return "write"  # 写权限

        return "read"  # 默认需要读权限

    def _check_permission(self, user_info: dict, required_permission: str) -> bool:
        """检查用户权限"""
        user_permissions = user_info.get("permissions", [])

        # admin权限包含所有权限
        if "admin" in user_permissions:
            return True

        # 检查具体权限
        if required_permission in user_permissions:
            return True

        # write权限包含read权限
        if required_permission == "read" and "write" in user_permissions:
            return True

        # trade权限包含write和read权限
        if required_permission in ["read", "write"] and "trade" in user_permissions:
            return True

        return False

    def _create_auth_error_response(self, message: str):
        """创建认证错误响应"""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error": {
                    "code": "AUTHENTICATION_REQUIRED",
                    "message": message
                },
                "timestamp": datetime.now().isoformat() + "Z"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    def _create_permission_error_response(self, message: str):
        """创建权限错误响应"""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "error": {
                    "code": "INSUFFICIENT_PERMISSIONS",
                    "message": message
                },
                "timestamp": datetime.now().isoformat() + "Z"
            }
        )

class JWTManager:
    """JWT管理器"""

    def __init__(self):
        self.secret_key = api_config.SECRET_KEY
        self.algorithm = "HS256"
        self.token_expire_minutes = api_config.ACCESS_TOKEN_EXPIRE_MINUTES

    def create_access_token(self, user_id: str, username: str, permissions: list) -> str:
        """创建访问token"""
        expire = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)

        payload = {
            "user_id": user_id,
            "username": username,
            "permissions": permissions,
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "access"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_api_key(self, user_id: str, permissions: list, expires_days: int = 30) -> str:
        """创建API密钥（长期有效的token）"""
        expire = datetime.utcnow() + timedelta(days=expires_days)

        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": expire.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "type": "api_key"
        }

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """验证token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # 检查是否过期
            if datetime.utcnow().timestamp() > payload.get("exp", 0):
                return None

            return payload

        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """刷新token"""
        payload = self.verify_token(token)
        if not payload:
            return None

        # 创建新token
        return self.create_access_token(
            user_id=payload.get("user_id"),
            username=payload.get("username", payload.get("user_id")),
            permissions=payload.get("permissions", [])
        )

# JWT管理器实例
jwt_manager = JWTManager()

class SimpleAuthenticator:
    """简单认证器（用于开发和测试）"""

    def __init__(self):
        # 预定义用户（实际应用中应该从数据库获取）
        self.users = {
            "admin": {
                "user_id": "admin",
                "username": "admin",
                "password": "admin123",  # 实际应用中应该加密存储
                "permissions": ["admin", "trade", "write", "read"]
            },
            "trader": {
                "user_id": "trader",
                "username": "trader",
                "password": "trader123",
                "permissions": ["trade", "write", "read"]
            },
            "viewer": {
                "user_id": "viewer",
                "username": "viewer",
                "password": "viewer123",
                "permissions": ["read"]
            }
        }

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """认证用户"""
        user = self.users.get(username)
        if user and user["password"] == password:
            return {
                "user_id": user["user_id"],
                "username": user["username"],
                "permissions": user["permissions"]
            }
        return None

    def get_user(self, user_id: str) -> Optional[dict]:
        """获取用户信息"""
        for user in self.users.values():
            if user["user_id"] == user_id:
                return {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "permissions": user["permissions"]
                }
        return None

# 认证器实例
authenticator = SimpleAuthenticator()

# 认证相关的工具函数

def generate_test_token(user_type: str = "admin") -> str:
    """生成测试token"""
    user_configs = {
        "admin": {
            "user_id": "admin",
            "username": "admin",
            "permissions": ["admin", "trade", "write", "read"]
        },
        "trader": {
            "user_id": "trader",
            "username": "trader",
            "permissions": ["trade", "write", "read"]
        },
        "viewer": {
            "user_id": "viewer",
            "username": "viewer",
            "permissions": ["read"]
        }
    }

    config = user_configs.get(user_type, user_configs["viewer"])

    return jwt_manager.create_access_token(
        user_id=config["user_id"],
        username=config["username"],
        permissions=config["permissions"]
    )

def get_auth_headers(user_type: str = "admin") -> dict:
    """获取认证头部（用于测试）"""
    if not api_config.ENABLE_AUTH:
        return {}

    token = generate_test_token(user_type)
    return {"Authorization": f"Bearer {token}"}

# 权限验证装饰器
from functools import wraps
from fastapi import Depends

def require_auth(permissions: list = None):
    """权限验证装饰器"""

    def decorator(func):
        @wraps(func)

        async def wrapper(*args, **kwargs):
            # 这里可以添加权限检查逻辑
            return await func(*args, **kwargs)
        return wrapper
    return decorator
