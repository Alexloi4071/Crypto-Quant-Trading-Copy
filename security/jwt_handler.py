"""
JWT Handler
JWT令牌处理器，提供JWT令牌的生成、验证和管理
支持多种算法和安全特性
"""

import jwt
import time
import json
import secrets
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class TokenType(Enum):
    """令牌类型"""
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"
    TEMPORARY = "temporary"

class JWTAlgorithm(Enum):
    """JWT算法"""
    HS256 = "HS256"
    HS384 = "HS384"
    HS512 = "HS512"
    RS256 = "RS256"
    RS384 = "RS384"
    RS512 = "RS512"

@dataclass

class TokenConfig:
    """令牌配置"""
    token_type: TokenType
    algorithm: JWTAlgorithm = JWTAlgorithm.HS256
    expires_in_seconds: int = 3600  # 1小时
    issuer: str = "trading_system"
    audience: Optional[str] = None

    # 刷新令牌配置
    refresh_expires_in_seconds: int = 86400 * 7  # 7天
    allow_refresh: bool = True

    # 安全配置
    include_user_fingerprint: bool = True
    require_https: bool = True

@dataclass

class TokenClaims:
    """令牌声明"""
    # 标准声明 (RFC 7519)
    sub: str  # Subject (user_id)
    iss: str  # Issuer
    aud: Optional[str] = None  # Audience
    exp: Optional[int] = None  # Expiration Time
    nbf: Optional[int] = None  # Not Before
    iat: Optional[int] = None  # Issued At
    jti: Optional[str] = None  # JWT ID

    # 自定义声明
    token_type: TokenType = TokenType.ACCESS
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)

    # 安全声明
    user_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent_hash: Optional[str] = None
    session_id: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        data = {
            'sub': self.sub,
            'iss': self.iss,
            'token_type': self.token_type.value,
        }

        # 添加可选字段
        if self.aud:
            data['aud'] = self.aud
        if self.exp:
            data['exp'] = self.exp
        if self.nbf:
            data['nbf'] = self.nbf
        if self.iat:
            data['iat'] = self.iat
        if self.jti:
            data['jti'] = self.jti
        if self.user_id:
            data['user_id'] = self.user_id
        if self.username:
            data['username'] = self.username
        if self.email:
            data['email'] = self.email
        if self.roles:
            data['roles'] = self.roles
        if self.permissions:
            data['permissions'] = self.permissions
        if self.user_fingerprint:
            data['user_fingerprint'] = self.user_fingerprint
        if self.ip_address:
            data['ip_address'] = self.ip_address
        if self.user_agent_hash:
            data['user_agent_hash'] = self.user_agent_hash
        if self.session_id:
            data['session_id'] = self.session_id
        if self.metadata:
            data['metadata'] = self.metadata

        return data

    @classmethod

    def from_dict(cls, data: dict) -> 'TokenClaims':
        """从字典创建"""
        return cls(
            sub=data.get('sub', ''),
            iss=data.get('iss', ''),
            aud=data.get('aud'),
            exp=data.get('exp'),
            nbf=data.get('nbf'),
            iat=data.get('iat'),
            jti=data.get('jti'),
            token_type=TokenType(data.get('token_type', 'access')),
            user_id=data.get('user_id'),
            username=data.get('username'),
            email=data.get('email'),
            roles=data.get('roles', []),
            permissions=data.get('permissions', []),
            user_fingerprint=data.get('user_fingerprint'),
            ip_address=data.get('ip_address'),
            user_agent_hash=data.get('user_agent_hash'),
            session_id=data.get('session_id'),
            metadata=data.get('metadata', {})
        )

@dataclass

class TokenValidationResult:
    """令牌验证结果"""
    is_valid: bool
    claims: Optional[TokenClaims] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # 验证详情
    is_expired: bool = False
    is_not_yet_valid: bool = False
    signature_valid: bool = True
    fingerprint_match: bool = True

class JWTKeyManager:
    """JWT密钥管理器"""

    def __init__(self):
        self.signing_keys = {}  # algorithm -> key
        self.verification_keys = {}  # algorithm -> key
        self.key_rotation_schedule = {}  # algorithm -> next_rotation_time

        # 从配置加载密钥
        self._load_keys_from_config()

    def _load_keys_from_config(self):
        """从配置加载密钥"""
        jwt_config = config.get('jwt', {})

        # 对称密钥 (HMAC)
        hmac_secret = jwt_config.get('secret_key', self._generate_secret_key())
        self.signing_keys[JWTAlgorithm.HS256] = hmac_secret
        self.signing_keys[JWTAlgorithm.HS384] = hmac_secret
        self.signing_keys[JWTAlgorithm.HS512] = hmac_secret

        # 非对称密钥 (RSA) - 如果配置了的话
        private_key = jwt_config.get('private_key')
        public_key = jwt_config.get('public_key')

        if private_key and public_key:
            self.signing_keys[JWTAlgorithm.RS256] = private_key
            self.signing_keys[JWTAlgorithm.RS384] = private_key
            self.signing_keys[JWTAlgorithm.RS512] = private_key

            self.verification_keys[JWTAlgorithm.RS256] = public_key
            self.verification_keys[JWTAlgorithm.RS384] = public_key
            self.verification_keys[JWTAlgorithm.RS512] = public_key

    def _generate_secret_key(self, length: int = 64) -> str:
        """生成随机密钥"""
        return secrets.token_urlsafe(length)

    def get_signing_key(self, algorithm: JWTAlgorithm) -> Optional[str]:
        """获取签名密钥"""
        return self.signing_keys.get(algorithm)

    def get_verification_key(self, algorithm: JWTAlgorithm) -> Optional[str]:
        """获取验证密钥"""
        # 对称算法使用相同的密钥
        if algorithm.value.startswith('HS'):
            return self.signing_keys.get(algorithm)

        # 非对称算法使用公钥验证
        return self.verification_keys.get(algorithm)

    def rotate_key(self, algorithm: JWTAlgorithm):
        """轮换密钥"""
        if algorithm.value.startswith('HS'):
            # 对称密钥轮换
            new_key = self._generate_secret_key()
            old_key = self.signing_keys.get(algorithm)

            self.signing_keys[algorithm] = new_key

            logger.info(f"轮换JWT密钥: {algorithm.value}")

            # 可以保留旧密钥一段时间用于验证现有令牌
            return old_key
        else:
            # 非对称密钥轮换需要重新生成密钥对
            logger.warning(f"非对称密钥轮换需要手动操作: {algorithm.value}")

class TokenBlacklist:
    """令牌黑名单"""

    def __init__(self):
        self.blacklisted_tokens = set()  # jti -> expiration_time
        self.blacklisted_users = {}  # user_id -> blacklist_time

    def blacklist_token(self, jti: str, expiration_time: datetime = None):
        """将令牌加入黑名单"""
        if expiration_time is None:
            expiration_time = datetime.now() + timedelta(days=30)  # 默认30天

        self.blacklisted_tokens.add((jti, expiration_time.timestamp()))
        logger.debug(f"令牌加入黑名单: {jti}")

    def blacklist_user(self, user_id: str, blacklist_time: datetime = None):
        """将用户的所有令牌加入黑名单"""
        if blacklist_time is None:
            blacklist_time = datetime.now()

        self.blacklisted_users[user_id] = blacklist_time
        logger.info(f"用户所有令牌加入黑名单: {user_id}")

    def is_token_blacklisted(self, jti: str) -> bool:
        """检查令牌是否在黑名单中"""
        current_time = time.time()

        # 清理过期的黑名单条目
        self.blacklisted_tokens = {
            (token_jti, exp_time) for token_jti, exp_time in self.blacklisted_tokens
            if exp_time > current_time
        }

        # 检查是否在黑名单中
        for token_jti, _ in self.blacklisted_tokens:
            if token_jti == jti:
                return True

        return False

    def is_user_blacklisted(self, user_id: str, issued_at: datetime) -> bool:
        """检查用户令牌是否在黑名单中"""
        if user_id not in self.blacklisted_users:
            return False

        blacklist_time = self.blacklisted_users[user_id]
        return issued_at < blacklist_time

    def remove_user_blacklist(self, user_id: str):
        """移除用户黑名单"""
        if user_id in self.blacklisted_users:
            del self.blacklisted_users[user_id]
            logger.info(f"移除用户黑名单: {user_id}")

class JWTHandler:
    """JWT处理器主类"""

    def __init__(self):
        self.key_manager = JWTKeyManager()
        self.blacklist = TokenBlacklist()

        # 默认配置
        self.default_config = TokenConfig(
            token_type=TokenType.ACCESS,
            algorithm=JWTAlgorithm.HS256,
            expires_in_seconds=config.get('jwt', {}).get('access_token_expires', 3600),
            issuer=config.get('jwt', {}).get('issuer', 'trading_system'),
            audience=config.get('jwt', {}).get('audience')
        )

        # 令牌缓存和统计
        self.token_cache = {}  # jti -> (token, claims, created_at)
        self.stats = {
            'tokens_generated': 0,
            'tokens_validated': 0,
            'validation_errors': 0,
            'blacklisted_tokens': 0
        }

        logger.info("JWT处理器初始化完成")

    def generate_token(self, claims: TokenClaims, config: TokenConfig = None) -> str:
        """生成JWT令牌"""
        config = config or self.default_config

        # 设置时间声明
        now = int(time.time())
        claims.iat = now
        claims.exp = now + config.expires_in_seconds
        claims.nbf = now  # 立即生效

        # 设置发行人和受众
        claims.iss = config.issuer
        if config.audience:
            claims.aud = config.audience

        # 生成唯一ID
        if not claims.jti:
            claims.jti = secrets.token_urlsafe(16)

        try:
            # 获取签名密钥
            signing_key = self.key_manager.get_signing_key(config.algorithm)
            if not signing_key:
                raise ValueError(f"未找到算法 {config.algorithm.value} 的签名密钥")

            # 生成令牌
            payload = claims.to_dict()
            token = jwt.encode(
                payload,
                signing_key,
                algorithm=config.algorithm.value
            )

            # 缓存令牌信息
            self.token_cache[claims.jti] = (token, claims, datetime.now())

            # 更新统计
            self.stats['tokens_generated'] += 1

            logger.debug(f"生成JWT令牌: {claims.jti} ({config.token_type.value})")
            return token

        except Exception as e:
            logger.error(f"生成JWT令牌失败: {e}")
            raise

    def validate_token(self, token: str, config: TokenConfig = None,
                      user_fingerprint: str = None,
                      ip_address: str = None) -> TokenValidationResult:
        """验证JWT令牌"""
        config = config or self.default_config

        try:
            # 获取验证密钥
            verification_key = self.key_manager.get_verification_key(config.algorithm)
            if not verification_key:
                return TokenValidationResult(
                    is_valid=False,
                    error_message=f"未找到算法 {config.algorithm.value} 的验证密钥",
                    error_code="KEY_NOT_FOUND"
                )

            # 解码令牌
            payload = jwt.decode(
                token,
                verification_key,
                algorithms=[config.algorithm.value],
                issuer=config.issuer,
                audience=config.audience,
                options={
                    'verify_exp': True,
                    'verify_nbf': True,
                    'verify_iat': True,
                    'require_exp': True,
                    'require_iat': True
                }
            )

            # 解析声明
            claims = TokenClaims.from_dict(payload)

            # 检查令牌是否在黑名单中
            if self.blacklist.is_token_blacklisted(claims.jti):
                return TokenValidationResult(
                    is_valid=False,
                    claims=claims,
                    error_message="令牌已被吊销",
                    error_code="TOKEN_REVOKED"
                )

            # 检查用户是否在黑名单中
            if claims.user_id and claims.iat:
                issued_at = datetime.fromtimestamp(claims.iat)
                if self.blacklist.is_user_blacklisted(claims.user_id, issued_at):
                    return TokenValidationResult(
                        is_valid=False,
                        claims=claims,
                        error_message="用户令牌已被吊销",
                        error_code="USER_REVOKED"
                    )

            # 验证用户指纹
            fingerprint_match = True
            if config.include_user_fingerprint and user_fingerprint:
                if claims.user_fingerprint != user_fingerprint:
                    fingerprint_match = False
                    return TokenValidationResult(
                        is_valid=False,
                        claims=claims,
                        error_message="用户指纹不匹配",
                        error_code="FINGERPRINT_MISMATCH",
                        fingerprint_match=False
                    )

            # 验证IP地址（可选）
            if ip_address and claims.ip_address:
                if claims.ip_address != ip_address:
                    logger.warning(f"IP地址不匹配: 令牌={claims.ip_address}, 当前={ip_address}")

            # 更新统计
            self.stats['tokens_validated'] += 1

            return TokenValidationResult(
                is_valid=True,
                claims=claims,
                fingerprint_match=fingerprint_match
            )

        except jwt.ExpiredSignatureError:
            self.stats['validation_errors'] += 1
            return TokenValidationResult(
                is_valid=False,
                error_message="令牌已过期",
                error_code="TOKEN_EXPIRED",
                is_expired=True
            )

        except jwt.InvalidTokenError as e:
            self.stats['validation_errors'] += 1
            return TokenValidationResult(
                is_valid=False,
                error_message=f"无效的令牌: {str(e)}",
                error_code="INVALID_TOKEN",
                signature_valid=False
            )

        except Exception as e:
            self.stats['validation_errors'] += 1
            logger.error(f"验证JWT令牌失败: {e}")
            return TokenValidationResult(
                is_valid=False,
                error_message=f"令牌验证错误: {str(e)}",
                error_code="VALIDATION_ERROR"
            )

    def refresh_token(self, refresh_token: str, config: TokenConfig = None) -> Optional[str]:
        """刷新访问令牌"""
        config = config or self.default_config

        # 验证刷新令牌
        validation_result = self.validate_token(refresh_token, config)

        if not validation_result.is_valid:
            logger.warning(f"刷新令牌无效: {validation_result.error_message}")
            return None

        claims = validation_result.claims

        # 检查令牌类型
        if claims.token_type != TokenType.REFRESH:
            logger.warning("尝试使用非刷新令牌进行刷新")
            return None

        # 创建新的访问令牌声明
        new_claims = TokenClaims(
            sub=claims.sub,
            iss=claims.iss,
            aud=claims.aud,
            token_type=TokenType.ACCESS,
            user_id=claims.user_id,
            username=claims.username,
            email=claims.email,
            roles=claims.roles,
            permissions=claims.permissions,
            user_fingerprint=claims.user_fingerprint,
            ip_address=claims.ip_address,
            user_agent_hash=claims.user_agent_hash,
            session_id=claims.session_id,
            metadata=claims.metadata
        )

        # 生成新的访问令牌
        access_config = TokenConfig(
            token_type=TokenType.ACCESS,
            algorithm=config.algorithm,
            expires_in_seconds=config.expires_in_seconds,
            issuer=config.issuer,
            audience=config.audience
        )

        new_access_token = self.generate_token(new_claims, access_config)

        logger.debug(f"刷新访问令牌: {claims.user_id}")
        return new_access_token

    def revoke_token(self, token: str) -> bool:
        """吊销令牌"""
        try:
            # 解码获取JTI
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get('jti')
            exp = payload.get('exp')

            if jti:
                expiration_time = datetime.fromtimestamp(exp) if exp else None
                self.blacklist.blacklist_token(jti, expiration_time)
                self.stats['blacklisted_tokens'] += 1
                return True

            return False

        except Exception as e:
            logger.error(f"吊销令牌失败: {e}")
            return False

    def revoke_user_tokens(self, user_id: str):
        """吊销用户所有令牌"""
        self.blacklist.blacklist_user(user_id)
        logger.info(f"吊销用户所有令牌: {user_id}")

    def generate_token_pair(self, claims: TokenClaims, config: TokenConfig = None) -> Dict[str, str]:
        """生成令牌对（访问令牌和刷新令牌）"""
        config = config or self.default_config

        # 生成访问令牌
        access_claims = claims.copy()
        access_claims.token_type = TokenType.ACCESS
        access_claims.jti = None  # 重新生成

        access_config = TokenConfig(
            token_type=TokenType.ACCESS,
            algorithm=config.algorithm,
            expires_in_seconds=config.expires_in_seconds,
            issuer=config.issuer,
            audience=config.audience
        )

        access_token = self.generate_token(access_claims, access_config)

        # 生成刷新令牌
        refresh_claims = claims.copy()
        refresh_claims.token_type = TokenType.REFRESH
        refresh_claims.jti = None  # 重新生成

        refresh_config = TokenConfig(
            token_type=TokenType.REFRESH,
            algorithm=config.algorithm,
            expires_in_seconds=config.refresh_expires_in_seconds,
            issuer=config.issuer,
            audience=config.audience
        )

        refresh_token = self.generate_token(refresh_claims, refresh_config)

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': config.expires_in_seconds
        }

    def generate_user_fingerprint(self, ip_address: str, user_agent: str) -> str:
        """生成用户指纹"""
        import hashlib

        fingerprint_data = f"{ip_address}:{user_agent}"
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:32]

    def decode_token_unsafe(self, token: str) -> Optional[Dict[str, Any]]:
        """不验证签名解码令牌（仅用于调试）"""
        try:
            return jwt.decode(token, options={"verify_signature": False})
        except Exception as e:
            logger.error(f"解码令牌失败: {e}")
            return None

    def get_token_info(self, token: str) -> Dict[str, Any]:
        """获取令牌信息"""
        payload = self.decode_token_unsafe(token)
        if not payload:
            return {'error': 'Invalid token'}

        return {
            'jti': payload.get('jti'),
            'user_id': payload.get('user_id'),
            'username': payload.get('username'),
            'token_type': payload.get('token_type'),
            'issued_at': datetime.fromtimestamp(payload['iat']).isoformat() if payload.get('iat') else None,
            'expires_at': datetime.fromtimestamp(payload['exp']).isoformat() if payload.get('exp') else None,
            'roles': payload.get('roles', []),
            'permissions': payload.get('permissions', [])
        }

    def cleanup_expired_cache(self):
        """清理过期的令牌缓存"""
        current_time = datetime.now()
        expired_keys = []

        for jti, (token, claims, created_at) in self.token_cache.items():
            if claims.exp and current_time.timestamp() > claims.exp:
                expired_keys.append(jti)

        for key in expired_keys:
            del self.token_cache[key]

        if expired_keys:
            logger.debug(f"清理 {len(expired_keys)} 个过期令牌缓存")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stats': self.stats,
            'cache_size': len(self.token_cache),
            'blacklisted_tokens': len(self.blacklist.blacklisted_tokens),
            'blacklisted_users': len(self.blacklist.blacklisted_users)
        }

# 全局实例
_jwt_handler_instance = None

def get_jwt_handler() -> JWTHandler:
    """获取JWT处理器实例"""
    global _jwt_handler_instance
    if _jwt_handler_instance is None:
        _jwt_handler_instance = JWTHandler()
    return _jwt_handler_instance
