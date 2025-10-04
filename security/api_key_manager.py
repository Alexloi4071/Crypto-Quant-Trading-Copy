"""
API Key Manager
API密钥管理器，提供API密钥的生成、验证和管理
支持多种密钥类型和安全控制
"""

import secrets
import time
import hashlib
import hmac
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
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

class APIKeyType(Enum):
    """API密钥类型"""
    READ_ONLY = "read_only"
    TRADE = "trade"
    FULL_ACCESS = "full_access"
    WEBHOOK = "webhook"
    TEMPORARY = "temporary"

class APIKeyStatus(Enum):
    """API密钥状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

@dataclass

class APIKeyPermission:
    """API密钥权限"""
    resource: str
    actions: Set[str] = field(default_factory=set)
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'resource': self.resource,
            'actions': list(self.actions),
            'conditions': self.conditions
        }

@dataclass

class APIKey:
    """API密钥"""
    key_id: str
    user_id: str
    name: str
    key_hash: str  # 存储哈希值而不是原始密钥
    secret_hash: Optional[str] = None  # HMAC密钥的哈希

    # 密钥属性
    key_type: APIKeyType = APIKeyType.READ_ONLY
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    permissions: List[APIKeyPermission] = field(default_factory=list)

    # 时间限制
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None

    # 使用限制
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 3600
    daily_request_limit: Optional[int] = None

    # IP白名单
    ip_whitelist: Set[str] = field(default_factory=set)

    # 统计信息
    total_requests: int = 0
    last_request_ip: Optional[str] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """检查密钥是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_active(self) -> bool:
        """检查密钥是否可用"""
        return (self.status == APIKeyStatus.ACTIVE and
                not self.is_expired())

    def can_access_ip(self, ip_address: str) -> bool:
        """检查IP是否被允许"""
        if not self.ip_whitelist:
            return True  # 没有IP限制
        return ip_address in self.ip_whitelist

    def touch(self, ip_address: str = None):
        """更新使用时间"""
        self.last_used_at = datetime.now()
        self.total_requests += 1
        if ip_address:
            self.last_request_ip = ip_address

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """转换为字典"""
        data = {
            'key_id': self.key_id,
            'user_id': self.user_id,
            'name': self.name,
            'key_type': self.key_type.value,
            'status': self.status.value,
            'permissions': [p.to_dict() for p in self.permissions],
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'rate_limit_per_hour': self.rate_limit_per_hour,
            'daily_request_limit': self.daily_request_limit,
            'ip_whitelist': list(self.ip_whitelist),
            'total_requests': self.total_requests,
            'last_request_ip': self.last_request_ip,
            'metadata': self.metadata
        }

        if include_sensitive:
            data.update({
                'key_hash': self.key_hash,
                'secret_hash': self.secret_hash
            })

        return data

@dataclass

class APIKeyUsage:
    """API密钥使用记录"""
    key_id: str
    timestamp: datetime
    ip_address: str
    endpoint: str
    method: str
    user_agent: Optional[str] = None
    response_status: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'key_id': self.key_id,
            'timestamp': self.timestamp.isoformat(),
            'ip_address': self.ip_address,
            'endpoint': self.endpoint,
            'method': self.method,
            'user_agent': self.user_agent,
            'response_status': self.response_status
        }

class RateLimiter:
    """API密钥速率限制器"""

    def __init__(self):
        self.minute_requests = defaultdict(deque)  # key_id -> deque of timestamps
        self.hour_requests = defaultdict(deque)
        self.daily_requests = defaultdict(int)  # key_id -> count
        self.daily_reset_time = defaultdict(datetime)  # key_id -> reset_time

    def is_rate_limited(self, api_key: APIKey, current_time: datetime = None) -> tuple[bool, str]:
        """检查是否达到速率限制"""
        if current_time is None:
            current_time = datetime.now()

        current_timestamp = current_time.timestamp()
        key_id = api_key.key_id

        # 检查每分钟限制
        minute_window = current_timestamp - 60
        minute_queue = self.minute_requests[key_id]

        # 清理过期记录
        while minute_queue and minute_queue[0] < minute_window:
            minute_queue.popleft()

        if len(minute_queue) >= api_key.rate_limit_per_minute:
            return True, f"超过每分钟请求限制 ({api_key.rate_limit_per_minute})"

        # 检查每小时限制
        hour_window = current_timestamp - 3600
        hour_queue = self.hour_requests[key_id]

        while hour_queue and hour_queue[0] < hour_window:
            hour_queue.popleft()

        if len(hour_queue) >= api_key.rate_limit_per_hour:
            return True, f"超过每小时请求限制 ({api_key.rate_limit_per_hour})"

        # 检查每日限制
        if api_key.daily_request_limit:
            # 检查是否需要重置每日计数
            reset_time = self.daily_reset_time.get(key_id, current_time.replace(hour=0, minute=0, second=0, microsecond=0))
            if current_time >= reset_time + timedelta(days=1):
                self.daily_requests[key_id] = 0
                self.daily_reset_time[key_id] = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

            if self.daily_requests[key_id] >= api_key.daily_request_limit:
                return True, f"超过每日请求限制 ({api_key.daily_request_limit})"

        return False, ""

    def record_request(self, key_id: str, current_time: datetime = None):
        """记录请求"""
        if current_time is None:
            current_time = datetime.now()

        current_timestamp = current_time.timestamp()

        self.minute_requests[key_id].append(current_timestamp)
        self.hour_requests[key_id].append(current_timestamp)
        self.daily_requests[key_id] += 1

    def get_usage_stats(self, key_id: str, current_time: datetime = None) -> dict:
        """获取使用统计"""
        if current_time is None:
            current_time = datetime.now()

        current_timestamp = current_time.timestamp()

        # 清理过期记录
        minute_window = current_timestamp - 60
        hour_window = current_timestamp - 3600

        minute_queue = self.minute_requests[key_id]
        hour_queue = self.hour_requests[key_id]

        while minute_queue and minute_queue[0] < minute_window:
            minute_queue.popleft()

        while hour_queue and hour_queue[0] < hour_window:
            hour_queue.popleft()

        return {
            'requests_last_minute': len(minute_queue),
            'requests_last_hour': len(hour_queue),
            'requests_today': self.daily_requests[key_id],
            'daily_reset_time': self.daily_reset_time.get(key_id, current_time).isoformat()
        }

class APIKeyValidator:
    """API密钥验证器"""

    def __init__(self):
        pass

    def generate_key_pair(self) -> tuple[str, str]:
        """生成API密钥对（公钥和私钥）"""
        # 公钥：用于标识
        public_key = f"pk_{''.join(secrets.choice('0123456789abcdef') for _ in range(32))}"

        # 私钥：用于签名
        secret_key = f"sk_{''.join(secrets.choice('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(64))}"

        return public_key, secret_key

    def hash_key(self, key: str) -> str:
        """哈希密钥"""
        return hashlib.sha256(key.encode()).hexdigest()

    def verify_key(self, key: str, key_hash: str) -> bool:
        """验证密钥"""
        return self.hash_key(key) == key_hash

    def generate_hmac_signature(self, message: str, secret: str) -> str:
        """生成HMAC签名"""
        return hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def verify_hmac_signature(self, message: str, signature: str, secret: str) -> bool:
        """验证HMAC签名"""
        expected_signature = self.generate_hmac_signature(message, secret)
        return hmac.compare_digest(signature, expected_signature)

class APIKeyManager:
    """API密钥管理器主类"""

    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self.key_hash_to_id: Dict[str, str] = {}  # key_hash -> key_id
        self.user_keys: Dict[str, Set[str]] = defaultdict(set)  # user_id -> key_ids

        # 组件
        self.validator = APIKeyValidator()
        self.rate_limiter = RateLimiter()

        # 使用记录
        self.usage_history = deque(maxlen=100000)  # 最近10万条记录

        # 统计信息
        self.stats = {
            'total_keys': 0,
            'active_keys': 0,
            'requests_today': 0,
            'blocked_requests': 0,
            'key_generations': 0,
            'key_revocations': 0
        }

        # 预定义权限模板
        self._initialize_permission_templates()

        logger.info("API密钥管理器初始化完成")

    def _initialize_permission_templates(self):
        """初始化权限模板"""
        self.permission_templates = {
            APIKeyType.READ_ONLY: [
                APIKeyPermission("market_data", {"read"}),
                APIKeyPermission("portfolio", {"read"}),
                APIKeyPermission("orders", {"read"}),
            ],
            APIKeyType.TRADE: [
                APIKeyPermission("market_data", {"read"}),
                APIKeyPermission("portfolio", {"read"}),
                APIKeyPermission("orders", {"read", "create", "update", "cancel"}),
                APIKeyPermission("trading", {"execute"}),
            ],
            APIKeyType.FULL_ACCESS: [
                APIKeyPermission("*", {"*"}),  # 所有权限
            ],
            APIKeyType.WEBHOOK: [
                APIKeyPermission("webhooks", {"receive"}),
            ]
        }

    def create_api_key(self, user_id: str, name: str, key_type: APIKeyType = APIKeyType.READ_ONLY,
                      expires_in_days: Optional[int] = None,
                      ip_whitelist: Set[str] = None,
                      custom_permissions: List[APIKeyPermission] = None) -> tuple[bool, str, Optional[Dict[str, str]]]:
        """创建API密钥"""
        try:
            # 生成密钥对
            public_key, secret_key = self.validator.generate_key_pair()

            # 创建密钥ID
            key_id = f"key_{int(time.time() * 1000000)}"

            # 计算过期时间
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now() + timedelta(days=expires_in_days)

            # 获取权限
            permissions = custom_permissions or self.permission_templates.get(key_type, [])

            # 创建API密钥对象
            api_key = APIKey(
                key_id=key_id,
                user_id=user_id,
                name=name,
                key_hash=self.validator.hash_key(public_key),
                secret_hash=self.validator.hash_key(secret_key),
                key_type=key_type,
                permissions=permissions.copy(),
                expires_at=expires_at,
                ip_whitelist=ip_whitelist or set()
            )

            # 存储密钥
            self.api_keys[key_id] = api_key
            self.key_hash_to_id[api_key.key_hash] = key_id
            self.user_keys[user_id].add(key_id)

            # 更新统计
            self.stats['total_keys'] += 1
            self.stats['active_keys'] += 1
            self.stats['key_generations'] += 1

            logger.info(f"创建API密钥: {name} ({key_id}) for user {user_id}")

            return True, "API密钥创建成功", {
                'key_id': key_id,
                'public_key': public_key,
                'secret_key': secret_key,
                'key_type': key_type.value,
                'expires_at': expires_at.isoformat() if expires_at else None
            }

        except Exception as e:
            logger.error(f"创建API密钥失败: {e}")
            return False, f"创建失败: {str(e)}", None

    def validate_api_key(self, public_key: str, ip_address: str = None,
                        endpoint: str = None, method: str = None) -> tuple[bool, str, Optional[APIKey]]:
        """验证API密钥"""
        try:
            # 查找密钥
            key_hash = self.validator.hash_key(public_key)
            key_id = self.key_hash_to_id.get(key_hash)

            if not key_id:
                self.stats['blocked_requests'] += 1
                return False, "无效的API密钥", None

            api_key = self.api_keys.get(key_id)
            if not api_key:
                self.stats['blocked_requests'] += 1
                return False, "API密钥不存在", None

            # 检查密钥状态
            if not api_key.is_active():
                self.stats['blocked_requests'] += 1
                return False, f"API密钥状态: {api_key.status.value}", None

            # 检查IP白名单
            if ip_address and not api_key.can_access_ip(ip_address):
                self.stats['blocked_requests'] += 1
                return False, f"IP地址未授权: {ip_address}", None

            # 检查速率限制
            is_limited, limit_message = self.rate_limiter.is_rate_limited(api_key)
            if is_limited:
                self.stats['blocked_requests'] += 1
                return False, f"请求频率限制: {limit_message}", None

            # 记录使用
            self.rate_limiter.record_request(key_id)
            api_key.touch(ip_address)

            # 记录使用历史
            if endpoint and method:
                usage = APIKeyUsage(
                    key_id=key_id,
                    timestamp=datetime.now(),
                    ip_address=ip_address or "unknown",
                    endpoint=endpoint,
                    method=method
                )
                self.usage_history.append(usage)

            self.stats['requests_today'] += 1

            return True, "验证成功", api_key

        except Exception as e:
            logger.error(f"验证API密钥失败: {e}")
            self.stats['blocked_requests'] += 1
            return False, f"验证错误: {str(e)}", None

    def validate_hmac_request(self, public_key: str, timestamp: str, signature: str,
                            request_body: str = "", ip_address: str = None) -> tuple[bool, str, Optional[APIKey]]:
        """验证HMAC签名请求"""
        try:
            # 基础密钥验证
            is_valid, message, api_key = self.validate_api_key(public_key, ip_address)
            if not is_valid:
                return False, message, None

            # 时间戳验证（防重放攻击）
            try:
                request_timestamp = int(timestamp)
                current_timestamp = int(time.time())

                # 允许5分钟的时间偏差
                if abs(current_timestamp - request_timestamp) > 300:
                    return False, "请求时间戳过期", None

            except ValueError:
                return False, "无效的时间戳格式", None

            # 构造签名消息
            message_to_sign = f"{timestamp}{request_body}"

            # 获取密钥的secret
            # 注意：这里需要从安全存储中获取原始secret，而不是哈希
            # 实际实现中应该有安全的方式来获取原始secret
            secret_key = self._get_secret_key(api_key.key_id)  # 需要实现
            if not secret_key:
                return False, "无法获取密钥secret", None

            # 验证签名
            is_signature_valid = self.validator.verify_hmac_signature(
                message_to_sign, signature, secret_key
            )

            if not is_signature_valid:
                self.stats['blocked_requests'] += 1
                return False, "HMAC签名验证失败", None

            return True, "HMAC验证成功", api_key

        except Exception as e:
            logger.error(f"HMAC验证失败: {e}")
            self.stats['blocked_requests'] += 1
            return False, f"HMAC验证错误: {str(e)}", None

    def _get_secret_key(self, key_id: str) -> Optional[str]:
        """获取密钥的secret（需要安全实现）"""
        # 这里是简化实现，实际应该从安全存储中获取
        # 例如从加密数据库、密钥管理服务等
        logger.warning("_get_secret_key需要安全实现")
        return None

    def revoke_api_key(self, key_id: str, user_id: str = None) -> bool:
        """撤销API密钥"""
        if key_id not in self.api_keys:
            return False

        api_key = self.api_keys[key_id]

        # 检查用户权限
        if user_id and api_key.user_id != user_id:
            logger.warning(f"用户 {user_id} 尝试撤销不属于自己的密钥 {key_id}")
            return False

        # 更新状态
        api_key.status = APIKeyStatus.REVOKED

        # 更新统计
        self.stats['active_keys'] -= 1
        self.stats['key_revocations'] += 1

        logger.info(f"撤销API密钥: {key_id}")
        return True

    def delete_api_key(self, key_id: str, user_id: str = None) -> bool:
        """删除API密钥"""
        if key_id not in self.api_keys:
            return False

        api_key = self.api_keys[key_id]

        # 检查用户权限
        if user_id and api_key.user_id != user_id:
            logger.warning(f"用户 {user_id} 尝试删除不属于自己的密钥 {key_id}")
            return False

        # 从索引中移除
        self.key_hash_to_id.pop(api_key.key_hash, None)
        self.user_keys[api_key.user_id].discard(key_id)

        # 删除密钥
        del self.api_keys[key_id]

        # 更新统计
        self.stats['total_keys'] -= 1
        if api_key.status == APIKeyStatus.ACTIVE:
            self.stats['active_keys'] -= 1

        logger.info(f"删除API密钥: {key_id}")
        return True

    def update_api_key(self, key_id: str, **kwargs) -> bool:
        """更新API密钥属性"""
        if key_id not in self.api_keys:
            return False

        api_key = self.api_keys[key_id]

        # 允许更新的字段
        updatable_fields = {
            'name', 'status', 'expires_at', 'rate_limit_per_minute',
            'rate_limit_per_hour', 'daily_request_limit', 'ip_whitelist', 'metadata'
        }

        for field, value in kwargs.items():
            if field in updatable_fields and hasattr(api_key, field):
                setattr(api_key, field, value)

        logger.info(f"更新API密钥: {key_id}")
        return True

    def get_user_api_keys(self, user_id: str) -> List[APIKey]:
        """获取用户的API密钥列表"""
        key_ids = self.user_keys.get(user_id, set())
        return [self.api_keys[kid] for kid in key_ids if kid in self.api_keys]

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """获取API密钥"""
        return self.api_keys.get(key_id)

    def get_api_key_usage_stats(self, key_id: str) -> Dict[str, Any]:
        """获取API密钥使用统计"""
        if key_id not in self.api_keys:
            return {}

        api_key = self.api_keys[key_id]
        rate_stats = self.rate_limiter.get_usage_stats(key_id)

        # 获取使用历史
        usage_records = [u for u in self.usage_history if u.key_id == key_id]
        recent_usage = sorted(usage_records, key=lambda x: x.timestamp, reverse=True)[:100]

        return {
            'key_info': api_key.to_dict(),
            'rate_limiting': rate_stats,
            'recent_usage': [u.to_dict() for u in recent_usage],
            'usage_summary': {
                'total_requests': api_key.total_requests,
                'last_used': api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                'last_ip': api_key.last_request_ip
            }
        }

    def check_permission(self, api_key: APIKey, resource: str, action: str) -> bool:
        """检查API密钥权限"""
        for permission in api_key.permissions:
            # 通配符权限
            if permission.resource == "*" and "*" in permission.actions:
                return True

            # 精确匹配
            if permission.resource == resource:
                if "*" in permission.actions or action in permission.actions:
                    return True

        return False

    def cleanup_expired_keys(self):
        """清理过期密钥"""
        expired_keys = []

        for key_id, api_key in self.api_keys.items():
            if api_key.is_expired():
                api_key.status = APIKeyStatus.EXPIRED
                if api_key.status == APIKeyStatus.ACTIVE:
                    self.stats['active_keys'] -= 1
                expired_keys.append(key_id)

        if expired_keys:
            logger.info(f"标记 {len(expired_keys)} 个密钥为过期状态")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 更新活跃密钥数量
        active_count = sum(1 for key in self.api_keys.values() if key.is_active())
        self.stats['active_keys'] = active_count

        return {
            'stats': self.stats,
            'key_types_distribution': {
                key_type.value: sum(1 for key in self.api_keys.values()
                                  if key.key_type == key_type)
                for key_type in APIKeyType
            },
            'status_distribution': {
                status.value: sum(1 for key in self.api_keys.values()
                                if key.status == status)
                for status in APIKeyStatus
            },
            'usage_history_size': len(self.usage_history)
        }

# 全局实例
_api_key_manager_instance = None

def get_api_key_manager() -> APIKeyManager:
    """获取API密钥管理器实例"""
    global _api_key_manager_instance
    if _api_key_manager_instance is None:
        _api_key_manager_instance = APIKeyManager()
    return _api_key_manager_instance
