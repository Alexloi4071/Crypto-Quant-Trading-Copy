"""
Authentication Manager
认证管理器，提供用户认证、会话管理和安全控制
支持多种认证方式和安全策略
"""

import asyncio
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import bcrypt
import hmac

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class AuthMethod(Enum):
    """认证方式枚举"""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH = "oauth"
    TWO_FACTOR = "two_factor"

class UserStatus(Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"

@dataclass

class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    password_hash: Optional[str] = None
    status: UserStatus = UserStatus.ACTIVE
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)

    # 安全设置
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    password_expires_at: Optional[datetime] = None

    # 审计字段
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    last_failed_login: Optional[datetime] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_sensitive: bool = False) -> dict:
        """转换为字典"""
        data = {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'status': self.status.value,
            'roles': list(self.roles),
            'permissions': list(self.permissions),
            'two_factor_enabled': self.two_factor_enabled,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'login_attempts': self.login_attempts,
            'metadata': self.metadata
        }

        if include_sensitive:
            data.update({
                'password_hash': self.password_hash,
                'two_factor_secret': self.two_factor_secret,
                'password_expires_at': self.password_expires_at.isoformat() if self.password_expires_at else None,
                'last_failed_login': self.last_failed_login.isoformat() if self.last_failed_login else None
            })

        return data

@dataclass

class AuthSession:
    """认证会话"""
    session_id: str
    user_id: str
    auth_method: AuthMethod
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.now)

    # 会话信息
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Dict[str, Any] = field(default_factory=dict)

    # 权限缓存
    cached_permissions: Set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """检查会话是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def is_inactive(self, max_inactive_minutes: int = 30) -> bool:
        """检查会话是否过期（基于非活跃时间）"""
        inactive_time = datetime.now() - self.last_activity
        return inactive_time.total_seconds() > (max_inactive_minutes * 60)

    def touch(self):
        """更新活动时间"""
        self.last_activity = datetime.now()

    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'auth_method': self.auth_method.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_activity': self.last_activity.isoformat(),
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'device_info': self.device_info,
            'cached_permissions': list(self.cached_permissions)
        }

class SecurityPolicy:
    """安全策略"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # 密码策略
        self.password_min_length = self.config.get('password_min_length', 8)
        self.password_require_uppercase = self.config.get('password_require_uppercase', True)
        self.password_require_lowercase = self.config.get('password_require_lowercase', True)
        self.password_require_numbers = self.config.get('password_require_numbers', True)
        self.password_require_special = self.config.get('password_require_special', True)
        self.password_expiry_days = self.config.get('password_expiry_days', 90)

        # 账户锁定策略
        self.max_login_attempts = self.config.get('max_login_attempts', 5)
        self.lockout_duration_minutes = self.config.get('lockout_duration_minutes', 30)

        # 会话策略
        self.session_timeout_hours = self.config.get('session_timeout_hours', 24)
        self.session_inactive_minutes = self.config.get('session_inactive_minutes', 30)
        self.max_concurrent_sessions = self.config.get('max_concurrent_sessions', 3)

        # 双因子认证
        self.require_2fa_for_admin = self.config.get('require_2fa_for_admin', True)
        self.require_2fa_for_api = self.config.get('require_2fa_for_api', False)

    def validate_password(self, password: str) -> tuple[bool, List[str]]:
        """验证密码强度"""
        errors = []

        if len(password) < self.password_min_length:
            errors.append(f"密码长度至少为 {self.password_min_length} 位")

        if self.password_require_uppercase and not any(c.isupper() for c in password):
            errors.append("密码必须包含大写字母")

        if self.password_require_lowercase and not any(c.islower() for c in password):
            errors.append("密码必须包含小写字母")

        if self.password_require_numbers and not any(c.isdigit() for c in password):
            errors.append("密码必须包含数字")

        if self.password_require_special:
            special_chars = "!@  # $%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("密码必须包含特殊字符")

        return len(errors) == 0, errors

    def should_require_2fa(self, user: User) -> bool:
        """判断是否需要双因子认证"""
        if self.require_2fa_for_admin and 'admin' in user.roles:
            return True

        if self.require_2fa_for_api and 'api_user' in user.roles:
            return True

        return False

class BruteForceProtection:
    """暴力破解保护"""

    def __init__(self):
        self.failed_attempts = defaultdict(list)  # IP -> [timestamp, ...]
        self.user_attempts = defaultdict(int)  # user_id -> count
        self.blocked_ips = {}  # IP -> block_until_timestamp
        self.blocked_users = {}  # user_id -> block_until_timestamp

        # 配置
        self.max_attempts_per_ip = 10  # IP级别限制
        self.max_attempts_per_user = 5  # 用户级别限制
        self.block_duration_minutes = 30
        self.attempt_window_minutes = 15  # 统计窗口

    def record_failed_attempt(self, ip_address: str, user_id: str = None):
        """记录失败尝试"""
        now = time.time()

        # 记录IP失败尝试
        self.failed_attempts[ip_address].append(now)

        # 清理过期记录
        cutoff_time = now - (self.attempt_window_minutes * 60)
        self.failed_attempts[ip_address] = [
            t for t in self.failed_attempts[ip_address] if t > cutoff_time
        ]

        # 检查是否需要阻止IP
        if len(self.failed_attempts[ip_address]) >= self.max_attempts_per_ip:
            block_until = now + (self.block_duration_minutes * 60)
            self.blocked_ips[ip_address] = block_until
            logger.warning(f"阻止IP地址: {ip_address}")

        # 记录用户失败尝试
        if user_id:
            self.user_attempts[user_id] += 1
            if self.user_attempts[user_id] >= self.max_attempts_per_user:
                block_until = now + (self.block_duration_minutes * 60)
                self.blocked_users[user_id] = block_until
                logger.warning(f"锁定用户: {user_id}")

    def is_blocked(self, ip_address: str, user_id: str = None) -> tuple[bool, str]:
        """检查是否被阻止"""
        now = time.time()

        # 清理过期的阻止记录
        self.blocked_ips = {ip: until for ip, until in self.blocked_ips.items() if until > now}
        self.blocked_users = {uid: until for uid, until in self.blocked_users.items() if until > now}

        # 检查IP是否被阻止
        if ip_address in self.blocked_ips:
            remaining = int((self.blocked_ips[ip_address] - now) / 60)
            return True, f"IP地址被阻止，剩余时间: {remaining} 分钟"

        # 检查用户是否被锁定
        if user_id and user_id in self.blocked_users:
            remaining = int((self.blocked_users[user_id] - now) / 60)
            return True, f"用户被锁定，剩余时间: {remaining} 分钟"

        return False, ""

    def reset_user_attempts(self, user_id: str):
        """重置用户尝试次数"""
        self.user_attempts[user_id] = 0
        if user_id in self.blocked_users:
            del self.blocked_users[user_id]

class PasswordManager:
    """密码管理器"""

    @staticmethod

    def hash_password(password: str) -> str:
        """哈希密码"""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')

    @staticmethod

    def verify_password(password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"密码验证失败: {e}")
            return False

    @staticmethod

    def generate_random_password(length: int = 12) -> str:
        """生成随机密码"""
        import string

        # 确保至少包含各种字符类型
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        special = "!@  # $%^&*"

        # 至少一个字符来自每个类别
        password_chars = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]

        # 填充剩余长度
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password_chars.append(secrets.choice(all_chars))

        # 随机打乱
        secrets.SystemRandom().shuffle(password_chars)

        return ''.join(password_chars)

class AuthenticationManager:
    """认证管理器主类"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.username_to_userid: Dict[str, str] = {}
        self.email_to_userid: Dict[str, str] = {}

        # 安全组件
        self.security_policy = SecurityPolicy(config.get('security_policy', {}))
        self.brute_force_protection = BruteForceProtection()
        self.password_manager = PasswordManager()

        # 监控和审计
        self.auth_events = deque(maxlen=10000)
        self.session_events = deque(maxlen=10000)

        # 统计信息
        self.stats = {
            'total_users': 0,
            'active_sessions': 0,
            'failed_logins': 0,
            'successful_logins': 0,
            'blocked_ips': 0,
            'locked_users': 0
        }

        # 清理任务
        self.cleanup_task = None
        self.is_running = False

        logger.info("认证管理器初始化完成")

    async def start(self):
        """启动认证管理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("认证管理器启动完成")

    async def stop(self):
        """停止认证管理器"""
        if not self.is_running:
            return

        logger.info("正在停止认证管理器...")
        self.is_running = False

        # 取消清理任务
        if self.cleanup_task:
            self.cleanup_task.cancel()

        logger.info("认证管理器已停止")

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(300)  # 5分钟清理一次
            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(300)

    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if (session.is_expired() or
                session.is_inactive(self.security_policy.session_inactive_minutes)):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self.logout_session(session_id)

        if expired_sessions:
            logger.debug(f"清理 {len(expired_sessions)} 个过期会话")

    # 用户管理

    def create_user(self, username: str, email: str, password: str,
                   roles: Set[str] = None) -> tuple[bool, str, Optional[User]]:
        """创建用户"""
        # 验证用户名和邮箱唯一性
        if username in self.username_to_userid:
            return False, "用户名已存在", None

        if email in self.email_to_userid:
            return False, "邮箱已被使用", None

        # 验证密码强度
        is_valid, errors = self.security_policy.validate_password(password)
        if not is_valid:
            return False, "; ".join(errors), None

        # 创建用户
        user_id = f"user_{int(time.time() * 1000000)}"
        password_hash = self.password_manager.hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or set(),
            password_expires_at=datetime.now() + timedelta(days=self.security_policy.password_expiry_days)
        )

        # 存储用户
        self.users[user_id] = user
        self.username_to_userid[username] = user_id
        self.email_to_userid[email] = user_id
        self.stats['total_users'] += 1

        # 记录事件
        self._record_auth_event("user_created", user_id, {"username": username, "email": email})

        logger.info(f"创建用户: {username} ({user_id})")
        return True, "用户创建成功", user

    def get_user(self, user_id: str = None, username: str = None,
                email: str = None) -> Optional[User]:
        """获取用户"""
        if user_id:
            return self.users.get(user_id)
        elif username:
            user_id = self.username_to_userid.get(username)
            return self.users.get(user_id) if user_id else None
        elif email:
            user_id = self.email_to_userid.get(email)
            return self.users.get(user_id) if user_id else None

        return None

    def update_user(self, user_id: str, **kwargs) -> bool:
        """更新用户信息"""
        if user_id not in self.users:
            return False

        user = self.users[user_id]

        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        self._record_auth_event("user_updated", user_id, kwargs)
        return True

    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        if user_id not in self.users:
            return False

        user = self.users[user_id]

        # 删除映射
        if user.username in self.username_to_userid:
            del self.username_to_userid[user.username]

        if user.email in self.email_to_userid:
            del self.email_to_userid[user.email]

        # 删除用户的所有会话
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        for session in user_sessions:
            del self.sessions[session.session_id]

        # 删除用户
        del self.users[user_id]
        self.stats['total_users'] -= 1

        self._record_auth_event("user_deleted", user_id, {"username": user.username})

        logger.info(f"删除用户: {user.username} ({user_id})")
        return True

    # 认证方法

    async def authenticate_password(self, username_or_email: str, password: str,
                                   ip_address: str = None, user_agent: str = None) -> tuple[bool, str, Optional[AuthSession]]:
        """密码认证"""
        # 检查暴力破解保护
        if ip_address:
            is_blocked, message = self.brute_force_protection.is_blocked(ip_address)
            if is_blocked:
                return False, message, None

        # 查找用户
        user = self.get_user(username=username_or_email)
        if not user:
            user = self.get_user(email=username_or_email)

        if not user:
            if ip_address:
                self.brute_force_protection.record_failed_attempt(ip_address)
            self.stats['failed_logins'] += 1
            return False, "用户不存在", None

        # 检查用户状态
        if user.status != UserStatus.ACTIVE:
            return False, f"用户状态: {user.status.value}", None

        # 检查暴力破解保护（用户级别）
        if ip_address:
            is_blocked, message = self.brute_force_protection.is_blocked(ip_address, user.user_id)
            if is_blocked:
                return False, message, None

        # 验证密码
        if not self.password_manager.verify_password(password, user.password_hash):
            user.login_attempts += 1
            user.last_failed_login = datetime.now()

            if ip_address:
                self.brute_force_protection.record_failed_attempt(ip_address, user.user_id)

            self.stats['failed_logins'] += 1
            self._record_auth_event("login_failed", user.user_id, {
                "reason": "invalid_password",
                "ip_address": ip_address
            })

            return False, "密码错误", None

        # 检查密码是否过期
        if user.password_expires_at and datetime.now() > user.password_expires_at:
            return False, "密码已过期，请重置密码", None

        # 检查是否需要双因子认证
        if self.security_policy.should_require_2fa(user) and not user.two_factor_enabled:
            return False, "需要启用双因子认证", None

        # 创建会话
        session = await self._create_session(user, AuthMethod.PASSWORD, ip_address, user_agent)

        # 重置失败尝试
        user.login_attempts = 0
        user.last_login = datetime.now()

        if ip_address:
            self.brute_force_protection.reset_user_attempts(user.user_id)

        self.stats['successful_logins'] += 1
        self._record_auth_event("login_success", user.user_id, {
            "session_id": session.session_id,
            "ip_address": ip_address
        })

        return True, "认证成功", session

    async def _create_session(self, user: User, auth_method: AuthMethod,
                            ip_address: str = None, user_agent: str = None) -> AuthSession:
        """创建认证会话"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=self.security_policy.session_timeout_hours)

        session = AuthSession(
            session_id=session_id,
            user_id=user.user_id,
            auth_method=auth_method,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
            cached_permissions=user.permissions.copy()
        )

        # 检查并发会话限制
        user_sessions = [s for s in self.sessions.values() if s.user_id == user.user_id]
        if len(user_sessions) >= self.security_policy.max_concurrent_sessions:
            # 移除最老的会话
            oldest_session = min(user_sessions, key=lambda s: s.created_at)
            await self.logout_session(oldest_session.session_id)

        self.sessions[session_id] = session
        self.stats['active_sessions'] += 1

        self._record_session_event("session_created", session_id, {
            "user_id": user.user_id,
            "auth_method": auth_method.value
        })

        return session

    async def logout_session(self, session_id: str) -> bool:
        """登出会话"""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        del self.sessions[session_id]
        self.stats['active_sessions'] -= 1

        self._record_session_event("session_destroyed", session_id, {
            "user_id": session.user_id
        })

        return True

    async def logout_user_all_sessions(self, user_id: str) -> int:
        """登出用户的所有会话"""
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]

        for session in user_sessions:
            await self.logout_session(session.session_id)

        return len(user_sessions)

    # 会话验证

    def validate_session(self, session_id: str) -> tuple[bool, Optional[AuthSession], Optional[User]]:
        """验证会话"""
        if session_id not in self.sessions:
            return False, None, None

        session = self.sessions[session_id]

        # 检查过期
        if session.is_expired():
            asyncio.create_task(self.logout_session(session_id))
            return False, None, None

        # 检查非活跃超时
        if session.is_inactive(self.security_policy.session_inactive_minutes):
            asyncio.create_task(self.logout_session(session_id))
            return False, None, None

        # 获取用户信息
        user = self.get_user(session.user_id)
        if not user or user.status != UserStatus.ACTIVE:
            asyncio.create_task(self.logout_session(session_id))
            return False, None, None

        # 更新活动时间
        session.touch()

        return True, session, user

    # 辅助方法

    def _record_auth_event(self, event_type: str, user_id: str, metadata: dict):
        """记录认证事件"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'metadata': metadata
        }
        self.auth_events.append(event)

    def _record_session_event(self, event_type: str, session_id: str, metadata: dict):
        """记录会话事件"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'session_id': session_id,
            'metadata': metadata
        }
        self.session_events.append(event)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'is_running': self.is_running,
            'stats': self.stats,
            'brute_force_stats': {
                'blocked_ips': len(self.brute_force_protection.blocked_ips),
                'blocked_users': len(self.brute_force_protection.blocked_users)
            },
            'recent_auth_events': list(self.auth_events)[-10:],
            'recent_session_events': list(self.session_events)[-10:]
        }

# 全局实例
_auth_manager_instance = None

def get_auth_manager() -> AuthenticationManager:
    """获取认证管理器实例"""
    global _auth_manager_instance
    if _auth_manager_instance is None:
        _auth_manager_instance = AuthenticationManager()
    return _auth_manager_instance
