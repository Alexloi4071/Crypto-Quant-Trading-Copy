"""
Permission Manager
权限管理器，提供细粒度的权限控制和访问控制
支持基于角色的访问控制(RBAC)和基于属性的访问控制(ABAC)
"""

import time
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict
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

class PermissionType(Enum):
    """权限类型"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    CREATE = "create"
    UPDATE = "update"

class ResourceType(Enum):
    """资源类型"""
    USER = "user"
    ORDER = "order"
    PORTFOLIO = "portfolio"
    STRATEGY = "strategy"
    MARKET_DATA = "market_data"
    TRADING = "trading"
    SYSTEM = "system"
    API = "api"
    REPORT = "report"

class AccessLevel(Enum):
    """访问级别"""
    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3

@dataclass

class Permission:
    """权限定义"""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    permission_type: PermissionType
    created_at: datetime = field(default_factory=datetime.now)

    # 权限约束
    conditions: Dict[str, Any] = field(default_factory=dict)
    time_restrictions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'permission_id': self.permission_id,
            'name': self.name,
            'description': self.description,
            'resource_type': self.resource_type.value,
            'permission_type': self.permission_type.value,
            'created_at': self.created_at.isoformat(),
            'conditions': self.conditions,
            'time_restrictions': self.time_restrictions
        }

@dataclass

class Role:
    """角色定义"""
    role_id: str
    name: str
    description: str
    permissions: Set[str] = field(default_factory=set)  # permission_ids
    parent_roles: Set[str] = field(default_factory=set)  # 继承的角色
    created_at: datetime = field(default_factory=datetime.now)

    # 角色属性
    is_system_role: bool = False
    max_users: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'role_id': self.role_id,
            'name': self.name,
            'description': self.description,
            'permissions': list(self.permissions),
            'parent_roles': list(self.parent_roles),
            'created_at': self.created_at.isoformat(),
            'is_system_role': self.is_system_role,
            'max_users': self.max_users
        }

@dataclass

class AccessContext:
    """访问上下文"""
    user_id: str
    resource_type: ResourceType
    resource_id: Optional[str] = None
    action: Optional[str] = None

    # 上下文信息
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None

    # 额外属性
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'user_id': self.user_id,
            'resource_type': self.resource_type.value,
            'resource_id': self.resource_id,
            'action': self.action,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'attributes': self.attributes
        }

@dataclass

class AccessDecision:
    """访问决策结果"""
    allowed: bool
    user_id: str
    resource_type: ResourceType
    action: str

    # 决策信息
    matched_permissions: List[str] = field(default_factory=list)
    denied_reason: Optional[str] = None
    conditions_met: bool = True
    time_valid: bool = True

    # 审计信息
    decision_time: datetime = field(default_factory=datetime.now)
    evaluator: str = "permission_manager"

    def to_dict(self) -> dict:
        return {
            'allowed': self.allowed,
            'user_id': self.user_id,
            'resource_type': self.resource_type.value,
            'action': self.action,
            'matched_permissions': self.matched_permissions,
            'denied_reason': self.denied_reason,
            'conditions_met': self.conditions_met,
            'time_valid': self.time_valid,
            'decision_time': self.decision_time.isoformat(),
            'evaluator': self.evaluator
        }

class PermissionCache:
    """权限缓存"""

    def __init__(self, ttl_seconds: int = 300):  # 5分钟默认TTL
        self.cache = {}  # (user_id, resource_type, action) -> (decision, expire_time)
        self.ttl_seconds = ttl_seconds

    def get(self, user_id: str, resource_type: ResourceType, action: str) -> Optional[AccessDecision]:
        """获取缓存的决策"""
        key = (user_id, resource_type.value, action)

        if key in self.cache:
            decision, expire_time = self.cache[key]

            if time.time() < expire_time:
                return decision
            else:
                # 过期，删除缓存
                del self.cache[key]

        return None

    def put(self, user_id: str, resource_type: ResourceType, action: str, decision: AccessDecision):
        """缓存决策"""
        key = (user_id, resource_type.value, action)
        expire_time = time.time() + self.ttl_seconds

        self.cache[key] = (decision, expire_time)

    def invalidate_user(self, user_id: str):
        """使用户的所有缓存失效"""
        keys_to_remove = [key for key in self.cache.keys() if key[0] == user_id]

        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            logger.debug(f"清除用户权限缓存: {user_id}, {len(keys_to_remove)} 条记录")

    def clear(self):
        """清空所有缓存"""
        self.cache.clear()

    def cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for key, (decision, expire_time) in self.cache.items():
            if current_time >= expire_time:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.debug(f"清理 {len(expired_keys)} 条过期权限缓存")

class ConditionEvaluator:
    """条件评估器"""

    def __init__(self):
        self.evaluators = {
            'time_range': self._evaluate_time_range,
            'ip_whitelist': self._evaluate_ip_whitelist,
            'user_attribute': self._evaluate_user_attribute,
            'resource_owner': self._evaluate_resource_owner,
            'trading_hours': self._evaluate_trading_hours
        }

    def evaluate(self, conditions: Dict[str, Any], context: AccessContext) -> tuple[bool, str]:
        """评估所有条件"""
        if not conditions:
            return True, ""

        for condition_type, condition_value in conditions.items():
            if condition_type in self.evaluators:
                is_met, reason = self.evaluators[condition_type](condition_value, context)
                if not is_met:
                    return False, reason
            else:
                logger.warning(f"未知的条件类型: {condition_type}")

        return True, ""

    def _evaluate_time_range(self, condition: Dict[str, Any], context: AccessContext) -> tuple[bool, str]:
        """评估时间范围条件"""
        start_time = condition.get('start_time')
        end_time = condition.get('end_time')
        current_time = context.timestamp.time()

        if start_time:
            start = datetime.strptime(start_time, '%H:%M').time()
            if current_time < start:
                return False, f"不在允许的时间范围内 (开始: {start_time})"

        if end_time:
            end = datetime.strptime(end_time, '%H:%M').time()
            if current_time > end:
                return False, f"不在允许的时间范围内 (结束: {end_time})"

        return True, ""

    def _evaluate_ip_whitelist(self, condition: List[str], context: AccessContext) -> tuple[bool, str]:
        """评估IP白名单条件"""
        if not context.ip_address:
            return False, "缺少IP地址信息"

        if context.ip_address not in condition:
            return False, f"IP地址不在白名单中: {context.ip_address}"

        return True, ""

    def _evaluate_user_attribute(self, condition: Dict[str, Any], context: AccessContext) -> tuple[bool, str]:
        """评估用户属性条件"""
        required_attr = condition.get('attribute')
        required_value = condition.get('value')

        if required_attr not in context.attributes:
            return False, f"缺少必需的用户属性: {required_attr}"

        if context.attributes[required_attr] != required_value:
            return False, f"用户属性不匹配: {required_attr}"

        return True, ""

    def _evaluate_resource_owner(self, condition: Dict[str, Any], context: AccessContext) -> tuple[bool, str]:
        """评估资源所有者条件"""
        # 简化实现，实际需要查询资源所有者信息
        owner_attribute = condition.get('owner_attribute', 'owner_id')

        if owner_attribute in context.attributes:
            if context.attributes[owner_attribute] != context.user_id:
                return False, "不是资源所有者"

        return True, ""

    def _evaluate_trading_hours(self, condition: Dict[str, Any], context: AccessContext) -> tuple[bool, str]:
        """评估交易时间条件"""
        # 简化实现，检查是否在交易时间内
        current_hour = context.timestamp.hour

        if condition.get('market') == 'US':
            # 美股交易时间 (简化)
            if not (9 <= current_hour <= 16):
                return False, "不在美股交易时间内"
        elif condition.get('market') == 'CRYPTO':
            # 加密货币24小时交易
            pass

        return True, ""

class PermissionManager:
    """权限管理器主类"""

    def __init__(self):
        # 权限和角色存储
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)  # user_id -> role_ids
        self.user_permissions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> permission_ids

        # 缓存和评估器
        self.permission_cache = PermissionCache()
        self.condition_evaluator = ConditionEvaluator()

        # 审计日志
        self.access_log = []  # 访问决策日志

        # 统计信息
        self.stats = {
            'total_permissions': 0,
            'total_roles': 0,
            'access_checks': 0,
            'access_granted': 0,
            'access_denied': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # 初始化系统权限
        self._initialize_system_permissions()
        self._initialize_system_roles()

        logger.info("权限管理器初始化完成")

    def _initialize_system_permissions(self):
        """初始化系统权限"""
        system_permissions = [
            # 用户管理
            ("user_read", "读取用户信息", ResourceType.USER, PermissionType.READ),
            ("user_write", "修改用户信息", ResourceType.USER, PermissionType.WRITE),
            ("user_create", "创建用户", ResourceType.USER, PermissionType.CREATE),
            ("user_delete", "删除用户", ResourceType.USER, PermissionType.DELETE),
            ("user_admin", "用户管理", ResourceType.USER, PermissionType.ADMIN),

            # 交易权限
            ("trading_read", "查看交易", ResourceType.TRADING, PermissionType.READ),
            ("trading_execute", "执行交易", ResourceType.TRADING, PermissionType.EXECUTE),
            ("trading_admin", "交易管理", ResourceType.TRADING, PermissionType.ADMIN),

            # 策略权限
            ("strategy_read", "查看策略", ResourceType.STRATEGY, PermissionType.READ),
            ("strategy_write", "修改策略", ResourceType.STRATEGY, PermissionType.WRITE),
            ("strategy_create", "创建策略", ResourceType.STRATEGY, PermissionType.CREATE),
            ("strategy_execute", "执行策略", ResourceType.STRATEGY, PermissionType.EXECUTE),

            # 市场数据
            ("market_data_read", "查看市场数据", ResourceType.MARKET_DATA, PermissionType.READ),

            # API权限
            ("api_read", "API读取", ResourceType.API, PermissionType.READ),
            ("api_write", "API写入", ResourceType.API, PermissionType.WRITE),

            # 系统管理
            ("system_admin", "系统管理", ResourceType.SYSTEM, PermissionType.ADMIN),
            ("system_read", "系统监控", ResourceType.SYSTEM, PermissionType.READ),
        ]

        for perm_id, name, resource_type, perm_type in system_permissions:
            permission = Permission(
                permission_id=perm_id,
                name=name,
                description=f"系统权限: {name}",
                resource_type=resource_type,
                permission_type=perm_type
            )
            self.permissions[perm_id] = permission
            self.stats['total_permissions'] += 1

    def _initialize_system_roles(self):
        """初始化系统角色"""
        # 管理员角色
        admin_role = Role(
            role_id="admin",
            name="系统管理员",
            description="系统管理员，拥有所有权限",
            permissions=set(self.permissions.keys()),
            is_system_role=True
        )
        self.roles["admin"] = admin_role

        # 交易员角色
        trader_permissions = {
            "trading_read", "trading_execute", "strategy_read", "strategy_write",
            "market_data_read", "api_read", "api_write"
        }
        trader_role = Role(
            role_id="trader",
            name="交易员",
            description="可以执行交易和管理策略",
            permissions=trader_permissions,
            is_system_role=True
        )
        self.roles["trader"] = trader_role

        # 分析师角色
        analyst_permissions = {
            "market_data_read", "strategy_read", "trading_read", "api_read"
        }
        analyst_role = Role(
            role_id="analyst",
            name="分析师",
            description="可以查看数据和分析",
            permissions=analyst_permissions,
            is_system_role=True
        )
        self.roles["analyst"] = analyst_role

        # 普通用户角色
        user_permissions = {
            "market_data_read", "api_read"
        }
        user_role = Role(
            role_id="user",
            name="普通用户",
            description="基本用户权限",
            permissions=user_permissions,
            is_system_role=True
        )
        self.roles["user"] = user_role

        self.stats['total_roles'] = len(self.roles)

    # 权限管理

    def create_permission(self, permission_id: str, name: str, description: str,
                         resource_type: ResourceType, permission_type: PermissionType,
                         conditions: Dict[str, Any] = None) -> bool:
        """创建权限"""
        if permission_id in self.permissions:
            logger.warning(f"权限已存在: {permission_id}")
            return False

        permission = Permission(
            permission_id=permission_id,
            name=name,
            description=description,
            resource_type=resource_type,
            permission_type=permission_type,
            conditions=conditions or {}
        )

        self.permissions[permission_id] = permission
        self.stats['total_permissions'] += 1

        logger.info(f"创建权限: {permission_id}")
        return True

    def delete_permission(self, permission_id: str) -> bool:
        """删除权限"""
        if permission_id not in self.permissions:
            return False

        # 从所有角色中移除此权限
        for role in self.roles.values():
            role.permissions.discard(permission_id)

        # 从所有用户中移除此权限
        for user_permissions in self.user_permissions.values():
            user_permissions.discard(permission_id)

        del self.permissions[permission_id]
        self.stats['total_permissions'] -= 1

        # 清空权限缓存
        self.permission_cache.clear()

        logger.info(f"删除权限: {permission_id}")
        return True

    # 角色管理

    def create_role(self, role_id: str, name: str, description: str,
                   permissions: Set[str] = None) -> bool:
        """创建角色"""
        if role_id in self.roles:
            logger.warning(f"角色已存在: {role_id}")
            return False

        # 验证权限是否存在
        permissions = permissions or set()
        invalid_permissions = permissions - set(self.permissions.keys())
        if invalid_permissions:
            logger.error(f"无效权限: {invalid_permissions}")
            return False

        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            permissions=permissions
        )

        self.roles[role_id] = role
        self.stats['total_roles'] += 1

        logger.info(f"创建角色: {role_id}")
        return True

    def delete_role(self, role_id: str) -> bool:
        """删除角色"""
        if role_id not in self.roles:
            return False

        role = self.roles[role_id]

        # 不能删除系统角色
        if role.is_system_role:
            logger.error(f"不能删除系统角色: {role_id}")
            return False

        # 从所有用户中移除此角色
        for user_roles in self.user_roles.values():
            user_roles.discard(role_id)

        del self.roles[role_id]
        self.stats['total_roles'] -= 1

        # 清空权限缓存
        self.permission_cache.clear()

        logger.info(f"删除角色: {role_id}")
        return True

    def add_permission_to_role(self, role_id: str, permission_id: str) -> bool:
        """为角色添加权限"""
        if role_id not in self.roles:
            logger.error(f"角色不存在: {role_id}")
            return False

        if permission_id not in self.permissions:
            logger.error(f"权限不存在: {permission_id}")
            return False

        self.roles[role_id].permissions.add(permission_id)

        # 清空相关缓存
        for user_id, user_role_set in self.user_roles.items():
            if role_id in user_role_set:
                self.permission_cache.invalidate_user(user_id)

        logger.debug(f"为角色 {role_id} 添加权限 {permission_id}")
        return True

    def remove_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """从角色中移除权限"""
        if role_id not in self.roles:
            return False

        self.roles[role_id].permissions.discard(permission_id)

        # 清空相关缓存
        for user_id, user_role_set in self.user_roles.items():
            if role_id in user_role_set:
                self.permission_cache.invalidate_user(user_id)

        logger.debug(f"从角色 {role_id} 移除权限 {permission_id}")
        return True

    # 用户权限管理

    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """为用户分配角色"""
        if role_id not in self.roles:
            logger.error(f"角色不存在: {role_id}")
            return False

        role = self.roles[role_id]

        # 检查角色用户数量限制
        if role.max_users:
            current_users = sum(1 for user_roles in self.user_roles.values()
                              if role_id in user_roles)
            if current_users >= role.max_users:
                logger.error(f"角色 {role_id} 已达到最大用户数限制: {role.max_users}")
                return False

        self.user_roles[user_id].add(role_id)

        # 清空用户缓存
        self.permission_cache.invalidate_user(user_id)

        logger.info(f"为用户 {user_id} 分配角色 {role_id}")
        return True

    def revoke_role_from_user(self, user_id: str, role_id: str) -> bool:
        """撤销用户角色"""
        if role_id not in self.user_roles[user_id]:
            return False

        self.user_roles[user_id].discard(role_id)

        # 清空用户缓存
        self.permission_cache.invalidate_user(user_id)

        logger.info(f"撤销用户 {user_id} 的角色 {role_id}")
        return True

    def grant_permission_to_user(self, user_id: str, permission_id: str) -> bool:
        """直接授予用户权限"""
        if permission_id not in self.permissions:
            logger.error(f"权限不存在: {permission_id}")
            return False

        self.user_permissions[user_id].add(permission_id)

        # 清空用户缓存
        self.permission_cache.invalidate_user(user_id)

        logger.info(f"直接授予用户 {user_id} 权限 {permission_id}")
        return True

    def revoke_permission_from_user(self, user_id: str, permission_id: str) -> bool:
        """撤销用户权限"""
        if permission_id not in self.user_permissions[user_id]:
            return False

        self.user_permissions[user_id].discard(permission_id)

        # 清空用户缓存
        self.permission_cache.invalidate_user(user_id)

        logger.info(f"撤销用户 {user_id} 的权限 {permission_id}")
        return True

    # 权限检查

    def check_permission(self, user_id: str, resource_type: ResourceType,
                        action: str, context: AccessContext = None) -> AccessDecision:
        """检查用户权限"""
        self.stats['access_checks'] += 1

        # 首先检查缓存
        cached_decision = self.permission_cache.get(user_id, resource_type, action)
        if cached_decision:
            self.stats['cache_hits'] += 1
            return cached_decision

        self.stats['cache_misses'] += 1

        # 创建访问上下文
        if context is None:
            context = AccessContext(user_id=user_id, resource_type=resource_type, action=action)

        # 收集用户的所有权限
        user_all_permissions = self._get_user_all_permissions(user_id)

        # 查找匹配的权限
        matched_permissions = []

        for perm_id in user_all_permissions:
            permission = self.permissions.get(perm_id)
            if permission and self._permission_matches(permission, resource_type, action):
                # 评估权限条件
                conditions_met, reason = self.condition_evaluator.evaluate(
                    permission.conditions, context
                )

                if conditions_met:
                    matched_permissions.append(perm_id)
                else:
                    # 有匹配权限但条件不满足
                    decision = AccessDecision(
                        allowed=False,
                        user_id=user_id,
                        resource_type=resource_type,
                        action=action,
                        denied_reason=f"权限条件不满足: {reason}",
                        conditions_met=False
                    )

                    self._log_access_decision(decision)
                    self.permission_cache.put(user_id, resource_type, action, decision)
                    self.stats['access_denied'] += 1

                    return decision

        # 做出决策
        allowed = len(matched_permissions) > 0

        decision = AccessDecision(
            allowed=allowed,
            user_id=user_id,
            resource_type=resource_type,
            action=action,
            matched_permissions=matched_permissions,
            denied_reason=None if allowed else f"缺少权限: {action} on {resource_type.value}"
        )

        # 记录决策
        self._log_access_decision(decision)

        # 缓存决策
        self.permission_cache.put(user_id, resource_type, action, decision)

        # 更新统计
        if allowed:
            self.stats['access_granted'] += 1
        else:
            self.stats['access_denied'] += 1

        return decision

    def _get_user_all_permissions(self, user_id: str) -> Set[str]:
        """获取用户的所有权限"""
        all_permissions = set()

        # 直接授予的权限
        all_permissions.update(self.user_permissions[user_id])

        # 通过角色获得的权限
        for role_id in self.user_roles[user_id]:
            if role_id in self.roles:
                role = self.roles[role_id]
                all_permissions.update(role.permissions)

                # 递归获取父角色的权限
                all_permissions.update(self._get_inherited_permissions(role_id, set()))

        return all_permissions

    def _get_inherited_permissions(self, role_id: str, visited: Set[str]) -> Set[str]:
        """递归获取继承的权限"""
        if role_id in visited or role_id not in self.roles:
            return set()

        visited.add(role_id)
        permissions = set()

        role = self.roles[role_id]
        for parent_role_id in role.parent_roles:
            if parent_role_id in self.roles:
                parent_role = self.roles[parent_role_id]
                permissions.update(parent_role.permissions)
                permissions.update(self._get_inherited_permissions(parent_role_id, visited))

        return permissions

    def _permission_matches(self, permission: Permission, resource_type: ResourceType,
                          action: str) -> bool:
        """检查权限是否匹配"""
        # 资源类型必须匹配
        if permission.resource_type != resource_type:
            return False

        # 动作匹配 (简化实现)
        permission_actions = {
            PermissionType.READ: ['read', 'view', 'get', 'list'],
            PermissionType.WRITE: ['write', 'update', 'modify', 'put', 'patch'],
            PermissionType.CREATE: ['create', 'add', 'post'],
            PermissionType.DELETE: ['delete', 'remove'],
            PermissionType.EXECUTE: ['execute', 'run', 'trade'],
            PermissionType.ADMIN: ['*']  # 管理员权限匹配所有动作
        }

        allowed_actions = permission_actions.get(permission.permission_type, [])

        return action in allowed_actions or '*' in allowed_actions

    def _log_access_decision(self, decision: AccessDecision):
        """记录访问决策"""
        self.access_log.append(decision)

        # 保持日志大小
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]  # 保留最近5000条

    # 查询接口

    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """获取用户权限列表"""
        permission_ids = self._get_user_all_permissions(user_id)
        return [self.permissions[pid] for pid in permission_ids if pid in self.permissions]

    def get_user_roles(self, user_id: str) -> List[Role]:
        """获取用户角色列表"""
        role_ids = self.user_roles[user_id]
        return [self.roles[rid] for rid in role_ids if rid in self.roles]

    def get_role_permissions(self, role_id: str) -> List[Permission]:
        """获取角色权限列表"""
        if role_id not in self.roles:
            return []

        role = self.roles[role_id]
        permissions = []

        for perm_id in role.permissions:
            if perm_id in self.permissions:
                permissions.append(self.permissions[perm_id])

        return permissions

    # 装饰器支持

    def requires_permission(self, resource_type: ResourceType, action: str):
        """权限检查装饰器"""

        def decorator(func):

            def wrapper(*args, **kwargs):
                # 从参数或上下文中获取user_id
                user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None) if args else None

                if not user_id:
                    raise ValueError("无法获取用户ID进行权限检查")

                decision = self.check_permission(user_id, resource_type, action)

                if not decision.allowed:
                    raise PermissionError(f"权限拒绝: {decision.denied_reason}")

                return func(*args, **kwargs)

            return wrapper

        return decorator

    def cleanup_expired_cache(self):
        """清理过期缓存"""
        self.permission_cache.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'stats': self.stats,
            'cache_stats': {
                'cache_size': len(self.permission_cache.cache),
                'hit_rate': self.stats['cache_hits'] / max(self.stats['access_checks'], 1)
            },
            'recent_access_log': [d.to_dict() for d in self.access_log[-10:]]
        }

# 全局实例
_permission_manager_instance = None

def get_permission_manager() -> PermissionManager:
    """获取权限管理器实例"""
    global _permission_manager_instance
    if _permission_manager_instance is None:
        _permission_manager_instance = PermissionManager()
    return _permission_manager_instance

# 便利装饰器

def requires_permission(resource_type: ResourceType, action: str):
    """权限检查装饰器"""
    manager = get_permission_manager()
    return manager.requires_permission(resource_type, action)
