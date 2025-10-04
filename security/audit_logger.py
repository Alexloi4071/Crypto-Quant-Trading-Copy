"""
Audit Logger
审计日志记录器，提供安全事件和用户行为的详细审计记录
支持多种日志格式和存储方式，确保合规性和可追溯性
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
import sys
from pathlib import Path
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class AuditEventType(Enum):
    """审计事件类型"""
    # 认证事件
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"

    # 授权事件
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"

    # API事件
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_REQUEST = "api_request"
    API_RATE_LIMITED = "api_rate_limited"

    # 交易事件
    ORDER_CREATED = "order_created"
    ORDER_EXECUTED = "order_executed"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"

    # 数据事件
    DATA_ACCESSED = "data_accessed"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"

    # 系统事件
    SYSTEM_ERROR = "system_error"
    CONFIGURATION_CHANGED = "configuration_changed"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"

    # 安全事件
    SECURITY_VIOLATION = "security_violation"
    BRUTE_FORCE_DETECTED = "brute_force_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"

class AuditLevel(Enum):
    """审计级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditOutcome(Enum):
    """审计结果"""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    ERROR = "error"

@dataclass

class AuditEvent:
    """审计事件"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]

    # 事件详情
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    level: AuditLevel = AuditLevel.MEDIUM
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    # 上下文信息
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None

    # 资源信息
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None

    # 技术信息
    application: str = "trading_system"
    service: Optional[str] = None
    version: Optional[str] = None

    # 合规字段
    data_classification: Optional[str] = None
    retention_period: Optional[int] = None  # 天数

    def to_dict(self) -> dict:
        """转换为字典"""
        data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'outcome': self.outcome.value,
            'level': self.level.value,
            'message': self.message,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id': self.request_id,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'action': self.action,
            'application': self.application,
            'service': self.service,
            'version': self.version,
            'data_classification': self.data_classification,
            'retention_period': self.retention_period
        }

        return {k: v for k, v in data.items() if v is not None}

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

@dataclass

class AuditFilter:
    """审计过滤器"""
    event_types: Optional[List[AuditEventType]] = None
    levels: Optional[List[AuditLevel]] = None
    outcomes: Optional[List[AuditOutcome]] = None
    user_ids: Optional[List[str]] = None
    resource_types: Optional[List[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    ip_addresses: Optional[List[str]] = None

    def matches(self, event: AuditEvent) -> bool:
        """检查事件是否匹配过滤器"""
        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.levels and event.level not in self.levels:
            return False

        if self.outcomes and event.outcome not in self.outcomes:
            return False

        if self.user_ids and event.user_id not in self.user_ids:
            return False

        if self.resource_types and event.resource_type not in self.resource_types:
            return False

        if self.start_time and event.timestamp < self.start_time:
            return False

        if self.end_time and event.timestamp > self.end_time:
            return False

        if self.ip_addresses and event.ip_address not in self.ip_addresses:
            return False

        return True

class AuditStorage:
    """审计存储接口"""

    async def store_event(self, event: AuditEvent) -> bool:
        """存储审计事件"""
        raise NotImplementedError

    async def query_events(self, filter: AuditFilter, limit: int = 1000) -> List[AuditEvent]:
        """查询审计事件"""
        raise NotImplementedError

class MemoryAuditStorage(AuditStorage):
    """内存审计存储"""

    def __init__(self, max_events: int = 100000):
        self.events = deque(maxlen=max_events)
        self.lock = threading.RLock()

    async def store_event(self, event: AuditEvent) -> bool:
        """存储审计事件到内存"""
        try:
            with self.lock:
                self.events.append(event)
            return True
        except Exception as e:
            logger.error(f"内存存储审计事件失败: {e}")
            return False

    async def query_events(self, filter: AuditFilter, limit: int = 1000) -> List[AuditEvent]:
        """从内存查询审计事件"""
        with self.lock:
            matched_events = []

            for event in reversed(self.events):  # 从最新开始
                if filter.matches(event):
                    matched_events.append(event)

                    if len(matched_events) >= limit:
                        break

            return matched_events

class FileAuditStorage(AuditStorage):
    """文件审计存储"""

    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.lock = threading.RLock()

    def _get_log_file(self, timestamp: datetime) -> Path:
        """获取日志文件路径"""
        date_str = timestamp.strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"

    async def store_event(self, event: AuditEvent) -> bool:
        """存储审计事件到文件"""
        try:
            log_file = self._get_log_file(event.timestamp)

            with self.lock:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(event.to_json() + '\n')

            return True

        except Exception as e:
            logger.error(f"文件存储审计事件失败: {e}")
            return False

    async def query_events(self, filter: AuditFilter, limit: int = 1000) -> List[AuditEvent]:
        """从文件查询审计事件"""
        matched_events = []

        try:
            # 确定要搜索的文件范围
            start_date = filter.start_time.date() if filter.start_time else None
            end_date = filter.end_time.date() if filter.end_time else None

            if not start_date:
                start_date = datetime.now().date() - timedelta(days=30)  # 默认搜索30天
            if not end_date:
                end_date = datetime.now().date()

            # 遍历日期范围内的文件
            current_date = start_date
            while current_date <= end_date:
                log_file = self._get_log_file(datetime.combine(current_date, datetime.min.time()))

                if log_file.exists():
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                event_data = json.loads(line.strip())
                                event = self._dict_to_audit_event(event_data)

                                if filter.matches(event):
                                    matched_events.append(event)

                                    if len(matched_events) >= limit:
                                        return matched_events

                            except Exception as e:
                                logger.error(f"解析审计事件失败: {e}")
                                continue

                current_date += timedelta(days=1)

            return matched_events

        except Exception as e:
            logger.error(f"文件查询审计事件失败: {e}")
            return []

    def _dict_to_audit_event(self, data: dict) -> AuditEvent:
        """将字典转换为审计事件对象"""
        return AuditEvent(
            event_id=data['event_id'],
            event_type=AuditEventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            outcome=AuditOutcome(data.get('outcome', 'success')),
            level=AuditLevel(data.get('level', 'medium')),
            message=data.get('message', ''),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            request_id=data.get('request_id'),
            resource_type=data.get('resource_type'),
            resource_id=data.get('resource_id'),
            action=data.get('action'),
            application=data.get('application', 'trading_system'),
            service=data.get('service'),
            version=data.get('version'),
            data_classification=data.get('data_classification'),
            retention_period=data.get('retention_period')
        )

class DatabaseAuditStorage(AuditStorage):
    """数据库审计存储（示例实现）"""

    def __init__(self, connection_pool=None):
        self.connection_pool = connection_pool

    async def store_event(self, event: AuditEvent) -> bool:
        """存储审计事件到数据库"""
        if not self.connection_pool:
            return False

        try:
            # 这里是示例，实际需要根据具体数据库实现
            query = """
            INSERT INTO audit_events (
                event_id, event_type, timestamp, user_id, session_id,
                outcome, level, message, details, ip_address, user_agent,
                request_id, resource_type, resource_id, action,
                application, service, version, data_classification, retention_period
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            # 获取连接并执行插入
            # connection = await self.connection_pool.acquire()
            # await connection.execute(query, ...)
            # await self.connection_pool.release(connection)

            return True

        except Exception as e:
            logger.error(f"数据库存储审计事件失败: {e}")
            return False

    async def query_events(self, filter: AuditFilter, limit: int = 1000) -> List[AuditEvent]:
        """从数据库查询审计事件"""
        # 实际实现中需要构建SQL查询
        return []

class AuditEventEnricher:
    """审计事件增强器"""

    def __init__(self):
        self.enrichers: List[Callable[[AuditEvent], AuditEvent]] = []

    def add_enricher(self, enricher: Callable[[AuditEvent], AuditEvent]):
        """添加增强器"""
        self.enrichers.append(enricher)

    def enrich_event(self, event: AuditEvent) -> AuditEvent:
        """增强审计事件"""
        enriched_event = event

        for enricher in self.enrichers:
            try:
                enriched_event = enricher(enriched_event)
            except Exception as e:
                logger.error(f"事件增强失败: {e}")

        return enriched_event

def geo_location_enricher(event: AuditEvent) -> AuditEvent:
    """地理位置增强器"""
    if event.ip_address and event.ip_address not in ['127.0.0.1', 'localhost']:
        try:
            # 这里可以调用地理位置服务
            # 示例：使用GeoIP数据库
            event.details['geo_location'] = {
                'country': 'Unknown',
                'city': 'Unknown',
                'latitude': None,
                'longitude': None
            }
        except Exception as e:
            logger.error(f"地理位置增强失败: {e}")

    return event

def risk_score_enricher(event: AuditEvent) -> AuditEvent:
    """风险评分增强器"""
    risk_score = 0

    # 基于事件类型的风险评分
    high_risk_events = {
        AuditEventType.LOGIN_FAILURE,
        AuditEventType.ACCESS_DENIED,
        AuditEventType.BRUTE_FORCE_DETECTED,
        AuditEventType.SECURITY_VIOLATION,
        AuditEventType.SUSPICIOUS_ACTIVITY
    }

    if event.event_type in high_risk_events:
        risk_score += 50

    # 基于结果的风险评分
    if event.outcome == AuditOutcome.FAILURE:
        risk_score += 20
    elif event.outcome == AuditOutcome.ERROR:
        risk_score += 30

    # 基于级别的风险评分
    if event.level == AuditLevel.CRITICAL:
        risk_score += 40
    elif event.level == AuditLevel.HIGH:
        risk_score += 30

    event.details['risk_score'] = min(risk_score, 100)
    return event

class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable[[AuditEvent], None]] = []

        # 预定义告警规则
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """设置默认告警规则"""
        self.alert_rules = [
            {
                'name': 'critical_security_event',
                'condition': lambda event: event.level == AuditLevel.CRITICAL,
                'description': '关键安全事件告警'
            },
            {
                'name': 'multiple_login_failures',
                'condition': lambda event: (
                    event.event_type == AuditEventType.LOGIN_FAILURE and
                    event.details.get('consecutive_failures', 0) >= 5
                ),
                'description': '多次登录失败告警'
            },
            {
                'name': 'suspicious_api_usage',
                'condition': lambda event: (
                    event.event_type == AuditEventType.API_RATE_LIMITED and
                    event.details.get('requests_per_minute', 0) > 1000
                ),
                'description': 'API使用异常告警'
            }
        ]

    def add_alert_rule(self, name: str, condition: Callable[[AuditEvent], bool], description: str):
        """添加告警规则"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'description': description
        })

    def add_alert_handler(self, handler: Callable[[AuditEvent], None]):
        """添加告警处理器"""
        self.alert_handlers.append(handler)

    def check_alerts(self, event: AuditEvent):
        """检查告警条件"""
        for rule in self.alert_rules:
            try:
                if rule['condition'](event):
                    self._trigger_alert(event, rule)
            except Exception as e:
                logger.error(f"告警规则检查失败 {rule['name']}: {e}")

    def _trigger_alert(self, event: AuditEvent, rule: Dict[str, Any]):
        """触发告警"""
        logger.warning(f"触发告警: {rule['name']} - {rule['description']}")

        for handler in self.alert_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"告警处理失败: {e}")

class AuditLogger:
    """审计日志记录器主类"""

    def __init__(self):
        # 存储配置
        audit_config = config.get('audit', {})
        storage_type = audit_config.get('storage_type', 'memory')

        if storage_type == 'file':
            self.storage = FileAuditStorage(audit_config.get('log_dir', 'logs/audit'))
        elif storage_type == 'database':
            # 需要传入数据库连接池
            self.storage = DatabaseAuditStorage()
        else:
            self.storage = MemoryAuditStorage(audit_config.get('max_events', 100000))

        # 组件
        self.enricher = AuditEventEnricher()
        self.alert_manager = AlertManager()

        # 设置默认增强器
        self.enricher.add_enricher(geo_location_enricher)
        self.enricher.add_enricher(risk_score_enricher)

        # 异步队列和处理器
        self.event_queue = asyncio.Queue(maxsize=10000)
        self.processing_task = None
        self.is_running = False

        # 统计信息
        self.stats = {
            'events_logged': 0,
            'events_processed': 0,
            'events_failed': 0,
            'alerts_triggered': 0,
            'start_time': datetime.now()
        }

        # 设置默认告警处理器
        self.alert_manager.add_alert_handler(self._default_alert_handler)

        logger.info("审计日志记录器初始化完成")

    async def start(self):
        """启动审计日志记录器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动事件处理任务
        self.processing_task = asyncio.create_task(self._process_events())

        logger.info("审计日志记录器启动完成")

    async def stop(self):
        """停止审计日志记录器"""
        if not self.is_running:
            return

        logger.info("正在停止审计日志记录器...")
        self.is_running = False

        # 处理剩余事件
        while not self.event_queue.empty():
            await asyncio.sleep(0.1)

        # 取消处理任务
        if self.processing_task:
            self.processing_task.cancel()

        logger.info("审计日志记录器已停止")

    async def _process_events(self):
        """处理审计事件队列"""
        while self.is_running:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)

                try:
                    # 增强事件
                    enriched_event = self.enricher.enrich_event(event)

                    # 存储事件
                    success = await self.storage.store_event(enriched_event)

                    if success:
                        self.stats['events_processed'] += 1
                    else:
                        self.stats['events_failed'] += 1

                    # 检查告警
                    self.alert_manager.check_alerts(enriched_event)

                except Exception as e:
                    logger.error(f"处理审计事件失败: {e}")
                    self.stats['events_failed'] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"审计事件处理循环错误: {e}")
                await asyncio.sleep(1)

    def _default_alert_handler(self, event: AuditEvent):
        """默认告警处理器"""
        logger.critical(f"安全告警: {event.event_type.value} - {event.message}")
        self.stats['alerts_triggered'] += 1

    # 公共接口

    async def log_event(self, event_type: AuditEventType, user_id: str = None,
                       session_id: str = None, message: str = "",
                       outcome: AuditOutcome = AuditOutcome.SUCCESS,
                       level: AuditLevel = AuditLevel.MEDIUM,
                       **kwargs) -> str:
        """记录审计事件"""
        event_id = f"audit_{int(time.time() * 1000000)}"

        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            outcome=outcome,
            level=level,
            message=message,
            **kwargs
        )

        try:
            await self.event_queue.put(event)
            self.stats['events_logged'] += 1
            return event_id
        except asyncio.QueueFull:
            logger.error("审计事件队列已满，丢弃事件")
            self.stats['events_failed'] += 1
            return event_id

    def log_event_sync(self, event_type: AuditEventType, user_id: str = None,
                      **kwargs) -> str:
        """同步记录审计事件"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.log_event(event_type, user_id, **kwargs)
            )
        except RuntimeError:
            # 如果没有事件循环，直接存储
            event_id = f"audit_{int(time.time() * 1000000)}"

            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                **kwargs
            )

            # 直接存储（阻塞操作）
            try:
                enriched_event = self.enricher.enrich_event(event)
                # 这里需要同步版本的存储方法
                logger.info(f"同步记录审计事件: {event_id}")
                self.stats['events_logged'] += 1
            except Exception as e:
                logger.error(f"同步记录审计事件失败: {e}")
                self.stats['events_failed'] += 1

            return event_id

    async def query_events(self, filter: AuditFilter = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """查询审计事件"""
        if filter is None:
            filter = AuditFilter()

        return await self.storage.query_events(filter, limit)

    # 便利方法

    async def log_login_success(self, user_id: str, session_id: str, ip_address: str = None):
        """记录登录成功"""
        await self.log_event(
            AuditEventType.LOGIN_SUCCESS,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            message=f"用户 {user_id} 登录成功"
        )

    async def log_login_failure(self, user_id: str, reason: str, ip_address: str = None):
        """记录登录失败"""
        await self.log_event(
            AuditEventType.LOGIN_FAILURE,
            user_id=user_id,
            ip_address=ip_address,
            outcome=AuditOutcome.FAILURE,
            level=AuditLevel.HIGH,
            message=f"用户 {user_id} 登录失败: {reason}",
            details={'failure_reason': reason}
        )

    async def log_access_denied(self, user_id: str, resource_type: str,
                              resource_id: str, action: str):
        """记录访问拒绝"""
        await self.log_event(
            AuditEventType.ACCESS_DENIED,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=AuditOutcome.FAILURE,
            level=AuditLevel.MEDIUM,
            message=f"用户 {user_id} 访问被拒绝: {action} on {resource_type}/{resource_id}"
        )

    async def log_trade_execution(self, user_id: str, order_id: str,
                                symbol: str, quantity: float, price: float):
        """记录交易执行"""
        await self.log_event(
            AuditEventType.TRADE_EXECUTED,
            user_id=user_id,
            resource_type="trade",
            resource_id=order_id,
            action="execute",
            level=AuditLevel.HIGH,
            message=f"执行交易: {symbol} {quantity}@{price}",
            details={
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'order_id': order_id
            }
        )

    async def log_security_violation(self, user_id: str, violation_type: str,
                                   description: str, ip_address: str = None):
        """记录安全违规"""
        await self.log_event(
            AuditEventType.SECURITY_VIOLATION,
            user_id=user_id,
            ip_address=ip_address,
            outcome=AuditOutcome.WARNING,
            level=AuditLevel.CRITICAL,
            message=f"安全违规: {violation_type}",
            details={
                'violation_type': violation_type,
                'description': description
            }
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = datetime.now() - self.stats['start_time']

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime.total_seconds(),
            'stats': self.stats,
            'queue_size': self.event_queue.qsize() if self.event_queue else 0,
            'storage_type': type(self.storage).__name__,
            'alert_rules_count': len(self.alert_manager.alert_rules),
            'enrichers_count': len(self.enricher.enrichers)
        }

# 全局实例
_audit_logger_instance = None

def get_audit_logger() -> AuditLogger:
    """获取审计日志记录器实例"""
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance

# 便利装饰器

def audit_action(event_type: AuditEventType, resource_type: str = None, action: str = None):
    """审计装饰器"""

    def decorator(func):

        async def async_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None) if args else None

            try:
                result = await func(*args, **kwargs)

                await audit_logger.log_event(
                    event_type,
                    user_id=user_id,
                    resource_type=resource_type,
                    action=action,
                    message=f"执行操作: {func.__name__}"
                )

                return result

            except Exception as e:
                await audit_logger.log_event(
                    event_type,
                    user_id=user_id,
                    resource_type=resource_type,
                    action=action,
                    outcome=AuditOutcome.ERROR,
                    level=AuditLevel.HIGH,
                    message=f"操作失败: {func.__name__}",
                    details={'error': str(e)}
                )
                raise

        def sync_wrapper(*args, **kwargs):
            audit_logger = get_audit_logger()
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None) if args else None

            try:
                result = func(*args, **kwargs)

                audit_logger.log_event_sync(
                    event_type,
                    user_id=user_id,
                    resource_type=resource_type,
                    action=action,
                    message=f"执行操作: {func.__name__}"
                )

                return result

            except Exception as e:
                audit_logger.log_event_sync(
                    event_type,
                    user_id=user_id,
                    resource_type=resource_type,
                    action=action,
                    outcome=AuditOutcome.ERROR,
                    level=AuditLevel.HIGH,
                    message=f"操作失败: {func.__name__}",
                    details={'error': str(e)}
                )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
