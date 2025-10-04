"""
API Dependencies
依赖注入管理，提供系统组件的单例访问
与现有系统深度集成，确保组件一致性
"""

from fastapi import Depends, HTTPException, status, Request
from typing import Optional, AsyncGenerator
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入现有系统组件
from src.trading.trading_system import TradingSystem
from src.data.data_manager import DataManager
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.alerting import AlertingSystem
from src.monitoring.notifications import NotificationService
from src.monitoring.health_checker import HealthChecker
from src.utils.database_manager import DatabaseManager
from src.utils.logger import setup_logger

# 导入API配置
from api.config import api_config

# 设置日志
logger = setup_logger(__name__)

class DependencyError(Exception):
    """依赖注入错误"""
    pass

class SystemDependencies:
    """系统依赖管理器"""
    
    def __init__(self):
        self._trading_system: Optional[TradingSystem] = None
        self._data_manager: Optional[DataManager] = None
        self._system_monitor: Optional[SystemMonitor] = None
        self._alerting_system: Optional[AlertingSystem] = None
        self._notification_service: Optional[NotificationService] = None
        self._health_checker: Optional[HealthChecker] = None
        self._db_manager: Optional[DatabaseManager] = None
        
    def set_instances(self, app_state):
        """从FastAPI应用状态设置实例"""
        self._trading_system = getattr(app_state, 'trading_system', None)
        self._data_manager = getattr(app_state, 'data_manager', None)
        self._system_monitor = getattr(app_state, 'system_monitor', None)
        self._alerting_system = getattr(app_state, 'alerting_system', None)
        self._notification_service = getattr(app_state, 'notification_service', None)
        self._health_checker = getattr(app_state, 'health_checker', None)
        self._db_manager = getattr(app_state, 'db_manager', None)
    
    @property
    def trading_system(self) -> Optional[TradingSystem]:
        return self._trading_system
    
    @property
    def data_manager(self) -> Optional[DataManager]:
        return self._data_manager
    
    @property
    def system_monitor(self) -> Optional[SystemMonitor]:
        return self._system_monitor
    
    @property
    def alerting_system(self) -> Optional[AlertingSystem]:
        return self._alerting_system
    
    @property
    def notification_service(self) -> Optional[NotificationService]:
        return self._notification_service
    
    @property
    def health_checker(self) -> Optional[HealthChecker]:
        return self._health_checker
    
    @property
    def db_manager(self) -> Optional[DatabaseManager]:
        return self._db_manager

# 全局依赖管理器实例
system_deps = SystemDependencies()

async def get_request_state(request: Request):
    """获取请求状态和应用实例"""
    return request.app.state

async def get_trading_system(request: Request) -> TradingSystem:
    """获取交易系统实例"""
    try:
        state = await get_request_state(request)
        trading_system = getattr(state, 'trading_system', None)
        
        if trading_system is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="交易系统未启动或不可用"
            )
        
        return trading_system
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问交易系统实例"
        )

async def get_trading_system_optional(request: Request) -> Optional[TradingSystem]:
    """获取交易系统实例（可选）"""
    try:
        state = await get_request_state(request)
        return getattr(state, 'trading_system', None)
    except:
        return None

async def get_data_manager(request: Request) -> DataManager:
    """获取数据管理器实例"""
    try:
        state = await get_request_state(request)
        data_manager = getattr(state, 'data_manager', None)
        
        if data_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据管理器未初始化或不可用"
            )
        
        return data_manager
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问数据管理器实例"
        )

async def get_system_monitor(request: Request) -> SystemMonitor:
    """获取系统监控实例"""
    try:
        state = await get_request_state(request)
        system_monitor = getattr(state, 'system_monitor', None)
        
        if system_monitor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="系统监控未启动或不可用"
            )
        
        return system_monitor
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问系统监控实例"
        )

async def get_alerting_system(request: Request) -> AlertingSystem:
    """获取告警系统实例"""
    try:
        state = await get_request_state(request)
        alerting_system = getattr(state, 'alerting_system', None)
        
        if alerting_system is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="告警系统未启动或不可用"
            )
        
        return alerting_system
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问告警系统实例"
        )

async def get_notification_service(request: Request) -> NotificationService:
    """获取通知服务实例"""
    try:
        state = await get_request_state(request)
        notification_service = getattr(state, 'notification_service', None)
        
        if notification_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="通知服务未启动或不可用"
            )
        
        return notification_service
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问通知服务实例"
        )

async def get_health_checker(request: Request) -> HealthChecker:
    """获取健康检查器实例"""
    try:
        state = await get_request_state(request)
        health_checker = getattr(state, 'health_checker', None)
        
        if health_checker is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="健康检查器未启动或不可用"
            )
        
        return health_checker
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问健康检查器实例"
        )

async def get_database_manager(request: Request) -> DatabaseManager:
    """获取数据库管理器实例"""
    try:
        state = await get_request_state(request)
        db_manager = getattr(state, 'db_manager', None)
        
        if db_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据库管理器未初始化或不可用"
            )
        
        if not db_manager.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据库连接不可用"
            )
        
        return db_manager
    
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法访问数据库管理器实例"
        )

# 分页参数依赖
class PaginationParams:
    """分页参数"""
    
    def __init__(self, page: int = 1, size: int = 20):
        self.page = max(1, page)
        self.size = min(max(1, size), 100)  # 限制最大页面大小
        self.offset = (self.page - 1) * self.size
        self.limit = self.size

def get_pagination(page: int = 1, size: int = 20) -> PaginationParams:
    """获取分页参数"""
    return PaginationParams(page, size)

# 查询参数依赖
class QueryParams:
    """查询参数"""
    
    def __init__(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.limit = min(limit or 100, 1000) if limit else 100

def get_query_params(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None
) -> QueryParams:
    """获取查询参数"""
    return QueryParams(symbol, timeframe, start_date, end_date, limit)

# 交易参数依赖
class TradingParams:
    """交易参数"""
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        mode: str = "paper",
        capital: float = 10000,
        max_positions: int = 5
    ):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.mode = mode
        self.capital = max(1000, capital)  # 最小资金限制
        self.max_positions = min(max(1, max_positions), 10)  # 位置数量限制

def get_trading_params(
    symbols: Optional[str] = None,
    mode: str = "paper", 
    capital: float = 10000,
    max_positions: int = 5
) -> TradingParams:
    """获取交易参数"""
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
    
    return TradingParams(symbol_list, mode, capital, max_positions)

# 监控参数依赖  
class MonitoringParams:
    """监控参数"""
    
    def __init__(
        self,
        metric_types: Optional[list] = None,
        time_range: str = "1h",
        include_history: bool = False
    ):
        self.metric_types = metric_types or ['system', 'trading']
        self.time_range = time_range
        self.include_history = include_history

def get_monitoring_params(
    metric_types: Optional[str] = None,
    time_range: str = "1h",
    include_history: bool = False
) -> MonitoringParams:
    """获取监控参数"""
    types_list = None
    if metric_types:
        types_list = [t.strip().lower() for t in metric_types.split(',')]
    
    return MonitoringParams(types_list, time_range, include_history)

# 认证依赖 (如果启用认证)
async def get_current_user(request: Request):
    """获取当前用户 (如果启用认证)"""
    if not api_config.ENABLE_AUTH:
        return {"user_id": "anonymous", "permissions": ["read", "write"]}
    
    # 这里实现JWT认证逻辑
    authorization = request.headers.get("Authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split(" ")[1]
    # TODO: 实现JWT令牌验证逻辑
    
    return {"user_id": "user123", "permissions": ["read", "write"]}

def require_permission(permission: str):
    """权限检查装饰器"""
    async def permission_checker(current_user: dict = Depends(get_current_user)):
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"缺少必需的权限: {permission}"
            )
        return current_user
    
    return permission_checker

# 系统状态依赖
async def check_system_healthy(request: Request):
    """检查系统是否健康"""
    try:
        # 检查关键组件是否可用
        state = await get_request_state(request)
        
        # 检查数据管理器
        data_manager = getattr(state, 'data_manager', None)
        if data_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="数据管理器不可用"
            )
        
        # 检查系统监控
        system_monitor = getattr(state, 'system_monitor', None)
        if system_monitor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="系统监控不可用"
            )
        
        return True
    
    except Exception as e:
        logger.error(f"系统健康检查失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="系统组件不健康"
        )

# 缓存依赖
class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self._cache = {}
        self._ttl = {}
    
    def get(self, key: str):
        """获取缓存"""
        import time
        
        if key not in self._cache:
            return None
        
        # 检查TTL
        if key in self._ttl and self._ttl[key] < time.time():
            del self._cache[key]
            del self._ttl[key]
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value, ttl_seconds: int = None):
        """设置缓存"""
        import time
        
        self._cache[key] = value
        
        if ttl_seconds:
            self._ttl[key] = time.time() + ttl_seconds
    
    def delete(self, key: str):
        """删除缓存"""
        self._cache.pop(key, None)
        self._ttl.pop(key, None)
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._ttl.clear()

# 全局缓存管理器实例
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """获取缓存管理器"""
    return cache_manager

# 响应包装器
async def create_response(data=None, message="操作成功", success=True):
    """创建标准API响应"""
    import datetime
    
    return {
        "success": success,
        "data": data,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

async def create_error_response(message="操作失败", code=None, details=None):
    """创建错误响应"""
    import datetime
    
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "details": details
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }