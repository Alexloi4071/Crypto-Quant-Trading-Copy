"""
API Response Models
API响应模型定义，提供标准化的响应格式
与现有系统组件数据结构完全兼容
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

# 基础响应模型

class APIResponse(BaseModel):
    """标准API响应格式"""
    success: bool
    data: Optional[Any] = None
    message: str = "操作成功"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")

class ErrorResponse(BaseModel):
    """错误响应格式"""
    success: bool = False
    error: "ErrorDetail"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat() + "Z")

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str
    message: str
    details: Optional[Any] = None

# 分页响应模型

class PaginationInfo(BaseModel):
    """分页信息"""
    page: int
    size: int
    total: int
    pages: int

class PaginatedResponse(BaseModel):
    """分页响应"""
    data: List[Any]
    pagination: PaginationInfo

# 组合相关模型

class PositionModel(BaseModel):
    """持仓模型"""
    id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: str = "open"
    risk_level: str = "medium"

class PortfolioSummary(BaseModel):
    """组合概览"""
    portfolio_value: float
    initial_capital: float
    total_pnl: float
    total_return_pct: float
    daily_pnl: float
    daily_return_pct: float
    cash: float
    positions: List[PositionModel]
    risk_metrics: Dict[str, Any]
    last_updated: str
    currency: str = "USD"
    status: str

class RiskMetrics(BaseModel):
    """风险指标"""
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    var_95: Optional[float] = None
    portfolio_correlation: Optional[float] = None

class PerformanceMetrics(BaseModel):
    """性能指标"""
    period: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_trade: float
    best_trade: float
    worst_trade: float

# 交易相关模型

class TradingStatus(BaseModel):
    """交易状态"""
    system_status: str
    trading_mode: str
    active_symbols: List[str]
    strategies_running: int
    auto_trading: bool
    last_signal_time: Optional[str] = None
    last_trade_time: Optional[str] = None
    uptime_seconds: int
    daily_trades: int = 0
    daily_pnl: float = 0
    signals_generated: int = 0
    positions_open: int = 0
    config: Dict[str, Any]

class SignalModel(BaseModel):
    """交易信号模型"""
    id: Optional[str] = None
    symbol: str
    timeframe: str
    signal_type: str
    confidence: float
    price: float
    source: str
    timestamp: Optional[str] = None
    strength: str  # 'very_strong', 'strong', 'medium', 'weak'
    direction: str  # 'bullish', 'bearish', 'neutral'
    metadata: Dict[str, Any] = {}
    valid_until: Optional[str] = None
    generated_by: str = "system"

class TradeModel(BaseModel):
    """交易记录模型"""
    id: str
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration: Optional[int] = None  # 持仓时间（秒）
    strategy: Optional[str] = None
    signal_confidence: Optional[float] = None
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    status: str
    notes: Optional[str] = None

# 数据相关模型

class OHLCVPoint(BaseModel):
    """OHLCV数据点"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketData(BaseModel):
    """市场数据"""
    symbol: str
    timeframe: str
    data: List[OHLCVPoint]
    statistics: Dict[str, Any]

class SymbolInfo(BaseModel):
    """交易对信息"""
    symbol: str
    exchange: str
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None
    status: str
    price_precision: Optional[int] = None
    quantity_precision: Optional[int] = None
    min_quantity: Optional[float] = None
    last_update: Optional[str] = None

class PriceStats(BaseModel):
    """价格统计"""
    current_price: float
    open_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    price_change_24h: float
    price_change_pct_24h: float
    high_30d: Optional[float] = None
    low_30d: Optional[float] = None
    avg_volume_30d: Optional[float] = None
    volatility_30d: Optional[float] = None

# 监控相关模型

class MetricModel(BaseModel):
    """系统指标模型"""
    name: str
    value: float
    unit: str
    status: str  # 'healthy', 'warning', 'critical'
    category: str
    timestamp: str
    threshold: Optional[Dict[str, float]] = None
    history: Optional[List[Dict[str, Any]]] = None

class AlertModel(BaseModel):
    """告警模型"""
    id: str
    title: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str
    status: str  # 'active', 'acknowledged', 'resolved'
    source: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    created_at: str
    updated_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[str] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class SystemHealth(BaseModel):
    """系统健康状态"""
    overall_status: str  # 'healthy', 'degraded', 'unhealthy'
    health_score: int  # 0-100
    last_check: str
    components: Dict[str, str]
    system_resources: Dict[str, Any]
    services: Dict[str, str]
    detailed_checks: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    recent_issues: Optional[List[Dict[str, Any]]] = None

# 认证相关模型

class UserModel(BaseModel):
    """用户模型"""
    user_id: str
    username: str
    permissions: List[str]
    exp: Optional[float] = None

class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserModel

class TokenResponse(BaseModel):
    """Token响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

# 请求模型

class ManualTradeRequest(BaseModel):
    """手动交易请求"""
    symbol: str = Field(..., description="交易对")
    side: str = Field(..., description="交易方向: buy, sell")
    size: Optional[float] = Field(None, description="交易数量")
    order_type: str = Field("market", description="订单类型: market, limit")
    price: Optional[float] = Field(None, description="限价单价格")
    stop_loss_pct: Optional[float] = Field(0.02, description="止损百分比")
    take_profit_pct: Optional[float] = Field(0.04, description="止盈百分比")
    notes: Optional[str] = Field(None, description="交易备注")

class TradingControlRequest(BaseModel):
    """交易控制请求"""
    action: str = Field(..., description="操作: start, stop, pause, resume, emergency_stop")
    symbols: Optional[List[str]] = Field(None, description="影响的交易对")
    reason: Optional[str] = Field(None, description="操作原因")

class StrategyUpdateRequest(BaseModel):
    """策略更新请求"""
    strategy_name: str = Field(..., description="策略名称")
    parameters: Dict[str, Any] = Field(..., description="策略参数")
    symbols: Optional[List[str]] = Field(None, description="应用的交易对")

class ManualSignalRequest(BaseModel):
    """手动信号请求"""
    symbol: str = Field(..., description="交易对")
    signal_type: str = Field(..., description="信号类型: buy, sell, neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="信心度 0-1")
    timeframe: str = Field("1h", description="时间框架")
    reason: Optional[str] = Field(None, description="信号原因")
    notes: Optional[str] = Field(None, description="备注")

# 枚举类型

class TradingMode(str, Enum):
    """交易模式"""
    SIMULATION = "simulation"
    PAPER = "paper"
    LIVE = "live"

class OrderType(str, Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class PositionSide(str, Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"

class SignalType(str, Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"

class AlertSeverity(str, Enum):
    """告警严重级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"

class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

# 响应构造工具

class ResponseBuilder:
    """响应构造工具类"""

    @staticmethod

    def success(data: Any = None, message: str = "操作成功") -> Dict[str, Any]:
        """构造成功响应"""
        return {
            "success": True,
            "data": data,
            "message": message,
            "timestamp": datetime.now().isoformat() + "Z"
        }

    @staticmethod

    def error(code: str, message: str, details: Any = None) -> Dict[str, Any]:
        """构造错误响应"""
        return {
            "success": False,
            "error": {
                "code": code,
                "message": message,
                "details": details
            },
            "timestamp": datetime.now().isoformat() + "Z"
        }

    @staticmethod

    def paginated(data: List[Any], page: int, size: int, total: int) -> Dict[str, Any]:
        """构造分页响应"""
        return {
            "success": True,
            "data": {
                "items": data,
                "pagination": {
                    "page": page,
                    "size": size,
                    "total": total,
                    "pages": (total + size - 1) // size
                }
            },
            "message": "数据获取成功",
            "timestamp": datetime.now().isoformat() + "Z"
        }

# 数据验证工具

class DataValidator:
    """数据验证工具"""

    @staticmethod

    def validate_symbol(symbol: str) -> bool:
        """验证交易对格式"""
        if not symbol or not isinstance(symbol, str):
            return False
        # 简单验证：应该包含字母和数字，长度在3-20之间
        return 3 <= len(symbol) <= 20 and symbol.isalnum()

    @staticmethod

    def validate_timeframe(timeframe: str) -> bool:
        """验证时间框架格式"""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        return timeframe in valid_timeframes

    @staticmethod

    def validate_confidence(confidence: float) -> bool:
        """验证信心度范围"""
        return 0.0 <= confidence <= 1.0

    @staticmethod

    def validate_percentage(percentage: float) -> bool:
        """验证百分比范围"""
        return 0.0 <= percentage <= 100.0

# 常用响应模板
COMMON_RESPONSES = {
    "UNAUTHORIZED": {
        "success": False,
        "error": {
            "code": "UNAUTHORIZED",
            "message": "未授权访问，请提供有效的认证令牌"
        }
    },

    "FORBIDDEN": {
        "success": False,
        "error": {
            "code": "FORBIDDEN",
            "message": "权限不足，无法执行此操作"
        }
    },

    "NOT_FOUND": {
        "success": False,
        "error": {
            "code": "NOT_FOUND",
            "message": "请求的资源不存在"
        }
    },

    "VALIDATION_ERROR": {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "请求参数验证失败"
        }
    },

    "SYSTEM_ERROR": {
        "success": False,
        "error": {
            "code": "SYSTEM_ERROR",
            "message": "系统内部错误，请稍后重试"
        }
    },

    "SERVICE_UNAVAILABLE": {
        "success": False,
        "error": {
            "code": "SERVICE_UNAVAILABLE",
            "message": "服务暂时不可用，请稍后重试"
        }
    }
}

# 导出所有模型类
__all__ = [
    # 基础响应模型
    'APIResponse', 'ErrorResponse', 'ErrorDetail',
    'PaginationInfo', 'PaginatedResponse',

    # 业务模型
    'PositionModel', 'PortfolioSummary', 'RiskMetrics', 'PerformanceMetrics',
    'TradingStatus', 'SignalModel', 'TradeModel',
    'OHLCVPoint', 'MarketData', 'SymbolInfo', 'PriceStats',
    'MetricModel', 'AlertModel', 'SystemHealth',
    'UserModel', 'LoginRequest', 'LoginResponse', 'TokenResponse',

    # 请求模型
    'ManualTradeRequest', 'TradingControlRequest', 'StrategyUpdateRequest',
    'ManualSignalRequest',

    # 枚举类型
    'TradingMode', 'OrderType', 'PositionSide', 'SignalType',
    'AlertSeverity', 'AlertStatus', 'HealthStatus',

    # 工具类
    'ResponseBuilder', 'DataValidator'
]
