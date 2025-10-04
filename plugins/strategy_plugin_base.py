"""
Strategy Plugin Base
策略插件基类，为交易策略插件提供标准接口和基础功能
支持策略参数配置、信号生成和性能跟踪
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.plugins.plugin_manager import PluginBase, PluginType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    REDUCE_POSITION = "reduce_position"
    INCREASE_POSITION = "increase_position"

class SignalStrength(Enum):
    """信号强度"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass

class TradingSignal:
    """交易信号"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1之间

    # 价格信息
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # 数量信息
    suggested_quantity: Optional[float] = None
    position_size_ratio: Optional[float] = None  # 仓位比例

    # 时间信息
    timestamp: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None

    # 额外信息
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 策略信息
    strategy_id: str = ""
    strategy_name: str = ""

    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'current_price': self.current_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'suggested_quantity': self.suggested_quantity,
            'position_size_ratio': self.position_size_ratio,
            'timestamp': self.timestamp.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'reason': self.reason,
            'metadata': self.metadata,
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name
        }

@dataclass

class StrategyParameters:
    """策略参数"""
    # 基础参数
    timeframe: str = "1D"  # 时间框架
    lookback_period: int = 20  # 回看期
    min_confidence: float = 0.6  # 最小置信度

    # 风险管理
    max_position_size: float = 0.1  # 最大仓位比例
    stop_loss_pct: float = 0.05  # 止损百分比
    take_profit_pct: float = 0.1  # 止盈百分比

    # 交易频率
    max_signals_per_day: int = 5
    min_signal_interval: int = 60  # 最小信号间隔（分钟）

    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'timeframe': self.timeframe,
            'lookback_period': self.lookback_period,
            'min_confidence': self.min_confidence,
            'max_position_size': self.max_position_size,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_signals_per_day': self.max_signals_per_day,
            'min_signal_interval': self.min_signal_interval,
            'custom_params': self.custom_params
        }

@dataclass

class StrategyPerformance:
    """策略性能指标"""
    # 基础指标
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    win_rate: float = 0.0

    # 收益指标
    total_return: float = 0.0
    average_return_per_signal: float = 0.0
    best_signal_return: float = 0.0
    worst_signal_return: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # 时间指标
    average_holding_time: float = 0.0  # 小时
    total_active_time: float = 0.0  # 小时

    # 其他指标
    profit_factor: float = 0.0
    recovery_factor: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            'total_signals': self.total_signals,
            'successful_signals': self.successful_signals,
            'failed_signals': self.failed_signals,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'average_return_per_signal': self.average_return_per_signal,
            'best_signal_return': self.best_signal_return,
            'worst_signal_return': self.worst_signal_return,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'average_holding_time': self.average_holding_time,
            'total_active_time': self.total_active_time,
            'profit_factor': self.profit_factor,
            'recovery_factor': self.recovery_factor,
            'calmar_ratio': self.calmar_ratio
        }

class StrategyPluginBase(PluginBase):
    """策略插件基类"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # 策略基础信息
        self.strategy_name = self.__class__.__name__
        self.strategy_version = "1.0.0"
        self.strategy_description = "基础策略插件"
        self.author = "Unknown"

        # 策略参数
        self.parameters = StrategyParameters()
        self._load_parameters_from_config()

        # 性能跟踪
        self.performance = StrategyPerformance()

        # 信号历史
        self.signal_history = []
        self.last_signals = {}  # symbol -> last_signal_time

        # 数据缓存
        self.data_cache = {}  # symbol -> DataFrame
        self.cache_timestamps = {}  # symbol -> timestamp
        self.cache_ttl = 300  # 缓存TTL（秒）

        # 状态管理
        self.is_active = False
        self.last_update = datetime.now()

        logger.debug(f"策略插件初始化: {self.strategy_name}")

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'id': self.plugin_id,
            'name': self.strategy_name,
            'version': self.strategy_version,
            'description': self.strategy_description,
            'author': self.author,
            'type': 'strategy',
            'config_schema': self.get_config_schema(),
            'default_config': self._get_default_config()
        }

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式"""
        return {
            'timeframe': {
                'type': 'string',
                'description': '时间框架 (1m, 5m, 15m, 1h, 4h, 1D)',
                'default': '1D',
                'enum': ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1D']
            },
            'lookback_period': {
                'type': 'integer',
                'description': '回看期（天数）',
                'default': 20,
                'minimum': 5,
                'maximum': 200
            },
            'min_confidence': {
                'type': 'number',
                'description': '最小置信度',
                'default': 0.6,
                'minimum': 0.0,
                'maximum': 1.0
            },
            'max_position_size': {
                'type': 'number',
                'description': '最大仓位比例',
                'default': 0.1,
                'minimum': 0.01,
                'maximum': 1.0
            },
            'stop_loss_pct': {
                'type': 'number',
                'description': '止损百分比',
                'default': 0.05,
                'minimum': 0.01,
                'maximum': 0.5
            },
            'take_profit_pct': {
                'type': 'number',
                'description': '止盈百分比',
                'default': 0.1,
                'minimum': 0.01,
                'maximum': 1.0
            }
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return self.parameters.to_dict()

    def _load_parameters_from_config(self):
        """从配置加载参数"""
        if not self.config:
            return

        # 更新策略参数
        for key, value in self.config.items():
            if hasattr(self.parameters, key):
                setattr(self.parameters, key, value)
            else:
                # 自定义参数
                self.parameters.custom_params[key] = value

    def initialize(self) -> bool:
        """初始化策略"""
        try:
            # 验证参数
            if not self._validate_parameters():
                logger.error(f"策略参数验证失败: {self.strategy_name}")
                return False

            # 执行自定义初始化
            if not self._on_initialize():
                logger.error(f"策略初始化失败: {self.strategy_name}")
                return False

            self.is_active = True
            self.last_update = datetime.now()

            logger.info(f"策略 {self.strategy_name} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"策略初始化异常 {self.strategy_name}: {e}")
            return False

    def cleanup(self) -> bool:
        """清理策略"""
        try:
            self.is_active = False

            # 执行自定义清理
            self._on_cleanup()

            # 清理缓存
            self.data_cache.clear()
            self.cache_timestamps.clear()

            logger.info(f"策略 {self.strategy_name} 清理完成")
            return True

        except Exception as e:
            logger.error(f"策略清理异常 {self.strategy_name}: {e}")
            return False

    def _validate_parameters(self) -> bool:
        """验证参数"""
        # 基础验证
        if self.parameters.min_confidence < 0 or self.parameters.min_confidence > 1:
            logger.error("最小置信度必须在0-1之间")
            return False

        if self.parameters.max_position_size <= 0 or self.parameters.max_position_size > 1:
            logger.error("最大仓位比例必须在0-1之间")
            return False

        if self.parameters.lookback_period <= 0:
            logger.error("回看期必须大于0")
            return False

        # 自定义验证
        return self._validate_custom_parameters()

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成交易信号（主要接口）"""
        if not self.is_active:
            return None

        start_time = time.time()

        try:
            # 检查信号频率限制
            if not self._check_signal_frequency(symbol):
                return None

            # 数据预处理
            processed_data = self._preprocess_data(data)
            if processed_data is None or len(processed_data) < self.parameters.lookback_period:
                return None

            # 生成信号
            signal = self._generate_signal_logic(symbol, processed_data)

            # 验证信号
            if signal and self._validate_signal(signal):
                # 记录信号
                self.signal_history.append(signal)
                self.last_signals[symbol] = datetime.now()

                # 更新性能统计
                self.performance.total_signals += 1

                # 记录性能统计
                execution_time = time.time() - start_time
                self.performance_stats['call_count'] += 1
                self.performance_stats['total_time'] += execution_time
                self.performance_stats['avg_time'] = (
                    self.performance_stats['total_time'] /
                    self.performance_stats['call_count']
                )

                logger.debug(f"策略 {self.strategy_name} 为 {symbol} 生成信号: {signal.signal_type.value}")
                return signal

        except Exception as e:
            logger.error(f"生成信号异常 {self.strategy_name}: {e}")
            self.performance_stats['error_count'] += 1

        return None

    def _check_signal_frequency(self, symbol: str) -> bool:
        """检查信号频率限制"""
        if symbol not in self.last_signals:
            return True

        last_signal_time = self.last_signals[symbol]
        time_diff = (datetime.now() - last_signal_time).total_seconds() / 60  # 分钟

        return time_diff >= self.parameters.min_signal_interval

    def _preprocess_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """数据预处理"""
        if data is None or len(data) == 0:
            return None

        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error("数据缺少必要的列")
            return None

        # 数据清理
        processed_data = data.copy()

        # 移除无效数据
        processed_data = processed_data.dropna()

        # 确保数据按时间排序
        if 'timestamp' in processed_data.columns:
            processed_data = processed_data.sort_values('timestamp')

        return processed_data

    def _validate_signal(self, signal: TradingSignal) -> bool:
        """验证信号"""
        # 基础验证
        if signal.confidence < self.parameters.min_confidence:
            return False

        # 价格验证
        if signal.current_price <= 0:
            return False

        # 仓位大小验证
        if (signal.position_size_ratio and
            signal.position_size_ratio > self.parameters.max_position_size):
            signal.position_size_ratio = self.parameters.max_position_size

        # 自定义验证
        return self._validate_custom_signal(signal)

    def update_performance(self, signal_id: str, actual_return: float,
                         holding_time_hours: float = 0):
        """更新策略性能"""
        # 更新基础统计
        if actual_return > 0:
            self.performance.successful_signals += 1
        else:
            self.performance.failed_signals += 1

        # 更新收益统计
        self.performance.total_return += actual_return
        self.performance.average_return_per_signal = (
            self.performance.total_return / max(self.performance.total_signals, 1)
        )

        # 更新最佳/最差收益
        if actual_return > self.performance.best_signal_return:
            self.performance.best_signal_return = actual_return

        if actual_return < self.performance.worst_signal_return:
            self.performance.worst_signal_return = actual_return

        # 更新胜率
        if self.performance.total_signals > 0:
            self.performance.win_rate = (
                self.performance.successful_signals / self.performance.total_signals
            )

        # 更新持仓时间
        if holding_time_hours > 0:
            total_time = (self.performance.average_holding_time *
                         (self.performance.total_signals - 1) + holding_time_hours)
            self.performance.average_holding_time = total_time / self.performance.total_signals

        logger.debug(f"更新策略性能: {signal_id}, 收益: {actual_return:.4f}")

    def get_recent_signals(self, limit: int = 10) -> List[TradingSignal]:
        """获取最近的信号"""
        return sorted(self.signal_history, key=lambda s: s.timestamp, reverse=True)[:limit]

    def get_signal_history(self, symbol: str = None,
                          days_back: int = 30) -> List[TradingSignal]:
        """获取信号历史"""
        cutoff_time = datetime.now() - timedelta(days=days_back)

        filtered_signals = [
            signal for signal in self.signal_history
            if signal.timestamp >= cutoff_time
        ]

        if symbol:
            filtered_signals = [
                signal for signal in filtered_signals
                if signal.symbol.upper() == symbol.upper()
            ]

        return sorted(filtered_signals, key=lambda s: s.timestamp, reverse=True)

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'strategy_name': self.strategy_name,
            'strategy_version': self.strategy_version,
            'is_active': self.is_active,
            'last_update': self.last_update.isoformat(),
            'parameters': self.parameters.to_dict(),
            'performance': self.performance.to_dict(),
            'recent_signals_count': len(self.get_recent_signals()),
            'plugin_stats': self.get_performance_stats()
        }

    # 抽象方法 - 子类必须实现
    @abstractmethod

    def _generate_signal_logic(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成信号的核心逻辑（子类必须实现）"""
        pass

    # 可选重写的方法

    def _on_initialize(self) -> bool:
        """自定义初始化逻辑"""
        return True

    def _on_cleanup(self):
        """自定义清理逻辑"""
        pass

    def _validate_custom_parameters(self) -> bool:
        """自定义参数验证"""
        return True

    def _validate_custom_signal(self, signal: TradingSignal) -> bool:
        """自定义信号验证"""
        return True

    # 实用工具方法

    def calculate_stop_loss(self, current_price: float, signal_type: SignalType) -> float:
        """计算止损价格"""
        if signal_type == SignalType.BUY:
            return current_price * (1 - self.parameters.stop_loss_pct)
        elif signal_type == SignalType.SELL:
            return current_price * (1 + self.parameters.stop_loss_pct)
        else:
            return current_price

    def calculate_take_profit(self, current_price: float, signal_type: SignalType) -> float:
        """计算止盈价格"""
        if signal_type == SignalType.BUY:
            return current_price * (1 + self.parameters.take_profit_pct)
        elif signal_type == SignalType.SELL:
            return current_price * (1 - self.parameters.take_profit_pct)
        else:
            return current_price

    def calculate_position_size(self, current_price: float, account_balance: float,
                              risk_level: float = 1.0) -> float:
        """计算建议仓位大小"""
        max_risk_amount = account_balance * self.parameters.max_position_size * risk_level
        shares = max_risk_amount / current_price
        return shares

    def create_signal(self, symbol: str, signal_type: SignalType,
                     strength: SignalStrength, confidence: float,
                     current_price: float, reason: str = "",
                     **kwargs) -> TradingSignal:
        """创建交易信号"""
        signal_id = f"{self.strategy_name}_{symbol}_{int(datetime.now().timestamp())}"

        # 计算止损止盈
        stop_loss = self.calculate_stop_loss(current_price, signal_type)
        take_profit = self.calculate_take_profit(current_price, signal_type)

        signal = TradingSignal(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            strategy_id=self.plugin_id,
            strategy_name=self.strategy_name,
            position_size_ratio=self.parameters.max_position_size,
            **kwargs
        )

        return signal

# 示例策略插件

class ExampleMovingAverageStrategy(StrategyPluginBase):
    """示例移动平均策略"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.strategy_name = "MovingAverageStrategy"
        self.strategy_description = "基于移动平均线的简单策略"
        self.author = "System"

        # 策略特有参数
        self.fast_period = self.parameters.custom_params.get('fast_period', 10)
        self.slow_period = self.parameters.custom_params.get('slow_period', 30)

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式"""
        schema = super().get_config_schema()

        # 添加策略特有配置
        schema.update({
            'fast_period': {
                'type': 'integer',
                'description': '快速移动平均期',
                'default': 10,
                'minimum': 5,
                'maximum': 50
            },
            'slow_period': {
                'type': 'integer',
                'description': '慢速移动平均期',
                'default': 30,
                'minimum': 10,
                'maximum': 200
            }
        })

        return schema

    def _validate_custom_parameters(self) -> bool:
        """自定义参数验证"""
        if self.fast_period >= self.slow_period:
            logger.error("快速移动平均期必须小于慢速移动平均期")
            return False

        return True

    def _generate_signal_logic(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """生成信号的核心逻辑"""
        if len(data) < self.slow_period:
            return None

        # 计算移动平均线
        data['sma_fast'] = data['close'].rolling(window=self.fast_period).mean()
        data['sma_slow'] = data['close'].rolling(window=self.slow_period).mean()

        # 获取最新数据
        latest = data.iloc[-1]
        previous = data.iloc[-2]

        current_price = latest['close']
        fast_ma = latest['sma_fast']
        slow_ma = latest['sma_slow']
        prev_fast_ma = previous['sma_fast']
        prev_slow_ma = previous['sma_slow']

        # 金叉：快速MA向上穿越慢速MA
        if (prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma):
            confidence = min(0.8, abs(fast_ma - slow_ma) / slow_ma * 10)

            return self.create_signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                current_price=current_price,
                reason=f"金叉：快速MA({self.fast_period})穿越慢速MA({self.slow_period})"
            )

        # 死叉：快速MA向下穿越慢速MA
        elif (prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma):
            confidence = min(0.8, abs(fast_ma - slow_ma) / slow_ma * 10)

            return self.create_signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=SignalStrength.MODERATE,
                confidence=confidence,
                current_price=current_price,
                reason=f"死叉：快速MA({self.fast_period})跌破慢速MA({self.slow_period})"
            )

        return None
