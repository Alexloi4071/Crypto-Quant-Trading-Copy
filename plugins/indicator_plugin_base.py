"""
Indicator Plugin Base
指标插件基类，为技术指标插件提供标准接口和计算框架
支持自定义指标计算、参数配置和结果缓存
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
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

class IndicatorType(Enum):
    """指标类型"""
    TREND = "trend"              # 趋势指标
    MOMENTUM = "momentum"        # 动量指标
    VOLATILITY = "volatility"    # 波动率指标
    VOLUME = "volume"           # 成交量指标
    SUPPORT_RESISTANCE = "support_resistance"  # 支撑阻力指标
    OSCILLATOR = "oscillator"   # 振荡器指标
    CUSTOM = "custom"           # 自定义指标

class IndicatorLevel(Enum):
    """指标级别"""
    OVERBOUGHT = "overbought"   # 超买
    OVERSOLD = "oversold"       # 超卖
    BULLISH = "bullish"         # 看涨
    BEARISH = "bearish"         # 看跌
    NEUTRAL = "neutral"         # 中性
    STRONG_BULL = "strong_bull" # 强烈看涨
    STRONG_BEAR = "strong_bear" # 强烈看跌

@dataclass

class IndicatorValue:
    """指标值"""
    timestamp: datetime
    value: Union[float, Dict[str, float], List[float]]
    level: Optional[IndicatorLevel] = None
    signal_strength: float = 0.0  # 信号强度 -1 到 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        value_data = self.value
        if isinstance(self.value, dict):
            value_data = self.value
        elif isinstance(self.value, (list, tuple)):
            value_data = list(self.value)
        else:
            value_data = float(self.value)

        return {
            'timestamp': self.timestamp.isoformat(),
            'value': value_data,
            'level': self.level.value if self.level else None,
            'signal_strength': self.signal_strength,
            'metadata': self.metadata
        }

@dataclass

class IndicatorResult:
    """指标计算结果"""
    indicator_name: str
    symbol: str
    timeframe: str
    calculation_time: datetime

    # 指标值序列
    values: List[IndicatorValue] = field(default_factory=list)

    # 当前值
    current_value: Optional[IndicatorValue] = None

    # 统计信息
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None

    # 性能指标
    calculation_duration: float = 0.0  # 计算耗时（秒）
    data_points: int = 0

    def to_dict(self) -> dict:
        return {
            'indicator_name': self.indicator_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'calculation_time': self.calculation_time.isoformat(),
            'current_value': self.current_value.to_dict() if self.current_value else None,
            'values_count': len(self.values),
            'min_value': self.min_value,
            'max_value': self.max_value,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'calculation_duration': self.calculation_duration,
            'data_points': self.data_points
        }

@dataclass

class IndicatorParameters:
    """指标参数"""
    # 通用参数
    period: int = 14
    timeframe: str = "1D"
    smoothing_factor: float = 0.1

    # 价格源
    price_source: str = "close"  # open, high, low, close, typical, weighted

    # 计算选项
    use_smoothing: bool = False
    fill_na_method: str = "forward"  # forward, backward, interpolate, drop

    # 自定义参数
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'period': self.period,
            'timeframe': self.timeframe,
            'smoothing_factor': self.smoothing_factor,
            'price_source': self.price_source,
            'use_smoothing': self.use_smoothing,
            'fill_na_method': self.fill_na_method,
            'custom_params': self.custom_params
        }

class IndicatorPluginBase(PluginBase):
    """指标插件基类"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # 指标基础信息
        self.indicator_name = self.__class__.__name__
        self.indicator_version = "1.0.0"
        self.indicator_description = "基础技术指标"
        self.indicator_type = IndicatorType.CUSTOM
        self.author = "Unknown"

        # 指标参数
        self.parameters = IndicatorParameters()
        self._load_parameters_from_config()

        # 结果缓存
        self.results_cache = {}  # (symbol, timeframe) -> IndicatorResult
        self.cache_timestamps = {}  # (symbol, timeframe) -> timestamp
        self.cache_ttl = 300  # 缓存TTL（秒）

        # 计算历史
        self.calculation_history = []

        # 状态管理
        self.is_active = False
        self.last_calculation = datetime.now()

        logger.debug(f"指标插件初始化: {self.indicator_name}")

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'id': self.plugin_id,
            'name': self.indicator_name,
            'version': self.indicator_version,
            'description': self.indicator_description,
            'type': 'indicator',
            'indicator_type': self.indicator_type.value,
            'config_schema': self.get_config_schema(),
            'default_config': self._get_default_config()
        }

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式"""
        return {
            'period': {
                'type': 'integer',
                'description': '计算周期',
                'default': 14,
                'minimum': 1,
                'maximum': 500
            },
            'timeframe': {
                'type': 'string',
                'description': '时间框架',
                'default': '1D',
                'enum': ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1D']
            },
            'price_source': {
                'type': 'string',
                'description': '价格数据源',
                'default': 'close',
                'enum': ['open', 'high', 'low', 'close', 'typical', 'weighted']
            },
            'smoothing_factor': {
                'type': 'number',
                'description': '平滑因子',
                'default': 0.1,
                'minimum': 0.0,
                'maximum': 1.0
            },
            'use_smoothing': {
                'type': 'boolean',
                'description': '是否使用平滑',
                'default': False
            }
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return self.parameters.to_dict()

    def _load_parameters_from_config(self):
        """从配置加载参数"""
        if not self.config:
            return

        # 更新指标参数
        for key, value in self.config.items():
            if hasattr(self.parameters, key):
                setattr(self.parameters, key, value)
            else:
                # 自定义参数
                self.parameters.custom_params[key] = value

    def initialize(self) -> bool:
        """初始化指标"""
        try:
            # 验证参数
            if not self._validate_parameters():
                logger.error(f"指标参数验证失败: {self.indicator_name}")
                return False

            # 执行自定义初始化
            if not self._on_initialize():
                logger.error(f"指标初始化失败: {self.indicator_name}")
                return False

            self.is_active = True
            self.last_calculation = datetime.now()

            logger.info(f"指标 {self.indicator_name} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"指标初始化异常 {self.indicator_name}: {e}")
            return False

    def cleanup(self) -> bool:
        """清理指标"""
        try:
            self.is_active = False

            # 执行自定义清理
            self._on_cleanup()

            # 清理缓存
            self.results_cache.clear()
            self.cache_timestamps.clear()

            logger.info(f"指标 {self.indicator_name} 清理完成")
            return True

        except Exception as e:
            logger.error(f"指标清理异常 {self.indicator_name}: {e}")
            return False

    def _validate_parameters(self) -> bool:
        """验证参数"""
        # 基础验证
        if self.parameters.period <= 0:
            logger.error("计算周期必须大于0")
            return False

        if self.parameters.price_source not in ['open', 'high', 'low', 'close', 'typical', 'weighted']:
            logger.error(f"不支持的价格源: {self.parameters.price_source}")
            return False

        # 自定义验证
        return self._validate_custom_parameters()

    def calculate(self, symbol: str, data: pd.DataFrame,
                 timeframe: str = None) -> Optional[IndicatorResult]:
        """计算指标（主要接口）"""
        if not self.is_active:
            return None

        timeframe = timeframe or self.parameters.timeframe
        cache_key = (symbol, timeframe)

        # 检查缓存
        if self._is_cache_valid(cache_key):
            cached_result = self.results_cache[cache_key]
            logger.debug(f"使用缓存的指标结果: {self.indicator_name} - {symbol}")
            return cached_result

        start_time = time.time()

        try:
            # 数据预处理
            processed_data = self._preprocess_data(data)
            if processed_data is None or len(processed_data) < self.parameters.period:
                return None

            # 计算指标
            result = self._calculate_indicator_logic(symbol, processed_data, timeframe)

            if result:
                # 计算统计信息
                self._calculate_statistics(result)

                # 缓存结果
                result.calculation_duration = time.time() - start_time
                self.results_cache[cache_key] = result
                self.cache_timestamps[cache_key] = datetime.now()

                # 记录计算历史
                self.calculation_history.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'calculation_time': result.calculation_time,
                    'duration': result.calculation_duration,
                    'data_points': result.data_points
                })

                # 更新性能统计
                self.performance_stats['call_count'] += 1
                self.performance_stats['total_time'] += result.calculation_duration
                self.performance_stats['avg_time'] = (
                    self.performance_stats['total_time'] /
                    self.performance_stats['call_count']
                )

                self.last_calculation = datetime.now()

                logger.debug(f"计算指标完成: {self.indicator_name} - {symbol} "
                           f"({result.calculation_duration:.4f}s)")

                return result

        except Exception as e:
            logger.error(f"计算指标异常 {self.indicator_name}: {e}")
            self.performance_stats['error_count'] += 1

        return None

    def _is_cache_valid(self, cache_key: Tuple[str, str]) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.results_cache:
            return False

        if cache_key not in self.cache_timestamps:
            return False

        cache_time = self.cache_timestamps[cache_key]
        elapsed = (datetime.now() - cache_time).total_seconds()

        return elapsed < self.cache_ttl

    def _preprocess_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """数据预处理"""
        if data is None or len(data) == 0:
            return None

        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            logger.error("数据缺少必要的OHLC列")
            return None

        processed_data = data.copy()

        # 确保数据按时间排序
        if 'timestamp' in processed_data.columns:
            processed_data = processed_data.sort_values('timestamp')
        elif processed_data.index.name in ['timestamp', 'date', 'datetime']:
            processed_data = processed_data.sort_index()

        # 处理缺失值
        if self.parameters.fill_na_method == 'forward':
            processed_data = processed_data.fillna(method='ffill')
        elif self.parameters.fill_na_method == 'backward':
            processed_data = processed_data.fillna(method='bfill')
        elif self.parameters.fill_na_method == 'interpolate':
            processed_data = processed_data.interpolate()
        elif self.parameters.fill_na_method == 'drop':
            processed_data = processed_data.dropna()

        # 添加派生价格列
        processed_data['typical'] = (processed_data['high'] +
                                    processed_data['low'] +
                                    processed_data['close']) / 3

        if 'volume' in processed_data.columns:
            processed_data['weighted'] = (processed_data['high'] * processed_data['volume'] +
                                        processed_data['low'] * processed_data['volume'] +
                                        processed_data['close'] * processed_data['volume'] * 2) / (processed_data['volume'] * 4)
        else:
            processed_data['weighted'] = processed_data['typical']

        return processed_data

    def _calculate_statistics(self, result: IndicatorResult):
        """计算统计信息"""
        if not result.values:
            return

        # 提取数值（处理复合值的情况）
        numeric_values = []
        for val in result.values:
            if isinstance(val.value, (int, float)):
                numeric_values.append(val.value)
            elif isinstance(val.value, dict) and 'main' in val.value:
                numeric_values.append(val.value['main'])
            elif isinstance(val.value, (list, tuple)) and len(val.value) > 0:
                numeric_values.append(val.value[0])

        if numeric_values:
            result.min_value = float(np.min(numeric_values))
            result.max_value = float(np.max(numeric_values))
            result.mean_value = float(np.mean(numeric_values))
            result.std_value = float(np.std(numeric_values))

    def get_price_series(self, data: pd.DataFrame) -> pd.Series:
        """获取指定的价格序列"""
        if self.parameters.price_source in data.columns:
            return data[self.parameters.price_source]
        else:
            logger.warning(f"价格源 {self.parameters.price_source} 不存在，使用收盘价")
            return data['close']

    def apply_smoothing(self, series: pd.Series) -> pd.Series:
        """应用平滑处理"""
        if not self.parameters.use_smoothing:
            return series

        # 指数移动平均平滑
        alpha = self.parameters.smoothing_factor
        return series.ewm(alpha=alpha).mean()

    def determine_level(self, value: float, **kwargs) -> IndicatorLevel:
        """判断指标级别（子类可重写）"""
        return IndicatorLevel.NEUTRAL

    def calculate_signal_strength(self, current_value: float,
                                previous_values: List[float]) -> float:
        """计算信号强度（子类可重写）"""
        return 0.0

    def get_latest_value(self, symbol: str, timeframe: str = None) -> Optional[IndicatorValue]:
        """获取最新的指标值"""
        timeframe = timeframe or self.parameters.timeframe
        cache_key = (symbol, timeframe)

        if cache_key in self.results_cache:
            result = self.results_cache[cache_key]
            return result.current_value

        return None

    def get_historical_values(self, symbol: str, timeframe: str = None,
                            limit: int = 100) -> List[IndicatorValue]:
        """获取历史指标值"""
        timeframe = timeframe or self.parameters.timeframe
        cache_key = (symbol, timeframe)

        if cache_key in self.results_cache:
            result = self.results_cache[cache_key]
            return result.values[-limit:] if result.values else []

        return []

    def clear_cache(self, symbol: str = None, timeframe: str = None):
        """清理缓存"""
        if symbol and timeframe:
            cache_key = (symbol, timeframe)
            self.results_cache.pop(cache_key, None)
            self.cache_timestamps.pop(cache_key, None)
        else:
            self.results_cache.clear()
            self.cache_timestamps.clear()

        logger.debug(f"清理指标缓存: {self.indicator_name}")

    def get_calculation_stats(self) -> Dict[str, Any]:
        """获取计算统计"""
        if not self.calculation_history:
            return {}

        recent_calcs = self.calculation_history[-100:]  # 最近100次计算
        durations = [calc['duration'] for calc in recent_calcs]

        return {
            'total_calculations': len(self.calculation_history),
            'recent_calculations': len(recent_calcs),
            'avg_calculation_time': np.mean(durations) if durations else 0,
            'min_calculation_time': np.min(durations) if durations else 0,
            'max_calculation_time': np.max(durations) if durations else 0,
            'cache_hit_ratio': self._calculate_cache_hit_ratio(),
            'last_calculation': self.last_calculation.isoformat(),
            'cache_size': len(self.results_cache)
        }

    def _calculate_cache_hit_ratio(self) -> float:
        """计算缓存命中率"""
        total_calls = self.performance_stats['call_count']
        if total_calls == 0:
            return 0.0

        # 估算缓存命中次数（总调用次数 - 实际计算次数）
        actual_calculations = len(self.calculation_history)
        cache_hits = max(0, total_calls - actual_calculations)

        return cache_hits / total_calls

    # 抽象方法 - 子类必须实现
    @abstractmethod

    def _calculate_indicator_logic(self, symbol: str, data: pd.DataFrame,
                                 timeframe: str) -> Optional[IndicatorResult]:
        """计算指标的核心逻辑（子类必须实现）"""
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

# 示例指标插件

class ExampleRSIIndicator(IndicatorPluginBase):
    """示例RSI指标"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.indicator_name = "RSI"
        self.indicator_description = "相对强弱指数"
        self.indicator_type = IndicatorType.MOMENTUM
        self.author = "System"

    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置模式"""
        schema = super().get_config_schema()

        # RSI特有配置
        schema.update({
            'overbought_threshold': {
                'type': 'number',
                'description': '超买阈值',
                'default': 70,
                'minimum': 50,
                'maximum': 90
            },
            'oversold_threshold': {
                'type': 'number',
                'description': '超卖阈值',
                'default': 30,
                'minimum': 10,
                'maximum': 50
            }
        })

        return schema

    def determine_level(self, value: float, **kwargs) -> IndicatorLevel:
        """判断RSI级别"""
        overbought = self.parameters.custom_params.get('overbought_threshold', 70)
        oversold = self.parameters.custom_params.get('oversold_threshold', 30)

        if value >= overbought:
            return IndicatorLevel.OVERBOUGHT
        elif value <= oversold:
            return IndicatorLevel.OVERSOLD
        elif value > 60:
            return IndicatorLevel.BULLISH
        elif value < 40:
            return IndicatorLevel.BEARISH
        else:
            return IndicatorLevel.NEUTRAL

    def calculate_signal_strength(self, current_value: float,
                                previous_values: List[float]) -> float:
        """计算RSI信号强度"""
        overbought = self.parameters.custom_params.get('overbought_threshold', 70)
        oversold = self.parameters.custom_params.get('oversold_threshold', 30)

        # 极端值产生更强的信号
        if current_value >= overbought:
            return -(current_value - overbought) / (100 - overbought)  # 负值表示看跌
        elif current_value <= oversold:
            return (oversold - current_value) / oversold  # 正值表示看涨
        else:
            return 0.0

    def _calculate_indicator_logic(self, symbol: str, data: pd.DataFrame,
                                 timeframe: str) -> Optional[IndicatorResult]:
        """计算RSI"""
        if len(data) < self.parameters.period:
            return None

        # 获取价格序列
        prices = self.get_price_series(data)

        # 计算价格变化
        delta = prices.diff()

        # 分离涨跌
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # 计算平均涨跌幅
        avg_gains = gains.rolling(window=self.parameters.period).mean()
        avg_losses = losses.rolling(window=self.parameters.period).mean()

        # 计算RS和RSI
        rs = avg_gains / (avg_losses + 1e-10)  # 避免除零
        rsi = 100 - (100 / (1 + rs))

        # 应用平滑处理
        if self.parameters.use_smoothing:
            rsi = self.apply_smoothing(rsi)

        # 创建结果
        result = IndicatorResult(
            indicator_name=self.indicator_name,
            symbol=symbol,
            timeframe=timeframe,
            calculation_time=datetime.now(),
            data_points=len(data)
        )

        # 填充指标值
        valid_indices = ~rsi.isna()
        timestamps = data.index[valid_indices] if hasattr(data.index, 'to_pydatetime') else range(len(data))

        for i, (idx, rsi_val) in enumerate(zip(timestamps, rsi[valid_indices])):
            timestamp = idx if isinstance(idx, datetime) else datetime.now() - timedelta(days=len(data)-i)

            # 获取历史值用于信号强度计算
            hist_values = rsi[max(0, len(rsi)-10):len(rsi)-1].tolist()

            indicator_value = IndicatorValue(
                timestamp=timestamp,
                value=float(rsi_val),
                level=self.determine_level(rsi_val),
                signal_strength=self.calculate_signal_strength(rsi_val, hist_values)
            )

            result.values.append(indicator_value)

        # 设置当前值
        if result.values:
            result.current_value = result.values[-1]

        return result
