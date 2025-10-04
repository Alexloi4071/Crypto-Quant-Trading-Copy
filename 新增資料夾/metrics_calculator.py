"""
Realtime Metrics Calculator
实时指标计算器，提供高性能的实时指标计算和聚合功能
支持技术指标、统计指标和自定义计算
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import statistics
import math
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

@dataclass
class MetricWindow:
    """指标窗口数据结构"""
    name: str
    window_size: int
    data: deque = field(default_factory=deque)
    timestamps: deque = field(default_factory=deque)
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        self.data = deque(maxlen=self.window_size)
        self.timestamps = deque(maxlen=self.window_size)
    
    def add_value(self, value: float, timestamp: datetime = None):
        """添加值到窗口"""
        timestamp = timestamp or datetime.now()
        self.data.append(value)
        self.timestamps.append(timestamp)
        self.last_update = timestamp
    
    def get_values(self) -> List[float]:
        """获取所有值"""
        return list(self.data)
    
    def get_recent_values(self, count: int) -> List[float]:
        """获取最近的N个值"""
        return list(self.data)[-count:] if count <= len(self.data) else list(self.data)
    
    def is_full(self) -> bool:
        """检查窗口是否已满"""
        return len(self.data) >= self.window_size
    
    def clear(self):
        """清空窗口"""
        self.data.clear()
        self.timestamps.clear()

@dataclass
class CalculatedMetric:
    """计算指标结果"""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

class TechnicalIndicators:
    """技术指标计算器"""
    
    @staticmethod
    def sma(values: List[float], period: int) -> Optional[float]:
        """简单移动平均"""
        if len(values) < period:
            return None
        return sum(values[-period:]) / period
    
    @staticmethod
    def ema(values: List[float], period: int, alpha: float = None) -> Optional[float]:
        """指数移动平均"""
        if len(values) < period:
            return None
        
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        ema_value = values[0]
        for value in values[1:]:
            ema_value = alpha * value + (1 - alpha) * ema_value
        
        return ema_value
    
    @staticmethod
    def rsi(values: List[float], period: int = 14) -> Optional[float]:
        """相对强弱指数"""
        if len(values) < period + 1:
            return None
        
        deltas = [values[i] - values[i-1] for i in range(1, len(values))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(values: List[float], period: int = 20, std_dev: float = 2) -> Optional[Tuple[float, float, float]]:
        """布林带"""
        if len(values) < period:
            return None
        
        recent_values = values[-period:]
        sma = sum(recent_values) / period
        variance = sum((x - sma) ** 2 for x in recent_values) / period
        std = math.sqrt(variance)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def macd(values: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Tuple[float, float, float]]:
        """MACD指标"""
        if len(values) < slow_period:
            return None
        
        fast_ema = TechnicalIndicators.ema(values, fast_period)
        slow_ema = TechnicalIndicators.ema(values, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        macd_line = fast_ema - slow_ema
        
        # 简化的信号线计算
        signal_line = macd_line  # 实际应该用EMA
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high_values: List[float], low_values: List[float], close_values: List[float], period: int = 14) -> Optional[float]:
        """随机振荡器"""
        if len(high_values) < period or len(low_values) < period or len(close_values) < period:
            return None
        
        recent_highs = high_values[-period:]
        recent_lows = low_values[-period:]
        current_close = close_values[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            return 50
        
        k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        return k_percent

class StatisticalIndicators:
    """统计指标计算器"""
    
    @staticmethod
    def mean(values: List[float]) -> float:
        """均值"""
        return statistics.mean(values) if values else 0
    
    @staticmethod
    def median(values: List[float]) -> float:
        """中位数"""
        return statistics.median(values) if values else 0
    
    @staticmethod
    def std_dev(values: List[float]) -> float:
        """标准差"""
        return statistics.stdev(values) if len(values) > 1 else 0
    
    @staticmethod
    def variance(values: List[float]) -> float:
        """方差"""
        return statistics.variance(values) if len(values) > 1 else 0
    
    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """百分位数"""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * (p / 100)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    @staticmethod
    def skewness(values: List[float]) -> float:
        """偏度"""
        if len(values) < 3:
            return 0
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return 0
        
        n = len(values)
        skew = sum(((x - mean) / std) ** 3 for x in values) * n / ((n - 1) * (n - 2))
        
        return skew
    
    @staticmethod
    def kurtosis(values: List[float]) -> float:
        """峰度"""
        if len(values) < 4:
            return 0
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        
        if std == 0:
            return 0
        
        n = len(values)
        kurt = sum(((x - mean) / std) ** 4 for x in values) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        
        return kurt
    
    @staticmethod
    def correlation(x_values: List[float], y_values: List[float]) -> float:
        """相关系数"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0
        
        try:
            return statistics.correlation(x_values, y_values)
        except:
            # 如果Python版本不支持correlation函数
            n = len(x_values)
            mean_x = statistics.mean(x_values)
            mean_y = statistics.mean(y_values)
            
            numerator = sum((x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
            sum_sq_x = sum((x - mean_x) ** 2 for x in x_values)
            sum_sq_y = sum((y - mean_y) ** 2 for y in y_values)
            
            if sum_sq_x == 0 or sum_sq_y == 0:
                return 0
            
            return numerator / math.sqrt(sum_sq_x * sum_sq_y)

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, calculator_id: str, config: dict = None):
        self.calculator_id = calculator_id
        self.config = config or {}
        
        # 数据窗口
        self.windows = {}  # window_name -> MetricWindow
        
        # 计算函数注册表
        self.calculators = {}  # metric_name -> calculator_function
        
        # 计算结果缓存
        self.results = {}  # metric_name -> CalculatedMetric
        self.result_history = defaultdict(lambda: deque(maxlen=1000))
        
        # 统计信息
        self.stats = {
            'calculations_performed': 0,
            'calculation_errors': 0,
            'total_calculation_time': 0.0,
            'start_time': datetime.now()
        }
        
        # 注册默认计算器
        self._register_default_calculators()
        
        logger.debug(f"指标计算器初始化: {calculator_id}")
    
    def _register_default_calculators(self):
        """注册默认计算器"""
        # 技术指标
        self.register_calculator('sma_5', lambda w: TechnicalIndicators.sma(w['price'].get_values(), 5))
        self.register_calculator('sma_10', lambda w: TechnicalIndicators.sma(w['price'].get_values(), 10))
        self.register_calculator('sma_20', lambda w: TechnicalIndicators.sma(w['price'].get_values(), 20))
        
        self.register_calculator('ema_12', lambda w: TechnicalIndicators.ema(w['price'].get_values(), 12))
        self.register_calculator('ema_26', lambda w: TechnicalIndicators.ema(w['price'].get_values(), 26))
        
        self.register_calculator('rsi_14', lambda w: TechnicalIndicators.rsi(w['price'].get_values(), 14))
        
        # 统计指标
        self.register_calculator('price_mean', lambda w: StatisticalIndicators.mean(w['price'].get_values()))
        self.register_calculator('price_std', lambda w: StatisticalIndicators.std_dev(w['price'].get_values()))
        self.register_calculator('price_p95', lambda w: StatisticalIndicators.percentile(w['price'].get_values(), 95))
        
        # 成交量相关
        self.register_calculator('volume_mean', lambda w: StatisticalIndicators.mean(w['volume'].get_values()))
        self.register_calculator('volume_std', lambda w: StatisticalIndicators.std_dev(w['volume'].get_values()))
        
        # 价格-成交量相关性
        self.register_calculator('price_volume_corr', lambda w: StatisticalIndicators.correlation(
            w['price'].get_values(), w['volume'].get_values()
        ))
    
    def create_window(self, name: str, size: int) -> MetricWindow:
        """创建数据窗口"""
        window = MetricWindow(name, size)
        self.windows[name] = window
        logger.debug(f"创建数据窗口: {name} (size={size})")
        return window
    
    def get_window(self, name: str) -> Optional[MetricWindow]:
        """获取数据窗口"""
        return self.windows.get(name)
    
    def register_calculator(self, metric_name: str, calculator_func: Callable):
        """注册计算函数"""
        self.calculators[metric_name] = calculator_func
        logger.debug(f"注册计算器: {metric_name}")
    
    def add_data_point(self, window_name: str, value: float, timestamp: datetime = None):
        """添加数据点"""
        window = self.windows.get(window_name)
        if window:
            window.add_value(value, timestamp)
            
            # 触发相关指标重新计算
            asyncio.create_task(self.calculate_metrics())
    
    def add_multiple_data_points(self, data_points: Dict[str, float], timestamp: datetime = None):
        """添加多个数据点"""
        for window_name, value in data_points.items():
            self.add_data_point(window_name, value, timestamp)
    
    async def calculate_metrics(self, metric_names: List[str] = None) -> Dict[str, CalculatedMetric]:
        """计算指标"""
        start_time = asyncio.get_event_loop().time()
        calculated = {}
        
        # 确定要计算的指标
        if metric_names is None:
            metric_names = list(self.calculators.keys())
        
        for metric_name in metric_names:
            calculator = self.calculators.get(metric_name)
            if not calculator:
                continue
            
            try:
                # 执行计算
                result = calculator(self.windows)
                
                if result is not None:
                    calculated_metric = CalculatedMetric(
                        name=metric_name,
                        value=float(result),
                        timestamp=datetime.now(),
                        metadata={'calculator_id': self.calculator_id}
                    )
                    
                    calculated[metric_name] = calculated_metric
                    self.results[metric_name] = calculated_metric
                    self.result_history[metric_name].append(calculated_metric)
                
                self.stats['calculations_performed'] += 1
                
            except Exception as e:
                logger.error(f"计算指标 {metric_name} 失败: {e}")
                self.stats['calculation_errors'] += 1
        
        # 更新统计
        calculation_time = asyncio.get_event_loop().time() - start_time
        self.stats['total_calculation_time'] += calculation_time
        
        return calculated
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """获取指标值"""
        metric = self.results.get(metric_name)
        return metric.value if metric else None
    
    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[CalculatedMetric]:
        """获取指标历史"""
        history = self.result_history.get(metric_name, deque())
        return list(history)[-limit:]
    
    def get_all_current_metrics(self) -> Dict[str, CalculatedMetric]:
        """获取所有当前指标"""
        return self.results.copy()
    
    def clear_metric_history(self, metric_name: str = None):
        """清空指标历史"""
        if metric_name:
            self.result_history[metric_name].clear()
        else:
            self.result_history.clear()
    
    def get_stats(self) -> dict:
        """获取计算器统计信息"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        avg_calculation_time = (self.stats['total_calculation_time'] / 
                               max(self.stats['calculations_performed'], 1))
        
        success_rate = ((self.stats['calculations_performed'] - self.stats['calculation_errors']) / 
                       max(self.stats['calculations_performed'], 1))
        
        return {
            'calculator_id': self.calculator_id,
            'uptime_seconds': uptime,
            'window_count': len(self.windows),
            'calculator_count': len(self.calculators),
            'current_metrics_count': len(self.results),
            'calculations_performed': self.stats['calculations_performed'],
            'calculation_errors': self.stats['calculation_errors'],
            'success_rate': success_rate,
            'avg_calculation_time': avg_calculation_time,
            'window_stats': {
                name: {
                    'size': len(window.data),
                    'max_size': window.window_size,
                    'last_update': window.last_update.isoformat() if window.last_update else None
                }
                for name, window in self.windows.items()
            }
        }

class RealtimeMetricsCalculator:
    """实时指标计算器主类"""
    
    def __init__(self):
        self.calculators = {}  # calculator_id -> MetricsCalculator
        
        # 自动计算配置
        self.auto_calculation_enabled = True
        self.calculation_interval = 1.0  # 秒
        
        # 任务管理
        self.calculation_task = None
        self.is_running = False
        
        # 全局统计
        self.global_stats = {
            'total_calculators': 0,
            'total_calculations': 0,
            'total_calculation_time': 0.0,
            'start_time': datetime.now()
        }
        
        logger.info("实时指标计算器主系统初始化完成")
    
    async def start(self):
        """启动计算器系统"""
        if self.is_running:
            return
        
        self.is_running = True
        
        if self.auto_calculation_enabled:
            self.calculation_task = asyncio.create_task(self._auto_calculation_loop())
        
        logger.info("实时指标计算器系统启动完成")
    
    async def stop(self):
        """停止计算器系统"""
        if not self.is_running:
            return
        
        logger.info("正在停止实时指标计算器系统...")
        self.is_running = False
        
        if self.calculation_task:
            self.calculation_task.cancel()
        
        logger.info("实时指标计算器系统已停止")
    
    def create_calculator(self, calculator_id: str, config: dict = None) -> MetricsCalculator:
        """创建指标计算器"""
        if calculator_id in self.calculators:
            return self.calculators[calculator_id]
        
        calculator = MetricsCalculator(calculator_id, config)
        self.calculators[calculator_id] = calculator
        self.global_stats['total_calculators'] += 1
        
        logger.info(f"创建指标计算器: {calculator_id}")
        return calculator
    
    def get_calculator(self, calculator_id: str) -> Optional[MetricsCalculator]:
        """获取指标计算器"""
        return self.calculators.get(calculator_id)
    
    def remove_calculator(self, calculator_id: str):
        """移除指标计算器"""
        if calculator_id in self.calculators:
            del self.calculators[calculator_id]
            self.global_stats['total_calculators'] -= 1
            logger.info(f"移除指标计算器: {calculator_id}")
    
    async def _auto_calculation_loop(self):
        """自动计算循环"""
        while self.is_running:
            try:
                start_time = asyncio.get_event_loop().time()
                
                # 触发所有计算器计算
                calculation_tasks = []
                for calculator in self.calculators.values():
                    task = calculator.calculate_metrics()
                    calculation_tasks.append(task)
                
                if calculation_tasks:
                    await asyncio.gather(*calculation_tasks, return_exceptions=True)
                
                # 更新统计
                calculation_time = asyncio.get_event_loop().time() - start_time
                self.global_stats['total_calculation_time'] += calculation_time
                self.global_stats['total_calculations'] += len(calculation_tasks)
                
                await asyncio.sleep(self.calculation_interval)
                
            except Exception as e:
                logger.error(f"自动计算循环错误: {e}")
                await asyncio.sleep(1)
    
    # 便利方法
    async def add_price_data(self, calculator_id: str, symbol: str, price: float, 
                           volume: float = None, timestamp: datetime = None):
        """添加价格数据"""
        calculator = self.get_calculator(calculator_id)
        if not calculator:
            calculator = self.create_calculator(calculator_id)
        
        # 确保窗口存在
        if 'price' not in calculator.windows:
            calculator.create_window('price', 200)
        
        if volume is not None and 'volume' not in calculator.windows:
            calculator.create_window('volume', 200)
        
        # 添加数据
        data_points = {'price': price}
        if volume is not None:
            data_points['volume'] = volume
        
        calculator.add_multiple_data_points(data_points, timestamp)
    
    async def calculate_symbol_metrics(self, symbol: str) -> Dict[str, Any]:
        """计算特定交易对的所有指标"""
        calculator = self.get_calculator(symbol)
        if not calculator:
            return {}
        
        return await calculator.calculate_metrics()
    
    def get_symbol_metric_value(self, symbol: str, metric_name: str) -> Optional[float]:
        """获取特定交易对的指标值"""
        calculator = self.get_calculator(symbol)
        if not calculator:
            return None
        
        return calculator.get_metric_value(metric_name)
    
    def get_all_symbol_metrics(self, symbol: str) -> Dict[str, float]:
        """获取特定交易对的所有指标值"""
        calculator = self.get_calculator(symbol)
        if not calculator:
            return {}
        
        current_metrics = calculator.get_all_current_metrics()
        return {name: metric.value for name, metric in current_metrics.items()}
    
    def get_stats(self) -> dict:
        """获取系统统计信息"""
        uptime = (datetime.now() - self.global_stats['start_time']).total_seconds()
        
        calculator_stats = {}
        total_calculations = 0
        total_errors = 0
        
        for calc_id, calculator in self.calculators.items():
            stats = calculator.get_stats()
            calculator_stats[calc_id] = stats
            total_calculations += stats['calculations_performed']
            total_errors += stats['calculation_errors']
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'auto_calculation_enabled': self.auto_calculation_enabled,
            'calculation_interval': self.calculation_interval,
            'global_stats': {
                **self.global_stats,
                'success_rate': (total_calculations - total_errors) / max(total_calculations, 1)
            },
            'calculator_stats': calculator_stats
        }

# 全局实例
_metrics_calculator_instance = None

def get_realtime_metrics_calculator() -> RealtimeMetricsCalculator:
    """获取实时指标计算器实例"""
    global _metrics_calculator_instance
    if _metrics_calculator_instance is None:
        _metrics_calculator_instance = RealtimeMetricsCalculator()
    return _metrics_calculator_instance