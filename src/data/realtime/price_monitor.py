"""
Price Monitor
价格监控系统，实时监控加密货币价格变动
提供价格预警、异常检测和趋势分析功能
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import statistics
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class AlertType(Enum):
    """告警类型枚举"""
    PRICE_THRESHOLD = "price_threshold"  # 价格阈值
    PRICE_CHANGE = "price_change"        # 价格变化
    VOLUME_SPIKE = "volume_spike"        # 成交量异常
    VOLATILITY_HIGH = "volatility_high"  # 波动率过高
    MARKET_ANOMALY = "market_anomaly"    # 市场异常
    TECHNICAL_SIGNAL = "technical_signal" # 技术信号

class TrendDirection(Enum):
    """趋势方向"""
    BULLISH = "bullish"      # 看涨
    BEARISH = "bearish"      # 看跌
    SIDEWAYS = "sideways"    # 横盘
    VOLATILE = "volatile"    # 震荡

@dataclass

class PriceData:
    """价格数据结构"""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    source: str

    # 扩展数据
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_24h_pct: Optional[float] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'change_24h': self.change_24h,
            'change_24h_pct': self.change_24h_pct
        }

@dataclass

class PriceAlert:
    """价格告警"""
    alert_id: str
    symbol: str
    alert_type: AlertType
    message: str
    triggered_price: float
    timestamp: datetime = field(default_factory=datetime.now)

    # 告警条件
    threshold_value: Optional[float] = None
    comparison_operator: Optional[str] = None  # >, <, >=, <=

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'symbol': self.symbol,
            'alert_type': self.alert_type.value,
            'message': self.message,
            'triggered_price': self.triggered_price,
            'timestamp': self.timestamp.isoformat(),
            'threshold_value': self.threshold_value,
            'comparison_operator': self.comparison_operator,
            'metadata': self.metadata
        }

@dataclass

class PriceRule:
    """价格监控规则"""
    rule_id: str
    symbol: str
    alert_type: AlertType

    # 阈值条件
    threshold_value: Optional[float] = None
    comparison_operator: str = ">"  # >, <, >=, <=

    # 变化条件
    change_threshold_pct: Optional[float] = None  # 百分比变化
    change_timeframe_minutes: int = 5  # 时间窗口

    # 成交量条件
    volume_multiplier: Optional[float] = None  # 成交量倍数

    # 规则状态
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 15  # 冷却时间

    def should_trigger(self, current_price: float, price_history: List[PriceData]) -> bool:
        """检查是否应该触发"""
        if not self.enabled:
            return False

        # 检查冷却时间
        if self.last_triggered:
            cooldown_elapsed = (datetime.now() - self.last_triggered).total_seconds() / 60
            if cooldown_elapsed < self.cooldown_minutes:
                return False

        # 价格阈值检查
        if self.threshold_value is not None:
            if self.comparison_operator == ">" and current_price <= self.threshold_value:
                return False
            elif self.comparison_operator == "<" and current_price >= self.threshold_value:
                return False
            elif self.comparison_operator == ">=" and current_price < self.threshold_value:
                return False
            elif self.comparison_operator == "<=" and current_price > self.threshold_value:
                return False

        # 价格变化检查
        if self.change_threshold_pct is not None and price_history:
            cutoff_time = datetime.now() - timedelta(minutes=self.change_timeframe_minutes)
            recent_prices = [p for p in price_history if p.timestamp >= cutoff_time]

            if len(recent_prices) >= 2:
                old_price = recent_prices[0].price
                price_change_pct = (current_price - old_price) / old_price * 100

                if abs(price_change_pct) < abs(self.change_threshold_pct):
                    return False

        return True

class TechnicalAnalyzer:
    """技术分析器"""

    @staticmethod

    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """计算简单移动平均"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod

    def calculate_volatility(prices: List[float], period: int = 20) -> Optional[float]:
        """计算价格波动率"""
        if len(prices) < period:
            return None

        recent_prices = prices[-period:]
        if len(recent_prices) < 2:
            return None

        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(ret)

        if not returns:
            return None

        return statistics.stdev(returns) * (252 ** 0.5)  # 年化波动率

    @staticmethod

    def detect_trend(prices: List[float], period: int = 20) -> TrendDirection:
        """检测价格趋势"""
        if len(prices) < period:
            return TrendDirection.SIDEWAYS

        recent_prices = prices[-period:]

        # 计算价格变化率
        price_changes = []
        for i in range(1, len(recent_prices)):
            change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            price_changes.append(change)

        if not price_changes:
            return TrendDirection.SIDEWAYS

        avg_change = sum(price_changes) / len(price_changes)
        volatility = statistics.stdev(price_changes) if len(price_changes) > 1 else 0

        # 趋势判断
        if volatility > 0.02:  # 2%以上波动率
            return TrendDirection.VOLATILE
        elif avg_change > 0.001:  # 0.1%以上平均涨幅
            return TrendDirection.BULLISH
        elif avg_change < -0.001:  # 0.1%以上平均跌幅
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS

    @staticmethod

    def detect_support_resistance(prices: List[float], period: int = 20) -> Dict[str, Optional[float]]:
        """检测支撑和阻力位"""
        if len(prices) < period:
            return {'support': None, 'resistance': None}

        recent_prices = prices[-period:]

        # 简化的支撑阻力计算
        support = min(recent_prices)
        resistance = max(recent_prices)

        return {
            'support': support,
            'resistance': resistance
        }

class SymbolMonitor:
    """单个交易对监控器"""

    def __init__(self, symbol: str, max_history: int = 1000):
        self.symbol = symbol
        self.max_history = max_history

        # 价格历史
        self.price_history = deque(maxlen=max_history)
        self.volume_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)

        # 监控规则
        self.rules: Dict[str, PriceRule] = {}

        # 当前状态
        self.current_price = 0.0
        self.current_volume = 0.0
        self.last_update = None

        # 技术指标缓存
        self.indicators_cache = {}
        self.indicators_last_update = None

        # 统计信息
        self.stats = {
            'data_points': 0,
            'alerts_triggered': 0,
            'last_alert_time': None,
            'start_time': datetime.now()
        }

    def add_rule(self, rule: PriceRule):
        """添加监控规则"""
        self.rules[rule.rule_id] = rule
        logger.debug(f"添加监控规则 {self.symbol}: {rule.rule_id}")

    def remove_rule(self, rule_id: str):
        """移除监控规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.debug(f"移除监控规则 {self.symbol}: {rule_id}")

    def update_price(self, price_data: PriceData):
        """更新价格数据"""
        self.price_history.append(price_data.price)
        self.volume_history.append(price_data.volume)
        self.timestamp_history.append(price_data.timestamp)

        self.current_price = price_data.price
        self.current_volume = price_data.volume
        self.last_update = price_data.timestamp

        self.stats['data_points'] += 1

        # 清除过期的技术指标缓存
        if (self.indicators_last_update is None or
            (datetime.now() - self.indicators_last_update).total_seconds() > 60):
            self.indicators_cache.clear()

    def check_alerts(self) -> List[PriceAlert]:
        """检查告警条件"""
        alerts = []

        if not self.price_history:
            return alerts

        price_data_list = []
        for i in range(len(self.price_history)):
            price_data_list.append(PriceData(
                symbol=self.symbol,
                price=self.price_history[i],
                volume=self.volume_history[i] if i < len(self.volume_history) else 0,
                timestamp=self.timestamp_history[i] if i < len(self.timestamp_history) else datetime.now(),
                source='monitor'
            ))

        for rule in self.rules.values():
            if rule.should_trigger(self.current_price, price_data_list):
                alert = self._create_alert(rule)
                if alert:
                    alerts.append(alert)
                    rule.last_triggered = datetime.now()
                    self.stats['alerts_triggered'] += 1
                    self.stats['last_alert_time'] = datetime.now()

        return alerts

    def _create_alert(self, rule: PriceRule) -> Optional[PriceAlert]:
        """创建告警"""
        try:
            message = f"{self.symbol} "

            if rule.alert_type == AlertType.PRICE_THRESHOLD:
                message += f"价格 {rule.comparison_operator} {rule.threshold_value}, 当前价格: {self.current_price}"

            elif rule.alert_type == AlertType.PRICE_CHANGE:
                if rule.change_threshold_pct:
                    message += f"价格变化超过 {rule.change_threshold_pct}%"

            elif rule.alert_type == AlertType.VOLUME_SPIKE:
                message += f"成交量异常增长"

            alert_id = f"{self.symbol}_{rule.rule_id}_{int(time.time())}"

            return PriceAlert(
                alert_id=alert_id,
                symbol=self.symbol,
                alert_type=rule.alert_type,
                message=message,
                triggered_price=self.current_price,
                threshold_value=rule.threshold_value,
                comparison_operator=rule.comparison_operator
            )

        except Exception as e:
            logger.error(f"创建告警失败: {e}")
            return None

    def get_technical_indicators(self) -> Dict[str, Any]:
        """获取技术指标"""
        if not self.price_history:
            return {}

        # 检查缓存
        if (self.indicators_last_update and
            (datetime.now() - self.indicators_last_update).total_seconds() < 30):
            return self.indicators_cache

        prices = list(self.price_history)
        indicators = {}

        try:
            # 移动平均
            indicators['sma_5'] = TechnicalAnalyzer.calculate_sma(prices, 5)
            indicators['sma_20'] = TechnicalAnalyzer.calculate_sma(prices, 20)
            indicators['sma_50'] = TechnicalAnalyzer.calculate_sma(prices, 50)

            # 波动率
            indicators['volatility'] = TechnicalAnalyzer.calculate_volatility(prices)

            # 趋势
            indicators['trend'] = TechnicalAnalyzer.detect_trend(prices).value

            # 支撑阻力
            support_resistance = TechnicalAnalyzer.detect_support_resistance(prices)
            indicators.update(support_resistance)

            # 价格统计
            if len(prices) >= 20:
                recent_prices = prices[-20:]
                indicators['price_high_20'] = max(recent_prices)
                indicators['price_low_20'] = min(recent_prices)
                indicators['price_avg_20'] = sum(recent_prices) / len(recent_prices)

            # 缓存结果
            self.indicators_cache = indicators
            self.indicators_last_update = datetime.now()

        except Exception as e:
            logger.error(f"计算技术指标失败 {self.symbol}: {e}")

        return indicators

    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = datetime.now() - self.stats['start_time']

        return {
            'symbol': self.symbol,
            'uptime_seconds': uptime.total_seconds(),
            'current_price': self.current_price,
            'current_volume': self.current_volume,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'price_history_size': len(self.price_history),
            'rules_count': len(self.rules),
            'stats': self.stats,
            'last_alert_time': self.stats['last_alert_time'].isoformat() if self.stats['last_alert_time'] else None
        }

class PriceMonitor:
    """价格监控系统主类"""

    def __init__(self):
        self.symbol_monitors: Dict[str, SymbolMonitor] = {}
        self.alert_callbacks: List[Callable] = []

        # 任务管理
        self.monitoring_task = None
        self.cleanup_task = None
        self.is_running = False

        # 全局统计
        self.global_stats = {
            'total_symbols': 0,
            'total_rules': 0,
            'total_alerts': 0,
            'start_time': datetime.now()
        }

        logger.info("价格监控系统初始化完成")

    async def start(self):
        """启动价格监控系统"""
        if self.is_running:
            return

        self.is_running = True

        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("价格监控系统启动完成")

    async def stop(self):
        """停止价格监控系统"""
        if not self.is_running:
            return

        logger.info("正在停止价格监控系统...")
        self.is_running = False

        # 取消监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()

        if self.cleanup_task:
            self.cleanup_task.cancel()

        logger.info("价格监控系统已停止")

    def add_symbol(self, symbol: str) -> SymbolMonitor:
        """添加监控交易对"""
        if symbol not in self.symbol_monitors:
            monitor = SymbolMonitor(symbol)
            self.symbol_monitors[symbol] = monitor
            self.global_stats['total_symbols'] += 1
            logger.info(f"添加监控交易对: {symbol}")

        return self.symbol_monitors[symbol]

    def remove_symbol(self, symbol: str):
        """移除监控交易对"""
        if symbol in self.symbol_monitors:
            del self.symbol_monitors[symbol]
            self.global_stats['total_symbols'] -= 1
            logger.info(f"移除监控交易对: {symbol}")

    def get_monitor(self, symbol: str) -> Optional[SymbolMonitor]:
        """获取交易对监控器"""
        return self.symbol_monitors.get(symbol)

    def update_price(self, price_data: PriceData):
        """更新价格数据"""
        monitor = self.get_monitor(price_data.symbol)
        if monitor:
            monitor.update_price(price_data)

    def add_price_rule(self, symbol: str, rule: PriceRule):
        """添加价格监控规则"""
        monitor = self.get_monitor(symbol)
        if not monitor:
            monitor = self.add_symbol(symbol)

        monitor.add_rule(rule)
        self.global_stats['total_rules'] += 1

    def remove_price_rule(self, symbol: str, rule_id: str):
        """移除价格监控规则"""
        monitor = self.get_monitor(symbol)
        if monitor:
            monitor.remove_rule(rule_id)
            self.global_stats['total_rules'] -= 1

    def add_alert_callback(self, callback: Callable):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检查所有交易对的告警条件
                for monitor in self.symbol_monitors.values():
                    alerts = monitor.check_alerts()

                    for alert in alerts:
                        self.global_stats['total_alerts'] += 1

                        # 调用告警回调
                        for callback in self.alert_callbacks:
                            try:
                                await callback(alert)
                            except Exception as e:
                                logger.error(f"告警回调失败: {e}")

                await asyncio.sleep(1)  # 1秒检查间隔

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(1)

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                # 清理过期数据和缓存
                for monitor in self.symbol_monitors.values():
                    # 清理过期的技术指标缓存
                    if (monitor.indicators_last_update and
                        (datetime.now() - monitor.indicators_last_update).total_seconds() > 300):
                        monitor.indicators_cache.clear()
                        monitor.indicators_last_update = None

                await asyncio.sleep(300)  # 5分钟清理一次

            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(300)

    # 便利方法

    def create_price_threshold_rule(self, symbol: str, threshold: float,
                                  operator: str = ">", cooldown_minutes: int = 15) -> str:
        """创建价格阈值规则"""
        rule_id = f"price_threshold_{symbol}_{int(time.time())}"

        rule = PriceRule(
            rule_id=rule_id,
            symbol=symbol,
            alert_type=AlertType.PRICE_THRESHOLD,
            threshold_value=threshold,
            comparison_operator=operator,
            cooldown_minutes=cooldown_minutes
        )

        self.add_price_rule(symbol, rule)
        return rule_id

    def create_price_change_rule(self, symbol: str, change_pct: float,
                               timeframe_minutes: int = 5) -> str:
        """创建价格变化规则"""
        rule_id = f"price_change_{symbol}_{int(time.time())}"

        rule = PriceRule(
            rule_id=rule_id,
            symbol=symbol,
            alert_type=AlertType.PRICE_CHANGE,
            change_threshold_pct=change_pct,
            change_timeframe_minutes=timeframe_minutes
        )

        self.add_price_rule(symbol, rule)
        return rule_id

    def get_all_current_prices(self) -> Dict[str, float]:
        """获取所有当前价格"""
        return {
            symbol: monitor.current_price
            for symbol, monitor in self.symbol_monitors.items()
        }

    def get_symbol_indicators(self, symbol: str) -> Dict[str, Any]:
        """获取交易对技术指标"""
        monitor = self.get_monitor(symbol)
        return monitor.get_technical_indicators() if monitor else {}

    def get_stats(self) -> dict:
        """获取监控系统统计信息"""
        uptime = datetime.now() - self.global_stats['start_time']

        symbol_stats = {
            symbol: monitor.get_stats()
            for symbol, monitor in self.symbol_monitors.items()
        }

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime.total_seconds(),
            'global_stats': self.global_stats,
            'symbol_stats': symbol_stats,
            'alert_callbacks_count': len(self.alert_callbacks)
        }

# 全局实例
_price_monitor_instance = None

def get_price_monitor() -> PriceMonitor:
    """获取价格监控器实例"""
    global _price_monitor_instance
    if _price_monitor_instance is None:
        _price_monitor_instance = PriceMonitor()
    return _price_monitor_instance
