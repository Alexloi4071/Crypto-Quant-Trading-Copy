"""
Meta Strategy
元策略系统，基于策略组合和动态选择的高级交易策略框架
支持策略自适应、性能评估和风险管理的智能策略调度
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class StrategyType(Enum):
    """策略类型"""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    MOMENTUM = "momentum"                    # 动量策略
    ARBITRAGE = "arbitrage"                  # 套利策略
    PAIRS_TRADING = "pairs_trading"          # 配对交易
    MARKET_MAKING = "market_making"          # 做市策略
    VOLATILITY = "volatility"                # 波动率策略
    FUNDAMENTAL = "fundamental"              # 基本面策略
    SENTIMENT = "sentiment"                  # 情绪策略
    MULTI_FACTOR = "multi_factor"           # 多因子策略

class SelectionMethod(Enum):
    """选择方法"""
    PERFORMANCE_BASED = "performance_based"  # 基于性能
    RISK_ADJUSTED = "risk_adjusted"         # 风险调整
    ENSEMBLE = "ensemble"                   # 集成方法
    ADAPTIVE = "adaptive"                   # 自适应选择
    ROTATION = "rotation"                   # 轮换策略
    THRESHOLD = "threshold"                 # 阈值选择
    MACHINE_LEARNING = "machine_learning"   # 机器学习选择

class MarketRegime(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"             # 上升趋势
    TRENDING_DOWN = "trending_down"         # 下降趋势
    SIDEWAYS = "sideways"                   # 横盘整理
    HIGH_VOLATILITY = "high_volatility"     # 高波动
    LOW_VOLATILITY = "low_volatility"       # 低波动
    CRISIS = "crisis"                       # 危机状态
    RECOVERY = "recovery"                   # 复苏状态

@dataclass

class StrategySignal:
    """策略信号"""
    strategy_id: str
    signal_id: str
    timestamp: datetime

    # 信号内容
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    strength: float   # 信号强度

    # 价格信息
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    # 仓位管理
    position_size: float = 0.1
    max_risk: float = 0.02

    # 元信息
    market_regime: Optional[MarketRegime] = None
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'strategy_id': self.strategy_id,
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'confidence': self.confidence,
            'strength': self.strength,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'position_size': self.position_size,
            'max_risk': self.max_risk,
            'market_regime': self.market_regime.value if self.market_regime else None,
            'features': self.features
        }

@dataclass

class StrategyPerformance:
    """策略性能"""
    strategy_id: str
    evaluation_period: Tuple[datetime, datetime]

    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    cumulative_return: float = 0.0

    # 风险指标
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0

    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 交易统计
    total_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # 时间特征
    active_time: float = 0.0  # 策略活跃时间比例
    signal_frequency: float = 0.0  # 信号频率

    # 市场适应性
    regime_performance: Dict[str, float] = field(default_factory=dict)
    correlation_to_market: float = 0.0

    def to_dict(self) -> dict:
        return {
            'strategy_id': self.strategy_id,
            'evaluation_period': [self.evaluation_period[0].isoformat(), self.evaluation_period[1].isoformat()],
            'returns': {
                'total_return': self.total_return,
                'annual_return': self.annual_return,
                'cumulative_return': self.cumulative_return
            },
            'risk': {
                'volatility': self.volatility,
                'max_drawdown': self.max_drawdown,
                'var_95': self.var_95
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio
            },
            'trading': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor
            },
            'characteristics': {
                'active_time': self.active_time,
                'signal_frequency': self.signal_frequency,
                'correlation_to_market': self.correlation_to_market
            },
            'regime_performance': self.regime_performance
        }

class BaseStrategy(ABC):
    """基础策略接口"""

    def __init__(self, strategy_id: str, strategy_name: str, strategy_type: StrategyType):
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type

        # 策略状态
        self.is_active = True
        self.is_trained = False

        # 性能追踪
        self.signal_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)

        # 配置参数
        self.parameters = {}

    @abstractmethod

    def generate_signal(self, market_data: pd.DataFrame,
                       timestamp: datetime = None) -> Optional[StrategySignal]:
        """生成交易信号"""
        pass

    @abstractmethod

    def update_parameters(self, new_params: Dict[str, Any]):
        """更新策略参数"""
        pass

    def get_performance_metrics(self, start_date: datetime = None,
                              end_date: datetime = None) -> StrategyPerformance:
        """获取性能指标"""
        if not self.signal_history:
            return StrategyPerformance(
                strategy_id=self.strategy_id,
                evaluation_period=(datetime.now(), datetime.now())
            )

        # 过滤时间范围
        signals = list(self.signal_history)
        if start_date:
            signals = [s for s in signals if s.timestamp >= start_date]
        if end_date:
            signals = [s for s in signals if s.timestamp <= end_date]

        if not signals:
            return StrategyPerformance(
                strategy_id=self.strategy_id,
                evaluation_period=(start_date or datetime.now(), end_date or datetime.now())
            )

        # 计算基础统计
        confidences = [s.confidence for s in signals]
        strengths = [s.strength for s in signals]

        performance = StrategyPerformance(
            strategy_id=self.strategy_id,
            evaluation_period=(signals[0].timestamp, signals[-1].timestamp),
            total_trades=len(signals),
            signal_frequency=len(signals) / max(1, (signals[-1].timestamp - signals[0].timestamp).days),
            # 这里可以添加更多的计算逻辑
        )

        return performance

class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""

    def __init__(self, strategy_id: str, lookback_period: int = 20,
                 momentum_threshold: float = 0.02):
        super().__init__(strategy_id, "Trend Following", StrategyType.TREND_FOLLOWING)

        self.parameters = {
            'lookback_period': lookback_period,
            'momentum_threshold': momentum_threshold,
            'confidence_factor': 0.8
        }

    def generate_signal(self, market_data: pd.DataFrame,
                       timestamp: datetime = None) -> Optional[StrategySignal]:
        """生成趋势跟踪信号"""
        if len(market_data) < self.parameters['lookback_period']:
            return None

        timestamp = timestamp or datetime.now()

        try:
            # 计算移动平均
            lookback = self.parameters['lookback_period']
            prices = market_data['close'].iloc[-lookback:]

            # 短期和长期移动平均
            short_ma = prices.iloc[-5:].mean()
            long_ma = prices.mean()

            # 动量计算
            momentum = (short_ma - long_ma) / long_ma

            # 生成信号
            if momentum > self.parameters['momentum_threshold']:
                action = "BUY"
                confidence = min(abs(momentum) * 10, 1.0)
                strength = abs(momentum)
            elif momentum < -self.parameters['momentum_threshold']:
                action = "SELL"
                confidence = min(abs(momentum) * 10, 1.0)
                strength = abs(momentum)
            else:
                action = "HOLD"
                confidence = 0.3
                strength = 0.1

            # 创建信号
            signal = StrategySignal(
                strategy_id=self.strategy_id,
                signal_id=f"{self.strategy_id}_{int(timestamp.timestamp())}",
                timestamp=timestamp,
                action=action,
                confidence=confidence * self.parameters['confidence_factor'],
                strength=strength,
                entry_price=market_data['close'].iloc[-1],
                features={
                    'momentum': momentum,
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'price': market_data['close'].iloc[-1]
                }
            )

            self.signal_history.append(signal)
            return signal

        except Exception as e:
            logger.error(f"趋势策略 {self.strategy_id} 信号生成失败: {e}")
            return None

    def update_parameters(self, new_params: Dict[str, Any]):
        """更新参数"""
        for key, value in new_params.items():
            if key in self.parameters:
                self.parameters[key] = value

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""

    def __init__(self, strategy_id: str, lookback_period: int = 20,
                 deviation_threshold: float = 2.0):
        super().__init__(strategy_id, "Mean Reversion", StrategyType.MEAN_REVERSION)

        self.parameters = {
            'lookback_period': lookback_period,
            'deviation_threshold': deviation_threshold,
            'confidence_factor': 0.7
        }

    def generate_signal(self, market_data: pd.DataFrame,
                       timestamp: datetime = None) -> Optional[StrategySignal]:
        """生成均值回归信号"""
        if len(market_data) < self.parameters['lookback_period']:
            return None

        timestamp = timestamp or datetime.now()

        try:
            # 计算布林带
            lookback = self.parameters['lookback_period']
            prices = market_data['close'].iloc[-lookback:]

            mean_price = prices.mean()
            std_price = prices.std()
            current_price = market_data['close'].iloc[-1]

            # 计算Z分数
            z_score = (current_price - mean_price) / std_price

            # 生成信号
            if z_score > self.parameters['deviation_threshold']:
                action = "SELL"  # 价格过高，预期回归
                confidence = min(abs(z_score) / 3, 1.0)
                strength = abs(z_score) / 3
            elif z_score < -self.parameters['deviation_threshold']:
                action = "BUY"   # 价格过低，预期回归
                confidence = min(abs(z_score) / 3, 1.0)
                strength = abs(z_score) / 3
            else:
                action = "HOLD"
                confidence = 0.2
                strength = 0.1

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                signal_id=f"{self.strategy_id}_{int(timestamp.timestamp())}",
                timestamp=timestamp,
                action=action,
                confidence=confidence * self.parameters['confidence_factor'],
                strength=strength,
                entry_price=current_price,
                target_price=mean_price,  # 目标价格为均值
                features={
                    'z_score': z_score,
                    'mean_price': mean_price,
                    'std_price': std_price,
                    'current_price': current_price
                }
            )

            self.signal_history.append(signal)
            return signal

        except Exception as e:
            logger.error(f"均值回归策略 {self.strategy_id} 信号生成失败: {e}")
            return None

    def update_parameters(self, new_params: Dict[str, Any]):
        """更新参数"""
        for key, value in new_params.items():
            if key in self.parameters:
                self.parameters[key] = value

class VolatilityStrategy(BaseStrategy):
    """波动率策略"""

    def __init__(self, strategy_id: str, volatility_window: int = 20,
                 high_vol_threshold: float = 0.02, low_vol_threshold: float = 0.005):
        super().__init__(strategy_id, "Volatility Strategy", StrategyType.VOLATILITY)

        self.parameters = {
            'volatility_window': volatility_window,
            'high_vol_threshold': high_vol_threshold,
            'low_vol_threshold': low_vol_threshold,
            'confidence_factor': 0.6
        }

    def generate_signal(self, market_data: pd.DataFrame,
                       timestamp: datetime = None) -> Optional[StrategySignal]:
        """生成波动率策略信号"""
        if len(market_data) < self.parameters['volatility_window']:
            return None

        timestamp = timestamp or datetime.now()

        try:
            # 计算历史波动率
            window = self.parameters['volatility_window']
            returns = market_data['close'].pct_change().iloc[-window:]
            current_volatility = returns.std()

            # 计算波动率趋势
            vol_trend = returns.iloc[-5:].std() - returns.iloc[-10:-5].std()

            # 生成信号
            if current_volatility > self.parameters['high_vol_threshold']:
                if vol_trend > 0:
                    action = "SELL"  # 波动率上升，做空波动率
                    confidence = 0.7
                else:
                    action = "HOLD"
                    confidence = 0.3
            elif current_volatility < self.parameters['low_vol_threshold']:
                if vol_trend < 0:
                    action = "BUY"   # 波动率下降，做多
                    confidence = 0.6
                else:
                    action = "HOLD"
                    confidence = 0.3
            else:
                action = "HOLD"
                confidence = 0.2

            strength = abs(current_volatility - np.mean([self.parameters['high_vol_threshold'], self.parameters['low_vol_threshold']]))

            signal = StrategySignal(
                strategy_id=self.strategy_id,
                signal_id=f"{self.strategy_id}_{int(timestamp.timestamp())}",
                timestamp=timestamp,
                action=action,
                confidence=confidence * self.parameters['confidence_factor'],
                strength=strength,
                entry_price=market_data['close'].iloc[-1],
                features={
                    'current_volatility': current_volatility,
                    'vol_trend': vol_trend,
                    'high_threshold': self.parameters['high_vol_threshold'],
                    'low_threshold': self.parameters['low_vol_threshold']
                }
            )

            self.signal_history.append(signal)
            return signal

        except Exception as e:
            logger.error(f"波动率策略 {self.strategy_id} 信号生成失败: {e}")
            return None

    def update_parameters(self, new_params: Dict[str, Any]):
        """更新参数"""
        for key, value in new_params.items():
            if key in self.parameters:
                self.parameters[key] = value

class StrategySelector:
    """策略选择器"""

    def __init__(self, selection_method: SelectionMethod = SelectionMethod.PERFORMANCE_BASED):
        self.selection_method = selection_method
        self.strategies = {}
        self.selection_history = deque(maxlen=1000)

        # 选择参数
        self.performance_window = 30  # 性能评估窗口（天）
        self.min_performance_threshold = 0.1  # 最小性能阈值
        self.diversification_factor = 0.3  # 多样化因子

    def add_strategy(self, strategy: BaseStrategy):
        """添加策略"""
        self.strategies[strategy.strategy_id] = strategy
        logger.info(f"添加策略: {strategy.strategy_name} ({strategy.strategy_id})")

    def remove_strategy(self, strategy_id: str):
        """移除策略"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            logger.info(f"移除策略: {strategy_id}")

    def select_strategies(self, market_data: pd.DataFrame,
                         current_regime: MarketRegime = None,
                         max_strategies: int = 3) -> List[str]:
        """选择策略"""

        if not self.strategies:
            return []

        if self.selection_method == SelectionMethod.PERFORMANCE_BASED:
            return self._performance_based_selection(max_strategies)
        elif self.selection_method == SelectionMethod.RISK_ADJUSTED:
            return self._risk_adjusted_selection(max_strategies)
        elif self.selection_method == SelectionMethod.ADAPTIVE:
            return self._adaptive_selection(market_data, current_regime, max_strategies)
        elif self.selection_method == SelectionMethod.ENSEMBLE:
            return self._ensemble_selection(max_strategies)
        else:
            # 默认返回所有策略
            return list(self.strategies.keys())[:max_strategies]

    def _performance_based_selection(self, max_strategies: int) -> List[str]:
        """基于性能的选择"""
        strategy_scores = {}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.performance_window)

        for strategy_id, strategy in self.strategies.items():
            performance = strategy.get_performance_metrics(start_date, end_date)

            # 计算综合分数
            score = (
                performance.annual_return * 0.4 +
                performance.sharpe_ratio * 0.3 +
                performance.win_rate * 0.2 +
                (1 - performance.max_drawdown) * 0.1
            )

            strategy_scores[strategy_id] = score

        # 按分数排序并选择top N
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in sorted_strategies[:max_strategies]]

        self.selection_history.append({
            'timestamp': datetime.now(),
            'method': self.selection_method.value,
            'selected_strategies': selected,
            'scores': strategy_scores
        })

        return selected

    def _risk_adjusted_selection(self, max_strategies: int) -> List[str]:
        """风险调整选择"""
        strategy_scores = {}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.performance_window)

        for strategy_id, strategy in self.strategies.items():
            performance = strategy.get_performance_metrics(start_date, end_date)

            # 风险调整分数
            if performance.volatility > 0:
                risk_adjusted_return = performance.annual_return / performance.volatility
                drawdown_penalty = 1 - performance.max_drawdown

                score = risk_adjusted_return * drawdown_penalty
            else:
                score = 0

            strategy_scores[strategy_id] = score

        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in sorted_strategies[:max_strategies]]

        return selected

    def _adaptive_selection(self, market_data: pd.DataFrame,
                          current_regime: MarketRegime, max_strategies: int) -> List[str]:
        """自适应选择"""
        if current_regime is None:
            return self._performance_based_selection(max_strategies)

        # 根据市场状态选择适合的策略类型
        regime_strategy_mapping = {
            MarketRegime.TRENDING_UP: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM],
            MarketRegime.TRENDING_DOWN: [StrategyType.TREND_FOLLOWING, StrategyType.MEAN_REVERSION],
            MarketRegime.SIDEWAYS: [StrategyType.MEAN_REVERSION, StrategyType.PAIRS_TRADING],
            MarketRegime.HIGH_VOLATILITY: [StrategyType.VOLATILITY, StrategyType.MEAN_REVERSION],
            MarketRegime.LOW_VOLATILITY: [StrategyType.MOMENTUM, StrategyType.TREND_FOLLOWING]
        }

        preferred_types = regime_strategy_mapping.get(current_regime, list(StrategyType))

        # 筛选匹配的策略
        suitable_strategies = [
            strategy_id for strategy_id, strategy in self.strategies.items()
            if strategy.strategy_type in preferred_types
        ]

        if not suitable_strategies:
            return self._performance_based_selection(max_strategies)

        # 在合适的策略中选择性能最好的
        strategy_scores = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.performance_window)

        for strategy_id in suitable_strategies:
            strategy = self.strategies[strategy_id]
            performance = strategy.get_performance_metrics(start_date, end_date)

            score = (
                performance.annual_return * 0.5 +
                performance.sharpe_ratio * 0.3 +
                performance.win_rate * 0.2
            )

            strategy_scores[strategy_id] = score

        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [s[0] for s in sorted_strategies[:max_strategies]]

        return selected

    def _ensemble_selection(self, max_strategies: int) -> List[str]:
        """集成选择"""
        # 选择不同类型的策略以增加多样性
        strategy_by_type = defaultdict(list)

        for strategy_id, strategy in self.strategies.items():
            strategy_by_type[strategy.strategy_type].append(strategy_id)

        selected = []
        types_used = []

        # 从每种类型中选择最好的策略
        for strategy_type, strategy_list in strategy_by_type.items():
            if len(selected) >= max_strategies:
                break

            # 选择该类型中表现最好的策略
            best_strategy = self._select_best_in_type(strategy_list)
            if best_strategy:
                selected.append(best_strategy)
                types_used.append(strategy_type)

        # 如果还没达到max_strategies，从剩余策略中选择
        while len(selected) < max_strategies:
            remaining_strategies = [
                s_id for s_id in self.strategies.keys()
                if s_id not in selected
            ]

            if not remaining_strategies:
                break

            best_remaining = self._select_best_in_type(remaining_strategies)
            if best_remaining:
                selected.append(best_remaining)
            else:
                break

        return selected

    def _select_best_in_type(self, strategy_list: List[str]) -> Optional[str]:
        """在指定策略列表中选择最佳策略"""
        if not strategy_list:
            return None

        best_strategy = None
        best_score = -float('inf')

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.performance_window)

        for strategy_id in strategy_list:
            if strategy_id not in self.strategies:
                continue

            strategy = self.strategies[strategy_id]
            performance = strategy.get_performance_metrics(start_date, end_date)

            score = performance.sharpe_ratio

            if score > best_score:
                best_score = score
                best_strategy = strategy_id

        return best_strategy

class MetaStrategyEngine:
    """元策略引擎"""

    def __init__(self):
        self.strategies = {}
        self.strategy_selector = StrategySelector()

        # 信号聚合
        self.current_signals = {}
        self.signal_history = deque(maxlen=1000)

        # 性能监控
        self.performance_monitor = {}

        # 统计信息
        self.stats = {
            'total_strategies': 0,
            'active_strategies': 0,
            'total_signals_generated': 0,
            'selection_changes': 0
        }

        logger.info("元策略引擎初始化完成")

    def add_strategy(self, strategy: BaseStrategy):
        """添加策略"""
        self.strategies[strategy.strategy_id] = strategy
        self.strategy_selector.add_strategy(strategy)
        self.stats['total_strategies'] += 1
        self.stats['active_strategies'] += 1

        logger.info(f"添加策略到元引擎: {strategy.strategy_name}")

    def remove_strategy(self, strategy_id: str):
        """移除策略"""
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.strategy_selector.remove_strategy(strategy_id)
            self.stats['total_strategies'] -= 1
            if strategy_id in self.current_signals:
                del self.current_signals[strategy_id]

    async def generate_meta_signals(self, market_data: pd.DataFrame,
                                  current_regime: MarketRegime = None,
                                  max_active_strategies: int = 3) -> List[StrategySignal]:
        """生成元策略信号"""

        # 选择活跃策略
        selected_strategies = self.strategy_selector.select_strategies(
            market_data, current_regime, max_active_strategies
        )

        # 生成信号
        signals = []
        tasks = []

        for strategy_id in selected_strategies:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]

                # 异步生成信号
                task = asyncio.create_task(
                    self._generate_strategy_signal(strategy, market_data)
                )
                tasks.append((strategy_id, task))

        # 等待所有信号生成完成
        for strategy_id, task in tasks:
            try:
                signal = await task
                if signal:
                    signals.append(signal)
                    self.current_signals[strategy_id] = signal
                    self.stats['total_signals_generated'] += 1
            except Exception as e:
                logger.error(f"策略 {strategy_id} 信号生成异常: {e}")

        # 记录信号历史
        if signals:
            self.signal_history.append({
                'timestamp': datetime.now(),
                'signals': [s.to_dict() for s in signals],
                'selected_strategies': selected_strategies,
                'market_regime': current_regime.value if current_regime else None
            })

        return signals

    async def _generate_strategy_signal(self, strategy: BaseStrategy,
                                      market_data: pd.DataFrame) -> Optional[StrategySignal]:
        """异步生成单个策略信号"""
        try:
            return strategy.generate_signal(market_data)
        except Exception as e:
            logger.error(f"策略 {strategy.strategy_id} 信号生成失败: {e}")
            return None

    def aggregate_signals(self, signals: List[StrategySignal],
                         aggregation_method: str = "weighted_vote") -> Optional[StrategySignal]:
        """聚合多个策略信号"""
        if not signals:
            return None

        if aggregation_method == "weighted_vote":
            return self._weighted_vote_aggregation(signals)
        elif aggregation_method == "consensus":
            return self._consensus_aggregation(signals)
        elif aggregation_method == "best_performer":
            return self._best_performer_aggregation(signals)
        else:
            return signals[0]  # 默认返回第一个信号

    def _weighted_vote_aggregation(self, signals: List[StrategySignal]) -> StrategySignal:
        """加权投票聚合"""
        # 按置信度加权
        total_weight = sum(s.confidence * s.strength for s in signals)

        if total_weight == 0:
            return signals[0]

        # 计算加权平均
        weighted_confidence = sum(s.confidence * s.strength * s.confidence for s in signals) / total_weight
        weighted_strength = sum(s.confidence * s.strength * s.strength for s in signals) / total_weight

        # 决定最终动作
        buy_weight = sum(s.confidence * s.strength for s in signals if s.action == "BUY")
        sell_weight = sum(s.confidence * s.strength for s in signals if s.action == "SELL")
        hold_weight = sum(s.confidence * s.strength for s in signals if s.action == "HOLD")

        if buy_weight > sell_weight and buy_weight > hold_weight:
            final_action = "BUY"
        elif sell_weight > buy_weight and sell_weight > hold_weight:
            final_action = "SELL"
        else:
            final_action = "HOLD"

        # 创建聚合信号
        aggregated_signal = StrategySignal(
            strategy_id="meta_strategy",
            signal_id=f"meta_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            action=final_action,
            confidence=weighted_confidence,
            strength=weighted_strength,
            entry_price=np.mean([s.entry_price for s in signals if s.entry_price]),
            features={
                'source_strategies': [s.strategy_id for s in signals],
                'individual_confidences': [s.confidence for s in signals],
                'vote_weights': {'buy': buy_weight, 'sell': sell_weight, 'hold': hold_weight}
            }
        )

        return aggregated_signal

    def _consensus_aggregation(self, signals: List[StrategySignal]) -> StrategySignal:
        """共识聚合"""
        # 计算动作共识
        actions = [s.action for s in signals]
        action_counts = {action: actions.count(action) for action in set(actions)}

        # 需要超过一半的策略同意才生成信号
        consensus_threshold = len(signals) / 2

        for action, count in action_counts.items():
            if count > consensus_threshold:
                # 计算该动作的平均置信度
                action_signals = [s for s in signals if s.action == action]
                avg_confidence = np.mean([s.confidence for s in action_signals])
                avg_strength = np.mean([s.strength for s in action_signals])

                return StrategySignal(
                    strategy_id="meta_consensus",
                    signal_id=f"consensus_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    action=action,
                    confidence=avg_confidence,
                    strength=avg_strength,
                    entry_price=np.mean([s.entry_price for s in action_signals if s.entry_price]),
                    features={
                        'consensus_count': count,
                        'total_strategies': len(signals),
                        'consensus_ratio': count / len(signals)
                    }
                )

        # 没有达成共识，返回HOLD信号
        return StrategySignal(
            strategy_id="meta_consensus",
            signal_id=f"no_consensus_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            action="HOLD",
            confidence=0.1,
            strength=0.1,
            features={'consensus_achieved': False}
        )

    def _best_performer_aggregation(self, signals: List[StrategySignal]) -> StrategySignal:
        """最佳表现者聚合"""
        # 根据策略的历史性能选择信号
        best_signal = None
        best_score = -float('inf')

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        for signal in signals:
            if signal.strategy_id in self.strategies:
                strategy = self.strategies[signal.strategy_id]
                performance = strategy.get_performance_metrics(start_date, end_date)

                # 综合评分
                score = performance.sharpe_ratio * 0.5 + performance.win_rate * 0.3 + signal.confidence * 0.2

                if score > best_score:
                    best_score = score
                    best_signal = signal

        return best_signal or signals[0]

    def get_strategy_rankings(self) -> List[Dict[str, Any]]:
        """获取策略排名"""
        rankings = []

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        for strategy_id, strategy in self.strategies.items():
            performance = strategy.get_performance_metrics(start_date, end_date)

            ranking_info = {
                'strategy_id': strategy_id,
                'strategy_name': strategy.strategy_name,
                'strategy_type': strategy.strategy_type.value,
                'performance': performance.to_dict(),
                'is_active': strategy.is_active,
                'recent_signals': len([s for s in strategy.signal_history if s.timestamp >= start_date])
            }

            rankings.append(ranking_info)

        # 按夏普比率排序
        rankings.sort(key=lambda x: x['performance']['risk_adjusted']['sharpe_ratio'], reverse=True)

        return rankings

    def get_engine_summary(self) -> Dict[str, Any]:
        """获取引擎摘要"""
        active_strategies = [s for s in self.strategies.values() if s.is_active]

        return {
            'stats': self.stats,
            'strategies': {
                'total': len(self.strategies),
                'active': len(active_strategies),
                'by_type': {
                    strategy_type.value: len([s for s in self.strategies.values() if s.strategy_type == strategy_type])
                    for strategy_type in StrategyType
                }
            },
            'current_signals': len(self.current_signals),
            'signal_history_size': len(self.signal_history),
            'selection_method': self.strategy_selector.selection_method.value
        }

# 预定义策略创建函数

def create_default_strategies() -> List[BaseStrategy]:
    """创建默认策略集合"""
    strategies = [
        TrendFollowingStrategy("trend_1", lookback_period=20, momentum_threshold=0.02),
        TrendFollowingStrategy("trend_2", lookback_period=50, momentum_threshold=0.015),
        MeanReversionStrategy("mean_rev_1", lookback_period=20, deviation_threshold=2.0),
        MeanReversionStrategy("mean_rev_2", lookback_period=30, deviation_threshold=1.5),
        VolatilityStrategy("vol_1", volatility_window=20, high_vol_threshold=0.02, low_vol_threshold=0.005),
    ]

    return strategies

# 全局实例
_meta_strategy_engine_instance = None

def get_meta_strategy_engine() -> MetaStrategyEngine:
    """获取元策略引擎实例"""
    global _meta_strategy_engine_instance
    if _meta_strategy_engine_instance is None:
        _meta_strategy_engine_instance = MetaStrategyEngine()
    return _meta_strategy_engine_instance
