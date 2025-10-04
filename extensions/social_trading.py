"""
Social Trading
社交交易模块，提供跟单、信号分享、交易员排行等社交交易功能
支持多种跟单策略和风险控制机制
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import json
import numpy as np

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class TraderStatus(Enum):
    """交易员状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"

class FollowType(Enum):
    """跟单类型"""
    COPY_TRADING = "copy_trading"    # 复制交易
    SIGNAL_FOLLOWING = "signal_following"  # 信号跟随
    MIRROR_TRADING = "mirror_trading"  # 镜像交易
    CUSTOM = "custom"  # 自定义跟单

class RiskLevel(Enum):
    """风险级别"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    HIGH_RISK = "high_risk"

@dataclass

class TraderProfile:
    """交易员档案"""
    trader_id: str
    username: str
    display_name: str
    avatar_url: Optional[str] = None

    # 状态信息
    status: TraderStatus = TraderStatus.UNVERIFIED
    joined_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # 基本信息
    bio: str = ""
    country: Optional[str] = None
    experience_years: Optional[int] = None
    trading_style: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.MODERATE

    # 订阅信息
    followers_count: int = 0
    following_count: int = 0
    is_premium: bool = False
    subscription_fee: float = 0.0  # 月费

    # 验证信息
    is_verified: bool = False
    verification_date: Optional[datetime] = None
    verification_badges: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'trader_id': self.trader_id,
            'username': self.username,
            'display_name': self.display_name,
            'avatar_url': self.avatar_url,
            'status': self.status.value,
            'joined_date': self.joined_date.isoformat(),
            'last_active': self.last_active.isoformat(),
            'bio': self.bio,
            'country': self.country,
            'experience_years': self.experience_years,
            'trading_style': self.trading_style,
            'risk_level': self.risk_level.value,
            'followers_count': self.followers_count,
            'following_count': self.following_count,
            'is_premium': self.is_premium,
            'subscription_fee': self.subscription_fee,
            'is_verified': self.is_verified,
            'verification_date': self.verification_date.isoformat() if self.verification_date else None,
            'verification_badges': self.verification_badges
        }

@dataclass

class TraderPerformance:
    """交易员业绩"""
    trader_id: str
    period_start: datetime
    period_end: datetime

    # 基础指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # 收益指标
    total_return: float = 0.0
    monthly_return: float = 0.0
    annual_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 交易行为
    avg_holding_period: float = 0.0  # 小时
    avg_trade_size: float = 0.0
    trading_frequency: float = 0.0  # 每日交易次数

    # 一致性指标
    consistency_score: float = 0.0
    profit_factor: float = 0.0
    recovery_factor: float = 0.0

    def to_dict(self) -> dict:
        return {
            'trader_id': self.trader_id,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'monthly_return': self.monthly_return,
            'annual_return': self.annual_return,
            'best_trade': self.best_trade,
            'worst_trade': self.worst_trade,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'avg_holding_period': self.avg_holding_period,
            'avg_trade_size': self.avg_trade_size,
            'trading_frequency': self.trading_frequency,
            'consistency_score': self.consistency_score,
            'profit_factor': self.profit_factor,
            'recovery_factor': self.recovery_factor
        }

@dataclass

class FollowSettings:
    """跟单设置"""
    follower_id: str
    trader_id: str
    follow_type: FollowType

    # 基本设置
    is_active: bool = True
    started_date: datetime = field(default_factory=datetime.now)

    # 资金管理
    allocation_amount: float = 1000.0  # 分配资金
    allocation_ratio: float = 1.0  # 跟单比例
    max_position_size: float = 0.1  # 最大仓位比例

    # 风险控制
    max_daily_loss: float = 0.05  # 最大日亏损
    max_drawdown: float = 0.2  # 最大回撤
    stop_loss_ratio: float = 0.1  # 止损比例

    # 跟单策略
    copy_symbols: List[str] = field(default_factory=list)  # 跟单品种
    exclude_symbols: List[str] = field(default_factory=list)  # 排除品种
    min_signal_confidence: float = 0.6  # 最小信号置信度
    delay_seconds: int = 0  # 跟单延迟

    # 高级设置
    reverse_mode: bool = False  # 反向跟单
    partial_close_enabled: bool = True  # 允许部分平仓
    auto_adjust_size: bool = True  # 自动调整仓位大小

    def to_dict(self) -> dict:
        return {
            'follower_id': self.follower_id,
            'trader_id': self.trader_id,
            'follow_type': self.follow_type.value,
            'is_active': self.is_active,
            'started_date': self.started_date.isoformat(),
            'allocation_amount': self.allocation_amount,
            'allocation_ratio': self.allocation_ratio,
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'stop_loss_ratio': self.stop_loss_ratio,
            'copy_symbols': self.copy_symbols,
            'exclude_symbols': self.exclude_symbols,
            'min_signal_confidence': self.min_signal_confidence,
            'delay_seconds': self.delay_seconds,
            'reverse_mode': self.reverse_mode,
            'partial_close_enabled': self.partial_close_enabled,
            'auto_adjust_size': self.auto_adjust_size
        }

@dataclass

class TradingSignal:
    """交易信号"""
    signal_id: str
    trader_id: str
    symbol: str
    action: str  # BUY, SELL, CLOSE

    # 价格信息
    entry_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    # 数量信息
    quantity: Optional[float] = None
    position_size_ratio: float = 0.1

    # 信号属性
    confidence: float = 0.8
    signal_type: str = "manual"  # manual, automated, ai
    timeframe: str = "1D"

    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # 状态信息
    status: str = "active"  # active, expired, executed, cancelled

    # 描述信息
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'trader_id': self.trader_id,
            'symbol': self.symbol,
            'action': self.action,
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'quantity': self.quantity,
            'position_size_ratio': self.position_size_ratio,
            'confidence': self.confidence,
            'signal_type': self.signal_type,
            'timeframe': self.timeframe,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'status': self.status,
            'description': self.description,
            'tags': self.tags
        }

class TraderRegistry:
    """交易员注册表"""

    def __init__(self):
        self.traders: Dict[str, TraderProfile] = {}
        self.performances: Dict[str, List[TraderPerformance]] = defaultdict(list)
        self.rankings = {}
        self.last_ranking_update = datetime.now()

    def register_trader(self, profile: TraderProfile) -> bool:
        """注册交易员"""
        try:
            self.traders[profile.trader_id] = profile
            logger.info(f"注册交易员: {profile.username} ({profile.trader_id})")
            return True
        except Exception as e:
            logger.error(f"注册交易员失败: {e}")
            return False

    def update_trader_profile(self, trader_id: str, updates: Dict[str, Any]) -> bool:
        """更新交易员档案"""
        if trader_id not in self.traders:
            return False

        try:
            trader = self.traders[trader_id]
            for key, value in updates.items():
                if hasattr(trader, key):
                    setattr(trader, key, value)

            trader.last_active = datetime.now()
            return True
        except Exception as e:
            logger.error(f"更新交易员档案失败: {e}")
            return False

    def add_performance_record(self, performance: TraderPerformance):
        """添加业绩记录"""
        self.performances[performance.trader_id].append(performance)

        # 限制记录数量
        if len(self.performances[performance.trader_id]) > 100:
            self.performances[performance.trader_id] = self.performances[performance.trader_id][-100:]

    def get_trader_ranking(self, metric: str = "total_return",
                         period_days: int = 30, limit: int = 50) -> List[Dict[str, Any]]:
        """获取交易员排行榜"""
        cutoff_date = datetime.now() - timedelta(days=period_days)
        ranked_traders = []

        for trader_id, performances in self.performances.items():
            if trader_id not in self.traders:
                continue

            trader = self.traders[trader_id]
            if trader.status != TraderStatus.ACTIVE:
                continue

            # 获取指定期间的最新业绩
            recent_performances = [
                p for p in performances
                if p.period_end >= cutoff_date
            ]

            if not recent_performances:
                continue

            latest_performance = max(recent_performances, key=lambda x: x.period_end)
            metric_value = getattr(latest_performance, metric, 0)

            ranked_traders.append({
                'trader_id': trader_id,
                'profile': trader.to_dict(),
                'performance': latest_performance.to_dict(),
                'ranking_metric': metric_value
            })

        # 按指标排序
        ranked_traders.sort(key=lambda x: x['ranking_metric'], reverse=True)

        return ranked_traders[:limit]

    def get_trader_by_id(self, trader_id: str) -> Optional[TraderProfile]:
        """根据ID获取交易员"""
        return self.traders.get(trader_id)

    def search_traders(self, query: str, filters: Dict[str, Any] = None) -> List[TraderProfile]:
        """搜索交易员"""
        results = []
        filters = filters or {}

        for trader in self.traders.values():
            # 文本搜索
            if query and query.lower() not in trader.username.lower() and query.lower() not in trader.display_name.lower():
                continue

            # 过滤器
            if filters.get('status') and trader.status != TraderStatus(filters['status']):
                continue

            if filters.get('risk_level') and trader.risk_level != RiskLevel(filters['risk_level']):
                continue

            if filters.get('is_verified') is not None and trader.is_verified != filters['is_verified']:
                continue

            if filters.get('min_followers') and trader.followers_count < filters['min_followers']:
                continue

            results.append(trader)

        return results

class FollowManager:
    """跟单管理器"""

    def __init__(self):
        self.follow_relationships: Dict[str, List[FollowSettings]] = defaultdict(list)  # follower_id -> settings
        self.followers_by_trader: Dict[str, List[str]] = defaultdict(list)  # trader_id -> follower_ids

    def create_follow_relationship(self, settings: FollowSettings) -> bool:
        """创建跟单关系"""
        try:
            # 检查是否已存在
            existing = self.get_follow_settings(settings.follower_id, settings.trader_id)
            if existing:
                logger.warning(f"跟单关系已存在: {settings.follower_id} -> {settings.trader_id}")
                return False

            # 添加关系
            self.follow_relationships[settings.follower_id].append(settings)
            self.followers_by_trader[settings.trader_id].append(settings.follower_id)

            logger.info(f"创建跟单关系: {settings.follower_id} -> {settings.trader_id}")
            return True

        except Exception as e:
            logger.error(f"创建跟单关系失败: {e}")
            return False

    def update_follow_settings(self, follower_id: str, trader_id: str,
                             updates: Dict[str, Any]) -> bool:
        """更新跟单设置"""
        settings = self.get_follow_settings(follower_id, trader_id)
        if not settings:
            return False

        try:
            for key, value in updates.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)

            return True
        except Exception as e:
            logger.error(f"更新跟单设置失败: {e}")
            return False

    def remove_follow_relationship(self, follower_id: str, trader_id: str) -> bool:
        """移除跟单关系"""
        try:
            # 从follower列表中移除
            follow_list = self.follow_relationships[follower_id]
            self.follow_relationships[follower_id] = [
                s for s in follow_list if s.trader_id != trader_id
            ]

            # 从trader的followers中移除
            if follower_id in self.followers_by_trader[trader_id]:
                self.followers_by_trader[trader_id].remove(follower_id)

            logger.info(f"移除跟单关系: {follower_id} -> {trader_id}")
            return True

        except Exception as e:
            logger.error(f"移除跟单关系失败: {e}")
            return False

    def get_follow_settings(self, follower_id: str, trader_id: str) -> Optional[FollowSettings]:
        """获取跟单设置"""
        for settings in self.follow_relationships[follower_id]:
            if settings.trader_id == trader_id:
                return settings
        return None

    def get_follower_settings(self, follower_id: str) -> List[FollowSettings]:
        """获取跟随者的所有设置"""
        return self.follow_relationships[follower_id][:]

    def get_trader_followers(self, trader_id: str) -> List[str]:
        """获取交易员的所有跟随者"""
        return self.followers_by_trader[trader_id][:]

class SignalBroadcaster:
    """信号广播器"""

    def __init__(self):
        self.signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)  # trader_id -> callbacks

    def publish_signal(self, signal: TradingSignal):
        """发布交易信号"""
        self.signals[signal.signal_id] = signal
        self.signal_history.append(signal)

        # 限制历史记录
        if len(self.signal_history) > 10000:
            self.signal_history = self.signal_history[-10000:]

        # 通知订阅者
        subscribers = self.subscribers.get(signal.trader_id, [])
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(signal))
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"通知信号订阅者失败: {e}")

        logger.debug(f"发布信号: {signal.trader_id} -> {signal.symbol} {signal.action}")

    def subscribe_to_trader(self, trader_id: str, callback: Callable):
        """订阅交易员信号"""
        self.subscribers[trader_id].append(callback)
        logger.debug(f"订阅交易员信号: {trader_id}")

    def unsubscribe_from_trader(self, trader_id: str, callback: Callable):
        """取消订阅交易员信号"""
        if callback in self.subscribers[trader_id]:
            self.subscribers[trader_id].remove(callback)
            logger.debug(f"取消订阅交易员信号: {trader_id}")

    def get_recent_signals(self, trader_id: str = None, limit: int = 50) -> List[TradingSignal]:
        """获取最近的信号"""
        if trader_id:
            signals = [s for s in self.signal_history if s.trader_id == trader_id]
        else:
            signals = self.signal_history

        return sorted(signals, key=lambda x: x.created_at, reverse=True)[:limit]

    def update_signal_status(self, signal_id: str, status: str):
        """更新信号状态"""
        if signal_id in self.signals:
            self.signals[signal_id].status = status

class CopyTradingEngine:
    """复制交易引擎"""

    def __init__(self, follow_manager: FollowManager, signal_broadcaster: SignalBroadcaster):
        self.follow_manager = follow_manager
        self.signal_broadcaster = signal_broadcaster
        self.execution_queue = asyncio.Queue()
        self.is_running = False
        self.execution_task = None

        # 统计信息
        self.stats = {
            'signals_processed': 0,
            'orders_executed': 0,
            'orders_failed': 0,
            'total_volume': 0.0,
            'average_delay': 0.0
        }

    async def start(self):
        """启动复制交易引擎"""
        if self.is_running:
            return

        self.is_running = True

        # 启动执行任务
        self.execution_task = asyncio.create_task(self._execution_loop())

        # 订阅所有交易员的信号
        self._setup_signal_subscriptions()

        logger.info("复制交易引擎启动完成")

    async def stop(self):
        """停止复制交易引擎"""
        if not self.is_running:
            return

        self.is_running = False

        if self.execution_task:
            self.execution_task.cancel()

        logger.info("复制交易引擎已停止")

    def _setup_signal_subscriptions(self):
        """设置信号订阅"""
        # 获取所有活跃的交易员
        all_traders = set()
        for follower_settings in self.follow_manager.follow_relationships.values():
            for settings in follower_settings:
                if settings.is_active:
                    all_traders.add(settings.trader_id)

        # 为每个交易员订阅信号
        for trader_id in all_traders:
            self.signal_broadcaster.subscribe_to_trader(
                trader_id,
                self._handle_signal
            )

    async def _handle_signal(self, signal: TradingSignal):
        """处理信号"""
        # 找到所有跟随该交易员的设置
        followers = self.follow_manager.get_trader_followers(signal.trader_id)

        for follower_id in followers:
            settings = self.follow_manager.get_follow_settings(follower_id, signal.trader_id)

            if not settings or not settings.is_active:
                continue

            # 检查信号是否符合跟单条件
            if not self._should_copy_signal(signal, settings):
                continue

            # 添加到执行队列
            copy_order = {
                'signal': signal,
                'follower_id': follower_id,
                'settings': settings,
                'timestamp': datetime.now()
            }

            await self.execution_queue.put(copy_order)

    def _should_copy_signal(self, signal: TradingSignal, settings: FollowSettings) -> bool:
        """检查是否应该复制信号"""
        # 检查品种过滤
        if settings.copy_symbols and signal.symbol not in settings.copy_symbols:
            return False

        if signal.symbol in settings.exclude_symbols:
            return False

        # 检查置信度
        if signal.confidence < settings.min_signal_confidence:
            return False

        # 检查信号类型
        if settings.follow_type == FollowType.SIGNAL_FOLLOWING and signal.signal_type == "manual":
            return False

        return True

    async def _execution_loop(self):
        """执行循环"""
        while self.is_running:
            try:
                # 等待复制订单
                copy_order = await asyncio.wait_for(
                    self.execution_queue.get(), timeout=1.0
                )

                await self._execute_copy_order(copy_order)
                self.stats['signals_processed'] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"执行循环错误: {e}")
                await asyncio.sleep(1)

    async def _execute_copy_order(self, copy_order: Dict[str, Any]):
        """执行复制订单"""
        signal = copy_order['signal']
        follower_id = copy_order['follower_id']
        settings = copy_order['settings']

        try:
            # 应用延迟
            if settings.delay_seconds > 0:
                await asyncio.sleep(settings.delay_seconds)

            # 计算订单参数
            order_params = self._calculate_order_parameters(signal, settings)

            # 模拟订单执行（实际实现中需要调用交易接口）
            success = await self._simulate_order_execution(order_params)

            if success:
                self.stats['orders_executed'] += 1
                self.stats['total_volume'] += order_params.get('value', 0)
                logger.info(f"复制交易成功: {follower_id} -> {signal.symbol} {signal.action}")
            else:
                self.stats['orders_failed'] += 1
                logger.warning(f"复制交易失败: {follower_id} -> {signal.symbol} {signal.action}")

            # 计算执行延迟
            execution_delay = (datetime.now() - copy_order['timestamp']).total_seconds()

            # 更新平均延迟
            total_processed = self.stats['signals_processed'] + 1
            current_avg = self.stats['average_delay']
            self.stats['average_delay'] = (
                (current_avg * (total_processed - 1) + execution_delay) / total_processed
            )

        except Exception as e:
            logger.error(f"执行复制订单失败: {e}")
            self.stats['orders_failed'] += 1

    def _calculate_order_parameters(self, signal: TradingSignal,
                                  settings: FollowSettings) -> Dict[str, Any]:
        """计算订单参数"""
        # 基础参数
        params = {
            'symbol': signal.symbol,
            'action': signal.action,
            'price': signal.entry_price
        }

        # 计算数量
        if signal.quantity:
            # 按比例调整数量
            params['quantity'] = signal.quantity * settings.allocation_ratio
        else:
            # 基于仓位比例计算
            position_ratio = min(signal.position_size_ratio, settings.max_position_size)
            params['quantity'] = settings.allocation_amount * position_ratio / signal.entry_price

        # 反向模式
        if settings.reverse_mode:
            if params['action'] == 'BUY':
                params['action'] = 'SELL'
            elif params['action'] == 'SELL':
                params['action'] = 'BUY'

        # 止损止盈
        if signal.stop_loss:
            params['stop_loss'] = signal.stop_loss

        if signal.target_price:
            params['target_price'] = signal.target_price

        # 计算订单价值
        params['value'] = params['quantity'] * params['price']

        return params

    async def _simulate_order_execution(self, order_params: Dict[str, Any]) -> bool:
        """模拟订单执行"""
        # 简单的成功率模拟（实际实现中需要调用真实的交易接口）
        await asyncio.sleep(0.1)  # 模拟网络延迟

        # 90%成功率
        return np.random.random() > 0.1

class SocialTradingPlatform:
    """社交交易平台主类"""

    def __init__(self):
        self.trader_registry = TraderRegistry()
        self.follow_manager = FollowManager()
        self.signal_broadcaster = SignalBroadcaster()
        self.copy_trading_engine = CopyTradingEngine(
            self.follow_manager,
            self.signal_broadcaster
        )

        # 模拟数据
        self._create_mock_data()

        logger.info("社交交易平台初始化完成")

    def _create_mock_data(self):
        """创建模拟数据"""
        # 创建模拟交易员
        mock_traders = [
            TraderProfile(
                trader_id="trader_001",
                username="pro_trader_john",
                display_name="Professional John",
                status=TraderStatus.VERIFIED,
                experience_years=8,
                trading_style="Swing Trading",
                risk_level=RiskLevel.MODERATE,
                followers_count=1250,
                is_verified=True,
                subscription_fee=29.99
            ),
            TraderProfile(
                trader_id="trader_002",
                username="crypto_master",
                display_name="Crypto Master",
                status=TraderStatus.ACTIVE,
                experience_years=5,
                trading_style="Day Trading",
                risk_level=RiskLevel.AGGRESSIVE,
                followers_count=830,
                is_verified=True,
                subscription_fee=49.99
            ),
            TraderProfile(
                trader_id="trader_003",
                username="safe_investor",
                display_name="Safe Investor",
                status=TraderStatus.ACTIVE,
                experience_years=12,
                trading_style="Value Investing",
                risk_level=RiskLevel.CONSERVATIVE,
                followers_count=2100,
                is_verified=True,
                subscription_fee=19.99
            )
        ]

        for trader in mock_traders:
            self.trader_registry.register_trader(trader)

        # 创建模拟业绩数据
        for trader in mock_traders:
            performance = TraderPerformance(
                trader_id=trader.trader_id,
                period_start=datetime.now() - timedelta(days=30),
                period_end=datetime.now(),
                total_trades=np.random.randint(50, 200),
                winning_trades=np.random.randint(30, 120),
                total_return=np.random.uniform(-0.1, 0.3),
                max_drawdown=np.random.uniform(0.02, 0.15),
                sharpe_ratio=np.random.uniform(0.5, 2.5),
                win_rate=np.random.uniform(0.4, 0.8)
            )

            self.trader_registry.add_performance_record(performance)

    async def start(self):
        """启动平台"""
        await self.copy_trading_engine.start()
        logger.info("社交交易平台启动完成")

    async def stop(self):
        """停止平台"""
        await self.copy_trading_engine.stop()
        logger.info("社交交易平台已停止")

    # 公共接口

    def get_trader_rankings(self, metric: str = "total_return", limit: int = 20) -> List[Dict[str, Any]]:
        """获取交易员排行榜"""
        return self.trader_registry.get_trader_ranking(metric, limit=limit)

    def search_traders(self, query: str = "", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """搜索交易员"""
        traders = self.trader_registry.search_traders(query, filters)
        return [trader.to_dict() for trader in traders]

    def follow_trader(self, follower_id: str, trader_id: str,
                     settings: Dict[str, Any] = None) -> bool:
        """跟随交易员"""
        # 检查交易员是否存在
        trader = self.trader_registry.get_trader_by_id(trader_id)
        if not trader:
            return False

        # 创建默认设置
        default_settings = FollowSettings(
            follower_id=follower_id,
            trader_id=trader_id,
            follow_type=FollowType.COPY_TRADING
        )

        # 应用自定义设置
        if settings:
            for key, value in settings.items():
                if hasattr(default_settings, key):
                    setattr(default_settings, key, value)

        return self.follow_manager.create_follow_relationship(default_settings)

    def unfollow_trader(self, follower_id: str, trader_id: str) -> bool:
        """取消跟随交易员"""
        return self.follow_manager.remove_follow_relationship(follower_id, trader_id)

    def publish_signal(self, trader_id: str, signal_data: Dict[str, Any]) -> str:
        """发布交易信号"""
        signal_id = f"signal_{trader_id}_{int(time.time())}"

        signal = TradingSignal(
            signal_id=signal_id,
            trader_id=trader_id,
            symbol=signal_data['symbol'],
            action=signal_data['action'],
            entry_price=signal_data['entry_price'],
            target_price=signal_data.get('target_price'),
            stop_loss=signal_data.get('stop_loss'),
            confidence=signal_data.get('confidence', 0.8),
            description=signal_data.get('description', ''),
            position_size_ratio=signal_data.get('position_size_ratio', 0.1)
        )

        self.signal_broadcaster.publish_signal(signal)
        return signal_id

    def get_recent_signals(self, trader_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的信号"""
        signals = self.signal_broadcaster.get_recent_signals(trader_id, limit)
        return [signal.to_dict() for signal in signals]

    def get_follower_dashboard(self, follower_id: str) -> Dict[str, Any]:
        """获取跟随者仪表板"""
        settings_list = self.follow_manager.get_follower_settings(follower_id)

        dashboard = {
            'follower_id': follower_id,
            'following_count': len(settings_list),
            'total_allocated': sum(s.allocation_amount for s in settings_list),
            'active_follows': sum(1 for s in settings_list if s.is_active),
            'following_details': [s.to_dict() for s in settings_list]
        }

        return dashboard

    def get_trader_dashboard(self, trader_id: str) -> Dict[str, Any]:
        """获取交易员仪表板"""
        trader = self.trader_registry.get_trader_by_id(trader_id)
        if not trader:
            return {}

        followers = self.follow_manager.get_trader_followers(trader_id)
        recent_signals = self.get_recent_signals(trader_id, 10)

        # 获取最新业绩
        performances = self.trader_registry.performances.get(trader_id, [])
        latest_performance = performances[-1] if performances else None

        dashboard = {
            'trader': trader.to_dict(),
            'followers_count': len(followers),
            'recent_signals_count': len(recent_signals),
            'recent_signals': recent_signals,
            'latest_performance': latest_performance.to_dict() if latest_performance else None,
            'copy_engine_stats': self.copy_trading_engine.stats
        }

        return dashboard

    def get_platform_stats(self) -> Dict[str, Any]:
        """获取平台统计"""
        total_traders = len(self.trader_registry.traders)
        verified_traders = sum(1 for t in self.trader_registry.traders.values() if t.is_verified)

        total_follows = sum(
            len(settings) for settings in self.follow_manager.follow_relationships.values()
        )

        active_follows = sum(
            sum(1 for s in settings if s.is_active)
            for settings in self.follow_manager.follow_relationships.values()
        )

        return {
            'total_traders': total_traders,
            'verified_traders': verified_traders,
            'total_follow_relationships': total_follows,
            'active_follow_relationships': active_follows,
            'total_signals_published': len(self.signal_broadcaster.signal_history),
            'copy_engine_stats': self.copy_trading_engine.stats,
            'platform_uptime': datetime.now().isoformat()
        }

# 全局实例
_social_trading_platform_instance = None

def get_social_trading_platform() -> SocialTradingPlatform:
    """获取社交交易平台实例"""
    global _social_trading_platform_instance
    if _social_trading_platform_instance is None:
        _social_trading_platform_instance = SocialTradingPlatform()
    return _social_trading_platform_instance
