"""
Signal Broadcaster
信号广播器，负责将交易信号实时广播给订阅者
支持多种广播方式和智能路由分发
"""

import asyncio
import json
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
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

class BroadcastChannel(Enum):
    """广播频道枚举"""
    WEBSOCKET = "websocket"
    HTTP_WEBHOOK = "http_webhook"
    EMAIL = "email"
    SMS = "sms"
    TELEGRAM = "telegram"
    SLACK = "slack"
    REDIS_PUB = "redis_pub"

class SignalPriority(Enum):
    """信号优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass

class TradingSignal:
    """交易信号数据结构"""
    signal_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    price: float
    confidence: float  # 0-1
    priority: SignalPriority = SignalPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)

    # 信号详情
    strategy_name: Optional[str] = None
    indicators: Dict[str, float] = field(default_factory=dict)
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_return: Optional[float] = None

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'price': self.price,
            'confidence': self.confidence,
            'priority': self.priority.name,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'indicators': self.indicators,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'expected_return': self.expected_return,
            'metadata': self.metadata,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

    def is_expired(self) -> bool:
        """检查信号是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

@dataclass

class Subscriber:
    """订阅者信息"""
    subscriber_id: str
    channels: Set[BroadcastChannel]
    symbols: Set[str] = field(default_factory=set)  # 订阅的交易对
    strategies: Set[str] = field(default_factory=set)  # 订阅的策略
    min_confidence: float = 0.0  # 最低置信度
    priority_filter: Set[SignalPriority] = field(default_factory=set)

    # 联系方式
    websocket_client_id: Optional[str] = None
    webhook_url: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    slack_webhook: Optional[str] = None

    # 统计信息
    signals_received: int = 0
    last_signal_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def matches_signal(self, signal: TradingSignal) -> bool:
        """检查是否匹配信号"""
        # 交易对过滤
        if self.symbols and signal.symbol not in self.symbols:
            return False

        # 策略过滤
        if self.strategies and signal.strategy_name not in self.strategies:
            return False

        # 置信度过滤
        if signal.confidence < self.min_confidence:
            return False

        # 优先级过滤
        if self.priority_filter and signal.priority not in self.priority_filter:
            return False

        return True

class WebSocketBroadcaster:
    """WebSocket广播器"""

    def __init__(self, websocket_server=None):
        self.websocket_server = websocket_server

    async def broadcast(self, signal: TradingSignal, subscriber: Subscriber):
        """广播信号到WebSocket"""
        if not self.websocket_server or not subscriber.websocket_client_id:
            return False

        try:
            message_data = {
                'type': 'trading_signal',
                'signal': signal.to_dict()
            }

            # 通过WebSocket服务器发送
            await self.websocket_server.send_to_client(
                subscriber.websocket_client_id,
                message_data
            )

            return True

        except Exception as e:
            logger.error(f"WebSocket广播失败: {e}")
            return False

class HTTPWebhookBroadcaster:
    """HTTP Webhook广播器"""

    def __init__(self):
        self.session = None

    async def _ensure_session(self):
        """确保HTTP会话存在"""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()

    async def broadcast(self, signal: TradingSignal, subscriber: Subscriber):
        """通过HTTP Webhook广播信号"""
        if not subscriber.webhook_url:
            return False

        await self._ensure_session()

        try:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'signal': signal.to_dict()
            }

            async with self.session.post(
                subscriber.webhook_url,
                json=payload,
                timeout=10
            ) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"Webhook广播失败 {subscriber.webhook_url}: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
            self.session = None

class EmailBroadcaster:
    """邮件广播器"""

    def __init__(self):
        self.smtp_config = config.get('email', {})

    async def broadcast(self, signal: TradingSignal, subscriber: Subscriber):
        """通过邮件广播信号"""
        if not subscriber.email:
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # 构建邮件内容
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_address', 'trading@system.com')
            msg['To'] = subscriber.email
            msg['Subject'] = f"交易信号: {signal.signal_type} {signal.symbol}"

            body = f"""
            交易信号详情:

            交易对: {signal.symbol}
            信号类型: {signal.signal_type}
            价格: {signal.price}
            置信度: {signal.confidence:.2%}
            策略: {signal.strategy_name or 'Unknown'}
            时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

            目标价格: {signal.target_price or 'N/A'}
            止损价格: {signal.stop_loss or 'N/A'}
            预期收益: {signal.expected_return or 'N/A'}

            技术指标:
            {json.dumps(signal.indicators, indent=2) if signal.indicators else 'N/A'}
            """

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 发送邮件（异步处理）
            import threading

            def send_email():
                try:
                    server = smtplib.SMTP(
                        self.smtp_config.get('host', 'localhost'),
                        self.smtp_config.get('port', 587)
                    )
                    server.starttls()
                    server.login(
                        self.smtp_config.get('username', ''),
                        self.smtp_config.get('password', '')
                    )
                    server.send_message(msg)
                    server.quit()
                except Exception as e:
                    logger.error(f"发送邮件失败: {e}")

            thread = threading.Thread(target=send_email)
            thread.start()

            return True

        except ImportError:
            logger.error("邮件功能需要smtplib库")
            return False
        except Exception as e:
            logger.error(f"邮件广播失败: {e}")
            return False

class TelegramBroadcaster:
    """Telegram广播器"""

    def __init__(self):
        self.bot_token = config.get('telegram', {}).get('bot_token')
        self.session = None

    async def _ensure_session(self):
        """确保HTTP会话存在"""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()

    async def broadcast(self, signal: TradingSignal, subscriber: Subscriber):
        """通过Telegram广播信号"""
        if not self.bot_token or not subscriber.telegram_chat_id:
            return False

        await self._ensure_session()

        try:
            message = f"""
📊 *交易信号*

🔸 交易对: `{signal.symbol}`
🔸 信号: *{signal.signal_type}*
🔸 价格: `{signal.price}`
🔸 置信度: `{signal.confidence:.2%}`
🔸 策略: `{signal.strategy_name or 'Unknown'}`

🎯 目标: `{signal.target_price or 'N/A'}`
🛡️ 止损: `{signal.stop_loss or 'N/A'}`

⏰ 时间: `{signal.timestamp.strftime('%H:%M:%S')}`
            """

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': subscriber.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            async with self.session.post(url, json=payload, timeout=10) as response:
                return response.status == 200

        except Exception as e:
            logger.error(f"Telegram广播失败: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
            self.session = None

class SignalBroadcaster:
    """信号广播器主类"""

    def __init__(self):
        self.subscribers: Dict[str, Subscriber] = {}
        self.broadcasters = {
            BroadcastChannel.WEBSOCKET: WebSocketBroadcaster(),
            BroadcastChannel.HTTP_WEBHOOK: HTTPWebhookBroadcaster(),
            BroadcastChannel.EMAIL: EmailBroadcaster(),
            BroadcastChannel.TELEGRAM: TelegramBroadcaster(),
        }

        # 信号队列和历史
        self.signal_queue = asyncio.Queue(maxsize=10000)
        self.signal_history = deque(maxlen=10000)
        self.pending_signals = {}  # signal_id -> TradingSignal

        # 任务管理
        self.broadcast_task = None
        self.cleanup_task = None
        self.is_running = False

        # 统计信息
        self.stats = {
            'total_signals': 0,
            'total_broadcasts': 0,
            'successful_broadcasts': 0,
            'failed_broadcasts': 0,
            'start_time': datetime.now()
        }

        logger.info("信号广播器初始化完成")

    async def start(self):
        """启动信号广播器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动广播任务
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("信号广播器启动完成")

    async def stop(self):
        """停止信号广播器"""
        if not self.is_running:
            return

        logger.info("正在停止信号广播器...")
        self.is_running = False

        # 取消任务
        if self.broadcast_task:
            self.broadcast_task.cancel()

        if self.cleanup_task:
            self.cleanup_task.cancel()

        # 清理广播器资源
        for broadcaster in self.broadcasters.values():
            if hasattr(broadcaster, 'cleanup'):
                await broadcaster.cleanup()

        logger.info("信号广播器已停止")

    def add_subscriber(self, subscriber: Subscriber):
        """添加订阅者"""
        self.subscribers[subscriber.subscriber_id] = subscriber
        logger.info(f"添加订阅者: {subscriber.subscriber_id}")

    def remove_subscriber(self, subscriber_id: str):
        """移除订阅者"""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            logger.info(f"移除订阅者: {subscriber_id}")

    def get_subscriber(self, subscriber_id: str) -> Optional[Subscriber]:
        """获取订阅者"""
        return self.subscribers.get(subscriber_id)

    def update_subscriber(self, subscriber_id: str, **kwargs):
        """更新订阅者信息"""
        if subscriber_id in self.subscribers:
            subscriber = self.subscribers[subscriber_id]
            for key, value in kwargs.items():
                if hasattr(subscriber, key):
                    setattr(subscriber, key, value)

    async def broadcast_signal(self, signal: TradingSignal):
        """广播交易信号"""
        # 检查信号是否过期
        if signal.is_expired():
            logger.debug(f"信号已过期，跳过广播: {signal.signal_id}")
            return

        # 添加到队列
        try:
            await self.signal_queue.put(signal)
            self.stats['total_signals'] += 1

        except asyncio.QueueFull:
            logger.error("信号队列已满，丢弃信号")

    async def _broadcast_loop(self):
        """广播循环"""
        while self.is_running:
            try:
                # 从队列获取信号
                signal = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)

                # 处理信号广播
                await self._process_signal_broadcast(signal)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"广播循环错误: {e}")
                await asyncio.sleep(1)

    async def _process_signal_broadcast(self, signal: TradingSignal):
        """处理信号广播"""
        # 添加到历史
        self.signal_history.append(signal)

        # 查找匹配的订阅者
        matching_subscribers = []
        for subscriber in self.subscribers.values():
            if subscriber.matches_signal(signal):
                matching_subscribers.append(subscriber)

        if not matching_subscribers:
            logger.debug(f"没有订阅者匹配信号: {signal.signal_id}")
            return

        # 并发广播到所有匹配的订阅者
        broadcast_tasks = []

        for subscriber in matching_subscribers:
            for channel in subscriber.channels:
                if channel in self.broadcasters:
                    broadcaster = self.broadcasters[channel]
                    task = self._safe_broadcast(broadcaster, signal, subscriber)
                    broadcast_tasks.append(task)

        # 等待所有广播完成
        if broadcast_tasks:
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)

            # 统计结果
            for result in results:
                self.stats['total_broadcasts'] += 1
                if isinstance(result, Exception):
                    self.stats['failed_broadcasts'] += 1
                elif result:
                    self.stats['successful_broadcasts'] += 1
                else:
                    self.stats['failed_broadcasts'] += 1

        logger.info(f"信号广播完成: {signal.signal_id} -> {len(matching_subscribers)} 订阅者")

    async def _safe_broadcast(self, broadcaster, signal: TradingSignal, subscriber: Subscriber):
        """安全的广播操作"""
        try:
            success = await broadcaster.broadcast(signal, subscriber)

            if success:
                # 更新订阅者统计
                subscriber.signals_received += 1
                subscriber.last_signal_time = datetime.now()

            return success

        except Exception as e:
            logger.error(f"广播失败 {subscriber.subscriber_id}: {e}")
            return False

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                current_time = datetime.now()

                # 清理过期信号
                expired_signals = []
                for signal_id, signal in self.pending_signals.items():
                    if signal.is_expired():
                        expired_signals.append(signal_id)

                for signal_id in expired_signals:
                    del self.pending_signals[signal_id]

                if expired_signals:
                    logger.debug(f"清理 {len(expired_signals)} 个过期信号")

                await asyncio.sleep(300)  # 5分钟清理一次

            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(300)

    # 便利方法

    def create_websocket_subscriber(self, subscriber_id: str, client_id: str,
                                  symbols: Set[str] = None, min_confidence: float = 0.0) -> Subscriber:
        """创建WebSocket订阅者"""
        subscriber = Subscriber(
            subscriber_id=subscriber_id,
            channels={BroadcastChannel.WEBSOCKET},
            symbols=symbols or set(),
            min_confidence=min_confidence,
            websocket_client_id=client_id
        )

        self.add_subscriber(subscriber)
        return subscriber

    def create_webhook_subscriber(self, subscriber_id: str, webhook_url: str,
                                symbols: Set[str] = None, min_confidence: float = 0.0) -> Subscriber:
        """创建Webhook订阅者"""
        subscriber = Subscriber(
            subscriber_id=subscriber_id,
            channels={BroadcastChannel.HTTP_WEBHOOK},
            symbols=symbols or set(),
            min_confidence=min_confidence,
            webhook_url=webhook_url
        )

        self.add_subscriber(subscriber)
        return subscriber

    def get_signal_history(self, symbol: str = None, limit: int = 100) -> List[TradingSignal]:
        """获取信号历史"""
        signals = list(self.signal_history)

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return signals[-limit:]

    def get_subscriber_stats(self) -> Dict[str, Any]:
        """获取订阅者统计"""
        stats = {
            'total_subscribers': len(self.subscribers),
            'by_channel': defaultdict(int),
            'by_symbols': defaultdict(int),
            'active_subscribers': 0
        }

        for subscriber in self.subscribers.values():
            for channel in subscriber.channels:
                stats['by_channel'][channel.value] += 1

            for symbol in subscriber.symbols:
                stats['by_symbols'][symbol] += 1

            if subscriber.last_signal_time and \
               (datetime.now() - subscriber.last_signal_time).total_seconds() < 3600:  # 1小时内活跃
                stats['active_subscribers'] += 1

        return dict(stats)

    def get_stats(self) -> dict:
        """获取广播器统计信息"""
        uptime = datetime.now() - self.stats['start_time']

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime.total_seconds(),
            'stats': self.stats,
            'subscriber_stats': self.get_subscriber_stats(),
            'signal_queue_size': self.signal_queue.qsize(),
            'signal_history_size': len(self.signal_history)
        }

# 全局实例
_signal_broadcaster_instance = None

def get_signal_broadcaster() -> SignalBroadcaster:
    """获取信号广播器实例"""
    global _signal_broadcaster_instance
    if _signal_broadcaster_instance is None:
        _signal_broadcaster_instance = SignalBroadcaster()
    return _signal_broadcaster_instance
