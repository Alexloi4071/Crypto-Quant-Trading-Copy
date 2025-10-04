"""
Message Router and Dispatcher
消息路由和分发系统，提供高性能的消息路由、转发和处理功能
支持主题订阅、消息过滤和负载均衡分发
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any, Union, Set, Pattern
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import sys
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    BROADCAST = "broadcast"       # 广播消息
    UNICAST = "unicast"          # 单播消息
    MULTICAST = "multicast"      # 组播消息
    REQUEST = "request"          # 请求消息
    RESPONSE = "response"        # 响应消息
    NOTIFICATION = "notification" # 通知消息

class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Message:
    """消息数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.NOTIFICATION
    priority: MessagePriority = MessagePriority.NORMAL
    topic: str = ""
    sender: str = ""
    recipients: List[str] = field(default_factory=list)
    payload: Any = None
    headers: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type.value,
            'priority': self.priority.value,
            'topic': self.topic,
            'sender': self.sender,
            'recipients': self.recipients,
            'payload': self.payload,
            'headers': self.headers,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """从字典创建消息"""
        message = cls()
        message.id = data.get('id', message.id)
        message.type = MessageType(data.get('type', MessageType.NOTIFICATION.value))
        message.priority = MessagePriority(data.get('priority', MessagePriority.NORMAL.value))
        message.topic = data.get('topic', '')
        message.sender = data.get('sender', '')
        message.recipients = data.get('recipients', [])
        message.payload = data.get('payload')
        message.headers = data.get('headers', {})
        
        if data.get('timestamp'):
            message.timestamp = datetime.fromisoformat(data['timestamp'])
        if data.get('expires_at'):
            message.expires_at = datetime.fromisoformat(data['expires_at'])
            
        message.retry_count = data.get('retry_count', 0)
        message.max_retries = data.get('max_retries', 3)
        
        return message

@dataclass
class Subscription:
    """订阅信息"""
    subscriber_id: str
    topic_pattern: str
    callback: Optional[Callable] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    last_message_at: Optional[datetime] = None
    active: bool = True
    
    def matches_topic(self, topic: str) -> bool:
        """检查主题是否匹配"""
        # 支持通配符匹配
        pattern = self.topic_pattern.replace('*', '.*').replace('?', '.')
        return re.match(f'^{pattern}$', topic) is not None
    
    def matches_filters(self, message: Message) -> bool:
        """检查消息是否通过过滤器"""
        if not self.filters:
            return True
        
        for filter_key, filter_value in self.filters.items():
            if filter_key in message.headers:
                if message.headers[filter_key] != filter_value:
                    return False
            elif hasattr(message, filter_key):
                if getattr(message, filter_key) != filter_value:
                    return False
            else:
                return False  # 缺少必需的过滤字段
        
        return True

class MessageQueue:
    """优先级消息队列"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            MessagePriority.CRITICAL: deque(),
            MessagePriority.HIGH: deque(),
            MessagePriority.NORMAL: deque(),
            MessagePriority.LOW: deque()
        }
        self.size = 0
        self.lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'enqueued': 0,
            'dequeued': 0,
            'dropped': 0,
            'expired': 0
        }
    
    def enqueue(self, message: Message) -> bool:
        """入队消息"""
        with self.lock:
            if self.size >= self.max_size:
                # 队列满时，丢弃优先级最低的消息
                if not self._drop_low_priority_message():
                    self.stats['dropped'] += 1
                    return False
            
            if message.is_expired():
                self.stats['expired'] += 1
                return False
            
            self.queues[message.priority].append(message)
            self.size += 1
            self.stats['enqueued'] += 1
            return True
    
    def dequeue(self) -> Optional[Message]:
        """出队消息（优先级顺序）"""
        with self.lock:
            # 按优先级顺序检查队列
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                queue = self.queues[priority]
                
                while queue:
                    message = queue.popleft()
                    self.size -= 1
                    
                    if message.is_expired():
                        self.stats['expired'] += 1
                        continue
                    
                    self.stats['dequeued'] += 1
                    return message
            
            return None
    
    def _drop_low_priority_message(self) -> bool:
        """丢弃低优先级消息"""
        for priority in [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH]:
            queue = self.queues[priority]
            if queue:
                queue.popleft()
                self.size -= 1
                return True
        return False
    
    def get_size(self) -> int:
        """获取队列大小"""
        return self.size
    
    def get_stats(self) -> dict:
        """获取队列统计"""
        with self.lock:
            return {
                **self.stats,
                'current_size': self.size,
                'queue_sizes': {
                    priority.name: len(queue) 
                    for priority, queue in self.queues.items()
                }
            }
    
    def clear(self):
        """清空队列"""
        with self.lock:
            for queue in self.queues.values():
                queue.clear()
            self.size = 0

class MessageHandler:
    """消息处理器"""
    
    def __init__(self, handler_id: str, callback: Callable):
        self.handler_id = handler_id
        self.callback = callback
        self.processed_count = 0
        self.error_count = 0
        self.last_processed_at = None
        self.active = True
    
    async def handle_message(self, message: Message) -> bool:
        """处理消息"""
        if not self.active:
            return False
        
        try:
            result = await self.callback(message)
            self.processed_count += 1
            self.last_processed_at = datetime.now()
            return result is not False  # None或True都视为成功
            
        except Exception as e:
            logger.error(f"消息处理器 {self.handler_id} 处理失败: {e}")
            self.error_count += 1
            return False
    
    def get_stats(self) -> dict:
        """获取处理器统计"""
        return {
            'handler_id': self.handler_id,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'last_processed_at': self.last_processed_at.isoformat() if self.last_processed_at else None,
            'active': self.active
        }

class MessageRouter:
    """消息路由器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        
        # 订阅管理
        self.subscriptions = {}  # subscriber_id -> List[Subscription]
        self.topic_subscribers = defaultdict(set)  # topic -> Set[subscriber_id]
        
        # 消息处理器
        self.message_handlers = {}  # handler_id -> MessageHandler
        
        # 消息队列
        self.message_queue = MessageQueue()
        
        # 工作线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 路由表
        self.routes = {}  # topic_pattern -> List[handler_id]
        
        # 统计信息
        self.stats = {
            'messages_routed': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'subscribers_count': 0,
            'handlers_count': 0,
            'start_time': datetime.now()
        }
        
        # 任务管理
        self.processing_task = None
        self.cleanup_task = None
        self.is_running = False
        
        # 重试队列
        self.retry_queue = deque()
        
        logger.info("消息路由器初始化完成")
    
    async def start(self):
        """启动消息路由器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动消息处理任务
        self.processing_task = asyncio.create_task(self._message_processing_loop())
        
        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("消息路由器启动完成")
    
    async def stop(self):
        """停止消息路由器"""
        if not self.is_running:
            return
        
        logger.info("正在停止消息路由器...")
        self.is_running = False
        
        # 取消任务
        if self.processing_task:
            self.processing_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("消息路由器已停止")
    
    def subscribe(self, subscriber_id: str, topic_pattern: str, 
                 callback: Callable = None, filters: Dict[str, Any] = None) -> str:
        """订阅主题"""
        subscription = Subscription(
            subscriber_id=subscriber_id,
            topic_pattern=topic_pattern,
            callback=callback,
            filters=filters or {}
        )
        
        # 添加订阅
        if subscriber_id not in self.subscriptions:
            self.subscriptions[subscriber_id] = []
        
        self.subscriptions[subscriber_id].append(subscription)
        
        # 更新主题订阅者映射
        self.topic_subscribers[topic_pattern].add(subscriber_id)
        
        # 更新统计
        self.stats['subscribers_count'] = len(self.subscriptions)
        
        logger.debug(f"订阅者 {subscriber_id} 订阅主题 {topic_pattern}")
        return subscription.created_at.isoformat()
    
    def unsubscribe(self, subscriber_id: str, topic_pattern: str = None):
        """取消订阅"""
        if subscriber_id not in self.subscriptions:
            return
        
        if topic_pattern:
            # 取消特定主题订阅
            self.subscriptions[subscriber_id] = [
                sub for sub in self.subscriptions[subscriber_id]
                if sub.topic_pattern != topic_pattern
            ]
            
            self.topic_subscribers[topic_pattern].discard(subscriber_id)
            
            # 如果订阅者没有其他订阅，删除
            if not self.subscriptions[subscriber_id]:
                del self.subscriptions[subscriber_id]
        else:
            # 取消所有订阅
            for subscription in self.subscriptions[subscriber_id]:
                self.topic_subscribers[subscription.topic_pattern].discard(subscriber_id)
            
            del self.subscriptions[subscriber_id]
        
        # 更新统计
        self.stats['subscribers_count'] = len(self.subscriptions)
        
        logger.debug(f"订阅者 {subscriber_id} 取消订阅 {topic_pattern or 'all'}")
    
    def register_handler(self, handler_id: str, callback: Callable):
        """注册消息处理器"""
        handler = MessageHandler(handler_id, callback)
        self.message_handlers[handler_id] = handler
        
        # 更新统计
        self.stats['handlers_count'] = len(self.message_handlers)
        
        logger.debug(f"注册消息处理器: {handler_id}")
    
    def unregister_handler(self, handler_id: str):
        """注销消息处理器"""
        if handler_id in self.message_handlers:
            del self.message_handlers[handler_id]
            
            # 从路由表移除
            for topic_pattern, handlers in self.routes.items():
                if handler_id in handlers:
                    handlers.remove(handler_id)
            
            # 更新统计
            self.stats['handlers_count'] = len(self.message_handlers)
            
            logger.debug(f"注销消息处理器: {handler_id}")
    
    def add_route(self, topic_pattern: str, handler_id: str):
        """添加路由规则"""
        if topic_pattern not in self.routes:
            self.routes[topic_pattern] = []
        
        if handler_id not in self.routes[topic_pattern]:
            self.routes[topic_pattern].append(handler_id)
            logger.debug(f"添加路由: {topic_pattern} -> {handler_id}")
    
    def remove_route(self, topic_pattern: str, handler_id: str):
        """移除路由规则"""
        if topic_pattern in self.routes and handler_id in self.routes[topic_pattern]:
            self.routes[topic_pattern].remove(handler_id)
            
            if not self.routes[topic_pattern]:
                del self.routes[topic_pattern]
            
            logger.debug(f"移除路由: {topic_pattern} -> {handler_id}")
    
    async def route_message(self, message: Message) -> bool:
        """路由消息"""
        try:
            # 检查消息是否过期
            if message.is_expired():
                logger.debug(f"消息已过期: {message.id}")
                return False
            
            # 入队消息
            if not self.message_queue.enqueue(message):
                logger.warning(f"消息入队失败: {message.id}")
                self.stats['messages_failed'] += 1
                return False
            
            self.stats['messages_routed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"路由消息失败: {e}")
            self.stats['messages_failed'] += 1
            return False
    
    async def _message_processing_loop(self):
        """消息处理循环"""
        while self.is_running:
            try:
                # 处理重试队列
                await self._process_retry_queue()
                
                # 处理正常队列
                message = self.message_queue.dequeue()
                if message:
                    await self._process_message(message)
                else:
                    await asyncio.sleep(0.1)  # 没有消息时短暂休眠
                    
            except Exception as e:
                logger.error(f"消息处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Message) -> bool:
        """处理单个消息"""
        try:
            # 查找匹配的订阅者
            matching_subscribers = self._find_matching_subscribers(message)
            
            # 查找匹配的处理器
            matching_handlers = self._find_matching_handlers(message)
            
            delivery_success = False
            
            # 发送给订阅者
            for subscriber_id, subscriptions in matching_subscribers.items():
                for subscription in subscriptions:
                    try:
                        if subscription.callback:
                            await subscription.callback(message)
                            subscription.message_count += 1
                            subscription.last_message_at = datetime.now()
                            delivery_success = True
                    except Exception as e:
                        logger.error(f"订阅者回调错误 {subscriber_id}: {e}")
            
            # 发送给处理器
            for handler in matching_handlers:
                try:
                    if await handler.handle_message(message):
                        delivery_success = True
                except Exception as e:
                    logger.error(f"处理器处理错误 {handler.handler_id}: {e}")
            
            if delivery_success:
                self.stats['messages_delivered'] += 1
                return True
            else:
                # 没有成功投递，加入重试队列
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    self.retry_queue.append((message, datetime.now() + timedelta(seconds=5)))
                    logger.debug(f"消息加入重试队列: {message.id}")
                else:
                    logger.warning(f"消息重试次数超限，丢弃: {message.id}")
                    self.stats['messages_failed'] += 1
                
                return False
                
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            self.stats['messages_failed'] += 1
            return False
    
    async def _process_retry_queue(self):
        """处理重试队列"""
        current_time = datetime.now()
        
        # 处理所有到期的重试消息
        while self.retry_queue and self.retry_queue[0][1] <= current_time:
            message, retry_time = self.retry_queue.popleft()
            
            logger.debug(f"重试消息: {message.id} (第{message.retry_count}次)")
            await self._process_message(message)
    
    def _find_matching_subscribers(self, message: Message) -> Dict[str, List[Subscription]]:
        """查找匹配的订阅者"""
        matching = defaultdict(list)
        
        for subscriber_id, subscriptions in self.subscriptions.items():
            for subscription in subscriptions:
                if (subscription.active and 
                    subscription.matches_topic(message.topic) and 
                    subscription.matches_filters(message)):
                    matching[subscriber_id].append(subscription)
        
        return matching
    
    def _find_matching_handlers(self, message: Message) -> List[MessageHandler]:
        """查找匹配的处理器"""
        matching = []
        
        for topic_pattern, handler_ids in self.routes.items():
            # 检查主题模式是否匹配
            pattern = topic_pattern.replace('*', '.*').replace('?', '.')
            if re.match(f'^{pattern}$', message.topic):
                for handler_id in handler_ids:
                    if (handler_id in self.message_handlers and 
                        self.message_handlers[handler_id].active):
                        matching.append(self.message_handlers[handler_id])
        
        return matching
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(60)  # 每分钟清理一次
            except Exception as e:
                logger.error(f"清理任务错误: {e}")
                await asyncio.sleep(60)
    
    async def _perform_cleanup(self):
        """执行清理任务"""
        current_time = datetime.now()
        
        # 清理过期的重试消息
        while self.retry_queue and self.retry_queue[0][0].is_expired():
            message, retry_time = self.retry_queue.popleft()
            logger.debug(f"清理过期重试消息: {message.id}")
            self.stats['messages_failed'] += 1
        
        # 清理非活跃的订阅
        for subscriber_id in list(self.subscriptions.keys()):
            subscriptions = self.subscriptions[subscriber_id]
            active_subscriptions = [sub for sub in subscriptions if sub.active]
            
            if not active_subscriptions:
                del self.subscriptions[subscriber_id]
                logger.debug(f"清理非活跃订阅者: {subscriber_id}")
            else:
                self.subscriptions[subscriber_id] = active_subscriptions
        
        # 更新统计
        self.stats['subscribers_count'] = len(self.subscriptions)
    
    # 便利方法
    async def publish(self, topic: str, payload: Any, sender: str = "", 
                     priority: MessagePriority = MessagePriority.NORMAL,
                     headers: Dict[str, Any] = None, ttl_seconds: int = None) -> str:
        """发布消息"""
        message = Message(
            type=MessageType.BROADCAST,
            priority=priority,
            topic=topic,
            sender=sender,
            payload=payload,
            headers=headers or {}
        )
        
        if ttl_seconds:
            message.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        
        await self.route_message(message)
        return message.id
    
    async def send_request(self, topic: str, payload: Any, sender: str = "",
                          recipients: List[str] = None) -> str:
        """发送请求消息"""
        message = Message(
            type=MessageType.REQUEST,
            priority=MessagePriority.HIGH,
            topic=topic,
            sender=sender,
            recipients=recipients or [],
            payload=payload
        )
        
        await self.route_message(message)
        return message.id
    
    async def send_notification(self, topic: str, payload: Any, sender: str = "",
                              priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """发送通知消息"""
        message = Message(
            type=MessageType.NOTIFICATION,
            priority=priority,
            topic=topic,
            sender=sender,
            payload=payload
        )
        
        await self.route_message(message)
        return message.id
    
    def get_stats(self) -> dict:
        """获取路由器统计"""
        queue_stats = self.message_queue.get_stats()
        
        # 处理器统计
        handler_stats = {
            handler_id: handler.get_stats()
            for handler_id, handler in self.message_handlers.items()
        }
        
        # 订阅统计
        subscription_stats = {}
        for subscriber_id, subscriptions in self.subscriptions.items():
            subscription_stats[subscriber_id] = [
                {
                    'topic_pattern': sub.topic_pattern,
                    'message_count': sub.message_count,
                    'last_message_at': sub.last_message_at.isoformat() if sub.last_message_at else None,
                    'active': sub.active
                }
                for sub in subscriptions
            ]
        
        uptime = datetime.now() - self.stats['start_time']
        
        return {
            'router_stats': {
                **self.stats,
                'uptime_seconds': uptime.total_seconds(),
                'is_running': self.is_running,
                'retry_queue_size': len(self.retry_queue)
            },
            'queue_stats': queue_stats,
            'handler_stats': handler_stats,
            'subscription_stats': subscription_stats,
            'route_count': len(self.routes)
        }

# 全局实例
_message_router_instance = None

def get_message_router() -> MessageRouter:
    """获取消息路由器实例"""
    global _message_router_instance
    if _message_router_instance is None:
        _message_router_instance = MessageRouter()
    return _message_router_instance