"""
WebSocket Server
WebSocket服务器，提供实时数据推送和双向通信能力
支持多客户端连接、消息广播和订阅管理
"""

import asyncio
import websockets
import json
import time
import uuid
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import ssl
import logging

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class MessageType(Enum):
    """消息类型枚举"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    AUTH = "auth"
    NOTIFICATION = "notification"

@dataclass

class WebSocketClient:
    """WebSocket客户端信息"""
    client_id: str
    websocket: websockets.WebSocketServerProtocol
    connected_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    subscriptions: Set[str] = field(default_factory=set)
    authenticated: bool = False
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def is_alive(self, timeout: int = 60) -> bool:
        """检查客户端是否存活"""
        return (datetime.now() - self.last_heartbeat).total_seconds() < timeout

@dataclass

class WebSocketMessage:
    """WebSocket消息结构"""
    type: MessageType
    topic: Optional[str] = None
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'type': self.type.value,
            'topic': self.topic,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'client_id': self.client_id
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())

class WebSocketServer:
    """WebSocket服务器主类"""

    def __init__(self, host: str = "localhost", port: int = 8765,
                 ssl_context: ssl.SSLContext = None):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context

        # 客户端管理
        self.clients: Dict[str, WebSocketClient] = {}
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)

        # 消息队列和历史
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.message_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 服务器状态
        self.server = None
        self.is_running = False
        self.start_time = datetime.now()

        # 统计信息
        self.stats = {
            'total_connections': 0,
            'current_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0
        }

        # 任务管理
        self.message_processor_task = None
        self.heartbeat_task = None
        self.cleanup_task = None

        # 回调函数
        self.on_connect_callbacks: List[Callable] = []
        self.on_disconnect_callbacks: List[Callable] = []
        self.on_message_callbacks: List[Callable] = []
        self.on_subscribe_callbacks: List[Callable] = []

        # 认证配置
        self.auth_required = config.get('websocket', {}).get('auth_required', False)
        self.auth_timeout = config.get('websocket', {}).get('auth_timeout', 30)

        logger.info(f"WebSocket服务器初始化: {host}:{port}")

    async def start(self):
        """启动WebSocket服务器"""
        if self.is_running:
            return

        try:
            # 启动消息处理任务
            self.message_processor_task = asyncio.create_task(self._message_processor())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self.cleanup_task = asyncio.create_task(self._cleanup_task())

            # 启动WebSocket服务器
            self.server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
                ssl=self.ssl_context,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )

            self.is_running = True
            logger.info(f"WebSocket服务器启动成功: {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"启动WebSocket服务器失败: {e}")
            raise

    async def stop(self):
        """停止WebSocket服务器"""
        if not self.is_running:
            return

        logger.info("正在停止WebSocket服务器...")
        self.is_running = False

        # 取消任务
        if self.message_processor_task:
            self.message_processor_task.cancel()

        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.cleanup_task:
            self.cleanup_task.cancel()

        # 关闭所有客户端连接
        if self.clients:
            disconnect_tasks = []
            for client in self.clients.values():
                disconnect_tasks.append(client.websocket.close())

            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        # 关闭服务器
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        logger.info("WebSocket服务器已停止")

    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        client_id = str(uuid.uuid4())
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"

        # 创建客户端对象
        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            ip_address=client_ip
        )

        self.clients[client_id] = client
        self.stats['total_connections'] += 1
        self.stats['current_connections'] += 1

        logger.info(f"新客户端连接: {client_id} ({client_ip})")

        # 调用连接回调
        for callback in self.on_connect_callbacks:
            try:
                await callback(client)
            except Exception as e:
                logger.error(f"连接回调错误: {e}")

        try:
            # 发送欢迎消息
            welcome_msg = WebSocketMessage(
                type=MessageType.NOTIFICATION,
                data={
                    'message': 'Connected successfully',
                    'client_id': client_id,
                    'server_time': datetime.now().isoformat()
                }
            )
            await self._send_to_client(client, welcome_msg)

            # 处理客户端消息
            async for message in websocket:
                try:
                    await self._handle_message(client, message)
                except Exception as e:
                    logger.error(f"处理客户端消息失败 {client_id}: {e}")
                    await self._send_error(client, f"Message processing error: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端连接关闭: {client_id}")
        except Exception as e:
            logger.error(f"客户端处理错误 {client_id}: {e}")
        finally:
            await self._cleanup_client(client_id)

    async def _handle_message(self, client: WebSocketClient, raw_message: str):
        """处理客户端消息"""
        try:
            data = json.loads(raw_message)
            message_type = MessageType(data.get('type', 'data'))

            # 更新统计
            self.stats['messages_received'] += 1
            self.stats['bytes_received'] += len(raw_message)
            client.last_heartbeat = datetime.now()

            # 根据消息类型处理
            if message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(client, data)
            elif message_type == MessageType.AUTH:
                await self._handle_auth(client, data)
            elif message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(client, data)
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(client, data)
            else:
                # 调用消息回调
                for callback in self.on_message_callbacks:
                    try:
                        await callback(client, data)
                    except Exception as e:
                        logger.error(f"消息回调错误: {e}")

        except json.JSONDecodeError:
            await self._send_error(client, "Invalid JSON format")
        except ValueError as e:
            await self._send_error(client, f"Invalid message type: {str(e)}")
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self._send_error(client, "Internal server error")

    async def _handle_heartbeat(self, client: WebSocketClient, data: dict):
        """处理心跳消息"""
        response = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={'server_time': datetime.now().isoformat()},
            client_id=client.client_id
        )
        await self._send_to_client(client, response)

    async def _handle_auth(self, client: WebSocketClient, data: dict):
        """处理认证消息"""
        token = data.get('token')

        if not token:
            await self._send_error(client, "Authentication token required")
            return

        # 简化的认证逻辑 - 实际应该验证JWT token
        try:
            # 这里应该调用实际的认证服务
            client.authenticated = True
            client.user_id = data.get('user_id', 'anonymous')

            response = WebSocketMessage(
                type=MessageType.AUTH,
                data={'status': 'authenticated', 'user_id': client.user_id},
                client_id=client.client_id
            )
            await self._send_to_client(client, response)

            logger.info(f"客户端认证成功: {client.client_id} ({client.user_id})")

        except Exception as e:
            await self._send_error(client, f"Authentication failed: {str(e)}")

    async def _handle_subscribe(self, client: WebSocketClient, data: dict):
        """处理订阅消息"""
        topics = data.get('topics', [])

        if not isinstance(topics, list):
            await self._send_error(client, "Topics must be a list")
            return

        for topic in topics:
            if isinstance(topic, str):
                client.subscriptions.add(topic)
                self.topic_subscribers[topic].add(client.client_id)

                # 发送历史消息（如果有）
                if topic in self.message_history:
                    for historical_msg in list(self.message_history[topic])[-10:]:  # 最近10条
                        await self._send_to_client(client, historical_msg)

        response = WebSocketMessage(
            type=MessageType.SUBSCRIBE,
            data={'subscribed_topics': list(client.subscriptions)},
            client_id=client.client_id
        )
        await self._send_to_client(client, response)

        # 调用订阅回调
        for callback in self.on_subscribe_callbacks:
            try:
                await callback(client, topics)
            except Exception as e:
                logger.error(f"订阅回调错误: {e}")

        logger.debug(f"客户端订阅: {client.client_id} -> {topics}")

    async def _handle_unsubscribe(self, client: WebSocketClient, data: dict):
        """处理取消订阅消息"""
        topics = data.get('topics', [])

        for topic in topics:
            if isinstance(topic, str):
                client.subscriptions.discard(topic)
                if client.client_id in self.topic_subscribers[topic]:
                    self.topic_subscribers[topic].remove(client.client_id)

        response = WebSocketMessage(
            type=MessageType.UNSUBSCRIBE,
            data={'unsubscribed_topics': topics},
            client_id=client.client_id
        )
        await self._send_to_client(client, response)

        logger.debug(f"客户端取消订阅: {client.client_id} -> {topics}")

    async def _send_to_client(self, client: WebSocketClient, message: WebSocketMessage):
        """发送消息到客户端"""
        try:
            json_message = message.to_json()
            await client.websocket.send(json_message)

            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += len(json_message)

        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"客户端连接已关闭: {client.client_id}")
        except Exception as e:
            logger.error(f"发送消息失败 {client.client_id}: {e}")
            self.stats['errors'] += 1

    async def _send_error(self, client: WebSocketClient, error_message: str):
        """发送错误消息到客户端"""
        error_msg = WebSocketMessage(
            type=MessageType.ERROR,
            data={'error': error_message},
            client_id=client.client_id
        )
        await self._send_to_client(client, error_msg)

    async def _cleanup_client(self, client_id: str):
        """清理客户端连接"""
        if client_id not in self.clients:
            return

        client = self.clients[client_id]

        # 从订阅中移除
        for topic in client.subscriptions:
            self.topic_subscribers[topic].discard(client_id)

        # 调用断开连接回调
        for callback in self.on_disconnect_callbacks:
            try:
                await callback(client)
            except Exception as e:
                logger.error(f"断开连接回调错误: {e}")

        # 移除客户端
        del self.clients[client_id]
        self.stats['current_connections'] -= 1

        logger.info(f"客户端已清理: {client_id}")

    async def _message_processor(self):
        """消息处理器"""
        while self.is_running:
            try:
                # 从队列获取消息
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._process_queued_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"消息处理器错误: {e}")
                await asyncio.sleep(1)

    async def _process_queued_message(self, message: WebSocketMessage):
        """处理队列中的消息"""
        if message.topic:
            # 广播到订阅该主题的客户端
            subscribers = self.topic_subscribers.get(message.topic, set())

            send_tasks = []
            for client_id in subscribers:
                if client_id in self.clients:
                    client = self.clients[client_id]
                    send_tasks.append(self._send_to_client(client, message))

            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

            # 保存到历史
            self.message_history[message.topic].append(message)
        else:
            # 广播到所有客户端
            send_tasks = []
            for client in self.clients.values():
                send_tasks.append(self._send_to_client(client, message))

            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)

    async def _heartbeat_monitor(self):
        """心跳监控"""
        while self.is_running:
            try:
                current_time = datetime.now()
                timeout_clients = []

                for client_id, client in self.clients.items():
                    if not client.is_alive():
                        timeout_clients.append(client_id)

                # 清理超时客户端
                for client_id in timeout_clients:
                    logger.info(f"清理超时客户端: {client_id}")
                    await self._cleanup_client(client_id)

                await asyncio.sleep(30)  # 30秒检查一次

            except Exception as e:
                logger.error(f"心跳监控错误: {e}")
                await asyncio.sleep(30)

    async def _cleanup_task(self):
        """定期清理任务"""
        while self.is_running:
            try:
                # 清理空的订阅主题
                empty_topics = []
                for topic, subscribers in self.topic_subscribers.items():
                    if not subscribers:
                        empty_topics.append(topic)

                for topic in empty_topics:
                    del self.topic_subscribers[topic]
                    if topic in self.message_history:
                        del self.message_history[topic]

                await asyncio.sleep(300)  # 5分钟清理一次

            except Exception as e:
                logger.error(f"清理任务错误: {e}")
                await asyncio.sleep(300)

    # 公共接口方法

    async def broadcast_message(self, topic: str, data: Any):
        """广播消息到主题"""
        message = WebSocketMessage(
            type=MessageType.DATA,
            topic=topic,
            data=data
        )
        await self.message_queue.put(message)

    async def send_to_user(self, user_id: str, data: Any):
        """发送消息到特定用户"""
        message = WebSocketMessage(
            type=MessageType.DATA,
            data=data
        )

        for client in self.clients.values():
            if client.user_id == user_id:
                await self._send_to_client(client, message)

    def get_connected_clients(self) -> List[Dict[str, Any]]:
        """获取连接的客户端列表"""
        return [
            {
                'client_id': client.client_id,
                'user_id': client.user_id,
                'ip_address': client.ip_address,
                'connected_at': client.connected_at.isoformat(),
                'subscriptions': list(client.subscriptions),
                'authenticated': client.authenticated
            }
            for client in self.clients.values()
        ]

    def get_topic_subscribers(self, topic: str) -> List[str]:
        """获取主题订阅者列表"""
        return list(self.topic_subscribers.get(topic, set()))

    def get_stats(self) -> dict:
        """获取服务器统计信息"""
        uptime = datetime.now() - self.start_time

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime.total_seconds(),
            'host': self.host,
            'port': self.port,
            'stats': self.stats,
            'topics': {
                'total_topics': len(self.topic_subscribers),
                'active_topics': len([t for t, s in self.topic_subscribers.items() if s]),
                'topics_list': list(self.topic_subscribers.keys())
            }
        }

    # 回调注册方法

    def on_connect(self, callback: Callable):
        """注册连接回调"""
        self.on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable):
        """注册断开连接回调"""
        self.on_disconnect_callbacks.append(callback)

    def on_message(self, callback: Callable):
        """注册消息回调"""
        self.on_message_callbacks.append(callback)

    def on_subscribe(self, callback: Callable):
        """注册订阅回调"""
        self.on_subscribe_callbacks.append(callback)

# 全局实例
_websocket_server_instance = None

def get_websocket_server() -> WebSocketServer:
    """获取WebSocket服务器实例"""
    global _websocket_server_instance
    if _websocket_server_instance is None:
        ws_config = config.get('websocket', {})
        _websocket_server_instance = WebSocketServer(
            host=ws_config.get('host', 'localhost'),
            port=ws_config.get('port', 8765)
        )
    return _websocket_server_instance
