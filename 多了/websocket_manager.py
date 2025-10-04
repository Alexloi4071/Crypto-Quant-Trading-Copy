"""
WebSocket Connection Manager
WebSocket连接管理器，提供高性能的WebSocket连接池和管理功能
支持自动重连、负载均衡和连接监控
"""

import asyncio
import websockets
import json
import time
import ssl
from typing import Dict, List, Optional, Callable, Any, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import sys
from pathlib import Path
import weakref
from concurrent.futures import ThreadPoolExecutor
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class ConnectionStats:
    """连接统计信息"""
    connection_id: str
    url: str
    state: ConnectionState
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    reconnect_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_ping_time: Optional[datetime] = None
    last_pong_time: Optional[datetime] = None
    ping_latency_ms: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None

class WebSocketConnection:
    """WebSocket连接封装类"""
    
    def __init__(self, connection_id: str, url: str, config: dict = None):
        self.connection_id = connection_id
        self.url = url
        self.config = config or {}
        
        # 连接对象
        self.websocket = None
        self.state = ConnectionState.DISCONNECTED
        
        # 统计信息
        self.stats = ConnectionStats(connection_id, url, self.state)
        
        # 重连配置
        self.max_reconnect_attempts = self.config.get('max_reconnect_attempts', 5)
        self.reconnect_delay = self.config.get('reconnect_delay', 1.0)
        self.reconnect_backoff = self.config.get('reconnect_backoff', 2.0)
        self.max_reconnect_delay = self.config.get('max_reconnect_delay', 60.0)
        
        # 心跳配置
        self.ping_interval = self.config.get('ping_interval', 30)
        self.ping_timeout = self.config.get('ping_timeout', 10)
        
        # 消息队列
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.subscription_queue = deque()
        
        # 回调函数
        self.on_message_callbacks = []
        self.on_connect_callbacks = []
        self.on_disconnect_callbacks = []
        self.on_error_callbacks = []
        
        # 任务
        self.connection_task = None
        self.message_handler_task = None
        self.ping_task = None
        
        # 锁
        self.lock = asyncio.Lock()
        
        logger.debug(f"WebSocket连接初始化: {connection_id} -> {url}")
    
    async def connect(self):
        """建立WebSocket连接"""
        async with self.lock:
            if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
                logger.debug(f"连接 {self.connection_id} 已连接或正在连接")
                return True
            
            self.state = ConnectionState.CONNECTING
            self.stats.state = self.state
            
            try:
                # 设置SSL上下文（如果需要）
                ssl_context = None
                if self.url.startswith('wss://'):
                    ssl_context = ssl.create_default_context()
                    if self.config.get('ssl_verify', True) is False:
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_NONE
                
                # 建立连接
                extra_headers = self.config.get('headers', {})
                
                self.websocket = await websockets.connect(
                    self.url,
                    ssl=ssl_context,
                    extra_headers=extra_headers,
                    ping_interval=None,  # 我们自己处理心跳
                    ping_timeout=None,
                    close_timeout=10
                )
                
                self.state = ConnectionState.CONNECTED
                self.stats.state = self.state
                self.stats.connected_at = datetime.now()
                self.stats.reconnect_count = 0
                
                # 启动消息处理任务
                self.message_handler_task = asyncio.create_task(self._message_handler_loop())
                
                # 启动心跳任务
                if self.ping_interval > 0:
                    self.ping_task = asyncio.create_task(self._ping_loop())
                
                # 重新订阅
                await self._resubscribe()
                
                # 调用连接回调
                for callback in self.on_connect_callbacks:
                    try:
                        await callback(self.connection_id)
                    except Exception as e:
                        logger.error(f"连接回调错误: {e}")
                
                logger.info(f"WebSocket连接成功: {self.connection_id}")
                return True
                
            except Exception as e:
                self.state = ConnectionState.ERROR
                self.stats.state = self.state
                self.stats.error_count += 1
                self.stats.last_error = str(e)
                
                logger.error(f"WebSocket连接失败 {self.connection_id}: {e}")
                
                # 调用错误回调
                for callback in self.on_error_callbacks:
                    try:
                        await callback(self.connection_id, e)
                    except Exception as cb_e:
                        logger.error(f"错误回调异常: {cb_e}")
                
                return False
    
    async def disconnect(self):
        """断开WebSocket连接"""
        async with self.lock:
            if self.state in [ConnectionState.DISCONNECTED, ConnectionState.CLOSING, ConnectionState.CLOSED]:
                return
            
            logger.info(f"断开WebSocket连接: {self.connection_id}")
            self.state = ConnectionState.CLOSING
            self.stats.state = self.state
            
            # 取消任务
            if self.message_handler_task:
                self.message_handler_task.cancel()
            
            if self.ping_task:
                self.ping_task.cancel()
            
            # 关闭连接
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception as e:
                    logger.warning(f"关闭WebSocket时出错: {e}")
            
            self.state = ConnectionState.CLOSED
            self.stats.state = self.state
            self.stats.disconnected_at = datetime.now()
            
            # 调用断开连接回调
            for callback in self.on_disconnect_callbacks:
                try:
                    await callback(self.connection_id)
                except Exception as e:
                    logger.error(f"断开连接回调错误: {e}")
    
    async def send_message(self, message: Union[str, dict]) -> bool:
        """发送消息"""
        if self.state != ConnectionState.CONNECTED or not self.websocket:
            logger.warning(f"连接 {self.connection_id} 未连接，消息入队")
            
            try:
                await self.message_queue.put(message)
                return True
            except asyncio.QueueFull:
                logger.error(f"消息队列已满，丢弃消息: {self.connection_id}")
                return False
        
        try:
            # 转换消息格式
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = str(message)
            
            await self.websocket.send(message_str)
            
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(message_str)
            
            return True
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"连接 {self.connection_id} 已关闭")
            self.state = ConnectionState.DISCONNECTED
            self.stats.state = self.state
            return False
            
        except Exception as e:
            logger.error(f"发送消息失败 {self.connection_id}: {e}")
            self.stats.error_count += 1
            self.stats.last_error = str(e)
            return False
    
    async def subscribe(self, subscription: dict):
        """添加订阅"""
        self.subscription_queue.append(subscription)
        
        if self.state == ConnectionState.CONNECTED:
            await self.send_message(subscription)
    
    async def _resubscribe(self):
        """重新订阅"""
        for subscription in self.subscription_queue:
            await self.send_message(subscription)
    
    async def _message_handler_loop(self):
        """消息处理循环"""
        try:
            while self.state == ConnectionState.CONNECTED and self.websocket:
                try:
                    # 处理接收到的消息
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                    
                    self.stats.messages_received += 1
                    self.stats.bytes_received += len(message)
                    
                    # 尝试解析JSON
                    try:
                        message_data = json.loads(message)
                    except json.JSONDecodeError:
                        message_data = message
                    
                    # 调用消息回调
                    for callback in self.on_message_callbacks:
                        try:
                            await callback(self.connection_id, message_data)
                        except Exception as e:
                            logger.error(f"消息回调错误: {e}")
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"连接 {self.connection_id} 已关闭")
                    self.state = ConnectionState.DISCONNECTED
                    break
                except Exception as e:
                    logger.error(f"消息处理错误 {self.connection_id}: {e}")
                    self.stats.error_count += 1
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"消息处理循环异常 {self.connection_id}: {e}")
    
    async def _ping_loop(self):
        """心跳循环"""
        try:
            while self.state == ConnectionState.CONNECTED and self.websocket:
                try:
                    ping_start = time.time()
                    self.stats.last_ping_time = datetime.now()
                    
                    pong_waiter = await self.websocket.ping()
                    await asyncio.wait_for(pong_waiter, timeout=self.ping_timeout)
                    
                    ping_end = time.time()
                    self.stats.last_pong_time = datetime.now()
                    self.stats.ping_latency_ms = (ping_end - ping_start) * 1000
                    
                    await asyncio.sleep(self.ping_interval)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"心跳超时 {self.connection_id}")
                    self.state = ConnectionState.ERROR
                    break
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"连接已关闭，停止心跳 {self.connection_id}")
                    break
                except Exception as e:
                    logger.error(f"心跳错误 {self.connection_id}: {e}")
                    break
        
        except asyncio.CancelledError:
            pass
    
    async def reconnect(self):
        """重连"""
        if self.stats.reconnect_count >= self.max_reconnect_attempts:
            logger.error(f"连接 {self.connection_id} 重连次数超限")
            return False
        
        self.state = ConnectionState.RECONNECTING
        self.stats.state = self.state
        self.stats.reconnect_count += 1
        
        # 计算重连延迟
        delay = min(
            self.reconnect_delay * (self.reconnect_backoff ** (self.stats.reconnect_count - 1)),
            self.max_reconnect_delay
        )
        
        logger.info(f"连接 {self.connection_id} 将在 {delay:.1f}s 后重连 (第{self.stats.reconnect_count}次)")
        await asyncio.sleep(delay)
        
        return await self.connect()
    
    # 回调函数注册方法
    def on_message(self, callback: Callable):
        """注册消息回调"""
        self.on_message_callbacks.append(callback)
    
    def on_connect(self, callback: Callable):
        """注册连接回调"""
        self.on_connect_callbacks.append(callback)
    
    def on_disconnect(self, callback: Callable):
        """注册断开连接回调"""
        self.on_disconnect_callbacks.append(callback)
    
    def on_error(self, callback: Callable):
        """注册错误回调"""
        self.on_error_callbacks.append(callback)
    
    def get_stats(self) -> ConnectionStats:
        """获取连接统计"""
        return self.stats

class WebSocketConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.connections = {}  # connection_id -> WebSocketConnection
        self.url_connections = defaultdict(list)  # url -> [connection_ids]
        
        # 连接池配置
        self.pool_config = {
            'min_connections_per_url': 1,
            'max_connections_per_url': 10,
            'connection_timeout': 30,
            'idle_timeout': 300
        }
        
        # 统计信息
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'failed_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'reconnect_attempts': 0
        }
        
        # 管理任务
        self.health_check_task = None
        self.cleanup_task = None
        self.is_running = False
        
        # 负载均衡
        self.load_balancer = {}  # url -> next_connection_index
        
        # 回调函数
        self.global_callbacks = {
            'on_message': [],
            'on_connect': [],
            'on_disconnect': [],
            'on_error': []
        }
        
        logger.info("WebSocket连接管理器初始化完成")
    
    async def start(self):
        """启动连接管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动管理任务
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("WebSocket连接管理器启动完成")
    
    async def stop(self):
        """停止连接管理器"""
        if not self.is_running:
            return
        
        logger.info("正在停止WebSocket连接管理器...")
        self.is_running = False
        
        # 取消管理任务
        if self.health_check_task:
            self.health_check_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # 断开所有连接
        for connection in list(self.connections.values()):
            await connection.disconnect()
        
        self.connections.clear()
        self.url_connections.clear()
        
        logger.info("WebSocket连接管理器已停止")
    
    async def create_connection(self, url: str, connection_id: str = None, config: dict = None) -> str:
        """创建WebSocket连接"""
        if len(self.connections) >= self.max_connections:
            raise Exception(f"连接数已达上限: {self.max_connections}")
        
        # 生成连接ID
        if not connection_id:
            connection_id = f"conn_{int(time.time() * 1000)}"
        
        if connection_id in self.connections:
            raise Exception(f"连接ID已存在: {connection_id}")
        
        # 创建连接对象
        connection = WebSocketConnection(connection_id, url, config)
        
        # 注册全局回调
        for callback in self.global_callbacks['on_message']:
            connection.on_message(callback)
        
        for callback in self.global_callbacks['on_connect']:
            connection.on_connect(callback)
        
        for callback in self.global_callbacks['on_disconnect']:
            connection.on_disconnect(callback)
        
        for callback in self.global_callbacks['on_error']:
            connection.on_error(callback)
        
        # 添加到管理器
        self.connections[connection_id] = connection
        self.url_connections[url].append(connection_id)
        self.stats['total_connections'] += 1
        
        # 尝试连接
        if await connection.connect():
            self.stats['active_connections'] += 1
        else:
            self.stats['failed_connections'] += 1
        
        logger.info(f"创建WebSocket连接: {connection_id} -> {url}")
        return connection_id
    
    async def remove_connection(self, connection_id: str):
        """移除连接"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        # 断开连接
        await connection.disconnect()
        
        # 从管理器移除
        del self.connections[connection_id]
        
        # 从URL连接列表移除
        url = connection.url
        if connection_id in self.url_connections[url]:
            self.url_connections[url].remove(connection_id)
        
        if connection.state == ConnectionState.CONNECTED:
            self.stats['active_connections'] -= 1
        
        logger.info(f"移除WebSocket连接: {connection_id}")
    
    async def send_message(self, connection_id: str, message: Union[str, dict]) -> bool:
        """发送消息到指定连接"""
        if connection_id not in self.connections:
            logger.error(f"连接不存在: {connection_id}")
            return False
        
        connection = self.connections[connection_id]
        success = await connection.send_message(message)
        
        if success:
            self.stats['messages_sent'] += 1
        
        return success
    
    async def broadcast_message(self, url: str, message: Union[str, dict]) -> int:
        """广播消息到指定URL的所有连接"""
        if url not in self.url_connections:
            return 0
        
        success_count = 0
        connection_ids = self.url_connections[url].copy()
        
        for connection_id in connection_ids:
            if await self.send_message(connection_id, message):
                success_count += 1
        
        return success_count
    
    async def broadcast_message_all(self, message: Union[str, dict]) -> int:
        """广播消息到所有连接"""
        success_count = 0
        
        for connection_id in list(self.connections.keys()):
            if await self.send_message(connection_id, message):
                success_count += 1
        
        return success_count
    
    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """获取连接对象"""
        return self.connections.get(connection_id)
    
    def get_connections_by_url(self, url: str) -> List[WebSocketConnection]:
        """获取指定URL的所有连接"""
        connection_ids = self.url_connections.get(url, [])
        return [self.connections[cid] for cid in connection_ids if cid in self.connections]
    
    def get_active_connection(self, url: str) -> Optional[WebSocketConnection]:
        """获取指定URL的活跃连接（负载均衡）"""
        connections = self.get_connections_by_url(url)
        active_connections = [conn for conn in connections if conn.state == ConnectionState.CONNECTED]
        
        if not active_connections:
            return None
        
        # 简单的轮询负载均衡
        if url not in self.load_balancer:
            self.load_balancer[url] = 0
        
        index = self.load_balancer[url] % len(active_connections)
        self.load_balancer[url] = (self.load_balancer[url] + 1) % len(active_connections)
        
        return active_connections[index]
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                await asyncio.sleep(30)
    
    async def _perform_health_check(self):
        """执行健康检查"""
        # 检查连接状态并尝试重连
        for connection_id, connection in list(self.connections.items()):
            if connection.state == ConnectionState.ERROR:
                logger.info(f"尝试重连错误连接: {connection_id}")
                await connection.reconnect()
            
            elif connection.state == ConnectionState.DISCONNECTED:
                # 自动重连断开的连接
                if connection.stats.reconnect_count < connection.max_reconnect_attempts:
                    logger.info(f"尝试重连断开连接: {connection_id}")
                    await connection.reconnect()
        
        # 更新统计信息
        active_count = sum(1 for conn in self.connections.values() 
                         if conn.state == ConnectionState.CONNECTED)
        self.stats['active_connections'] = active_count
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(60)  # 每分钟清理一次
            except Exception as e:
                logger.error(f"清理错误: {e}")
                await asyncio.sleep(60)
    
    async def _perform_cleanup(self):
        """执行清理"""
        current_time = datetime.now()
        
        # 清理长时间失败的连接
        for connection_id, connection in list(self.connections.items()):
            if (connection.state in [ConnectionState.ERROR, ConnectionState.CLOSED] and
                connection.stats.reconnect_count >= connection.max_reconnect_attempts):
                
                logger.info(f"清理失败连接: {connection_id}")
                await self.remove_connection(connection_id)
        
        # 更新全局统计
        total_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
        for connection in self.connections.values():
            stats = connection.get_stats()
            total_stats['messages_sent'] += stats.messages_sent
            total_stats['messages_received'] += stats.messages_received
            total_stats['bytes_sent'] += stats.bytes_sent
            total_stats['bytes_received'] += stats.bytes_received
        
        self.stats.update(total_stats)
    
    # 全局回调注册
    def on_message(self, callback: Callable):
        """注册全局消息回调"""
        self.global_callbacks['on_message'].append(callback)
    
    def on_connect(self, callback: Callable):
        """注册全局连接回调"""
        self.global_callbacks['on_connect'].append(callback)
    
    def on_disconnect(self, callback: Callable):
        """注册全局断开连接回调"""
        self.global_callbacks['on_disconnect'].append(callback)
    
    def on_error(self, callback: Callable):
        """注册全局错误回调"""
        self.global_callbacks['on_error'].append(callback)
    
    def get_stats(self) -> dict:
        """获取管理器统计信息"""
        connection_stats = {
            connection_id: {
                'url': conn.url,
                'state': conn.state.value,
                'stats': asdict(conn.get_stats())
            }
            for connection_id, conn in self.connections.items()
        }
        
        return {
            'manager_stats': self.stats,
            'total_connections': len(self.connections),
            'connections_by_url': {url: len(conn_ids) for url, conn_ids in self.url_connections.items()},
            'connection_details': connection_stats
        }
    
    def get_health_summary(self) -> dict:
        """获取健康状态摘要"""
        state_counts = defaultdict(int)
        for connection in self.connections.values():
            state_counts[connection.state.value] += 1
        
        return {
            'total_connections': len(self.connections),
            'state_distribution': dict(state_counts),
            'is_running': self.is_running,
            'health_score': self._calculate_health_score()
        }
    
    def _calculate_health_score(self) -> float:
        """计算健康分数"""
        if not self.connections:
            return 100.0
        
        connected_count = sum(1 for conn in self.connections.values() 
                            if conn.state == ConnectionState.CONNECTED)
        
        return (connected_count / len(self.connections)) * 100

# 全局实例
_ws_manager_instance = None

def get_websocket_manager() -> WebSocketConnectionManager:
    """获取WebSocket连接管理器实例"""
    global _ws_manager_instance
    if _ws_manager_instance is None:
        _ws_manager_instance = WebSocketConnectionManager()
    return _ws_manager_instance