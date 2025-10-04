"""
Connection Pool Manager
连接池管理器，提供数据库、HTTP和WebSocket连接的统一管理
支持连接复用、负载均衡和故障恢复
"""

import asyncio
import aiohttp
import aiomysql
import aioredis
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from collections import deque
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

class ConnectionStatus(Enum):
    """连接状态枚举"""
    IDLE = "idle"
    ACTIVE = "active"
    BROKEN = "broken"
    CLOSED = "closed"

@dataclass

class ConnectionInfo:
    """连接信息"""
    connection_id: str
    connection_type: str
    status: ConnectionStatus
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    max_lifetime: Optional[int] = None  # 秒

    def is_expired(self) -> bool:
        """检查连接是否过期"""
        if not self.max_lifetime:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.max_lifetime

    def touch(self):
        """更新使用时间"""
        self.last_used = datetime.now()
        self.usage_count += 1

class BaseConnectionPool:
    """基础连接池"""

    def __init__(self, name: str, min_connections: int = 1,
                 max_connections: int = 10, max_lifetime: int = 3600):
        self.name = name
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.max_lifetime = max_lifetime

        # 连接池
        self.idle_connections = deque()
        self.active_connections = {}
        self.broken_connections = set()

        # 统计信息
        self.stats = {
            'total_created': 0,
            'total_closed': 0,
            'current_active': 0,
            'current_idle': 0,
            'connection_errors': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }

        # 锁
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # 任务
        self.cleanup_task = None
        self.is_running = False

    async def start(self):
        """启动连接池"""
        if self.is_running:
            return

        self.is_running = True

        # 预创建最小连接数
        await self._ensure_min_connections()

        # 启动清理任务
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(f"连接池 {self.name} 启动完成")

    async def stop(self):
        """停止连接池"""
        if not self.is_running:
            return

        self.is_running = False

        # 取消清理任务
        if self.cleanup_task:
            self.cleanup_task.cancel()

        # 关闭所有连接
        await self._close_all_connections()

        logger.info(f"连接池 {self.name} 已停止")

    async def acquire(self) -> Any:
        """获取连接"""
        async with self._condition:
            # 检查空闲连接
            while self.idle_connections:
                conn_info = self.idle_connections.popleft()

                # 检查连接是否过期或损坏
                if conn_info.is_expired() or conn_info.status == ConnectionStatus.BROKEN:
                    await self._close_connection(conn_info.connection_id)
                    continue

                # 验证连接
                if await self._validate_connection(conn_info.connection_id):
                    conn_info.status = ConnectionStatus.ACTIVE
                    conn_info.touch()
                    self.active_connections[conn_info.connection_id] = conn_info
                    self.stats['current_active'] += 1
                    self.stats['current_idle'] -= 1
                    self.stats['pool_hits'] += 1
                    return conn_info.connection_id
                else:
                    await self._close_connection(conn_info.connection_id)

            # 如果没有空闲连接，尝试创建新连接
            if len(self.active_connections) + len(self.idle_connections) < self.max_connections:
                conn_id = await self._create_connection()
                if conn_id:
                    conn_info = ConnectionInfo(
                        connection_id=conn_id,
                        connection_type=self.name,
                        status=ConnectionStatus.ACTIVE,
                        created_at=datetime.now(),
                        last_used=datetime.now(),
                        max_lifetime=self.max_lifetime
                    )
                    self.active_connections[conn_id] = conn_info
                    self.stats['current_active'] += 1
                    self.stats['pool_misses'] += 1
                    return conn_id

            # 等待连接可用
            self.stats['pool_misses'] += 1
            await self._condition.wait()
            return await self.acquire()  # 递归重试

    async def release(self, connection_id: str, broken: bool = False):
        """释放连接"""
        async with self._condition:
            if connection_id not in self.active_connections:
                return

            conn_info = self.active_connections.pop(connection_id)
            self.stats['current_active'] -= 1

            if broken or conn_info.is_expired():
                await self._close_connection(connection_id)
            else:
                conn_info.status = ConnectionStatus.IDLE
                self.idle_connections.append(conn_info)
                self.stats['current_idle'] += 1

            # 通知等待的协程
            self._condition.notify()

    async def _ensure_min_connections(self):
        """确保最小连接数"""
        current_total = len(self.active_connections) + len(self.idle_connections)

        while current_total < self.min_connections:
            conn_id = await self._create_connection()
            if conn_id:
                conn_info = ConnectionInfo(
                    connection_id=conn_id,
                    connection_type=self.name,
                    status=ConnectionStatus.IDLE,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    max_lifetime=self.max_lifetime
                )
                self.idle_connections.append(conn_info)
                self.stats['current_idle'] += 1
                current_total += 1
            else:
                break

    async def _create_connection(self) -> Optional[str]:
        """创建连接（子类实现）"""
        raise NotImplementedError

    async def _close_connection(self, connection_id: str):
        """关闭连接（子类实现）"""
        raise NotImplementedError

    async def _validate_connection(self, connection_id: str) -> bool:
        """验证连接（子类实现）"""
        raise NotImplementedError

    async def _close_all_connections(self):
        """关闭所有连接"""
        # 关闭活跃连接
        for conn_id in list(self.active_connections.keys()):
            await self._close_connection(conn_id)

        # 关闭空闲连接
        while self.idle_connections:
            conn_info = self.idle_connections.popleft()
            await self._close_connection(conn_info.connection_id)

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._cleanup_expired_connections()
                await self._ensure_min_connections()
                await asyncio.sleep(30)  # 30秒清理一次
            except Exception as e:
                logger.error(f"连接池清理错误 {self.name}: {e}")
                await asyncio.sleep(30)

    async def _cleanup_expired_connections(self):
        """清理过期连接"""
        async with self._lock:
            # 清理空闲连接中的过期连接
            expired_idle = []
            for i, conn_info in enumerate(self.idle_connections):
                if conn_info.is_expired():
                    expired_idle.append(i)

            # 从后往前删除，避免索引问题
            for i in reversed(expired_idle):
                conn_info = self.idle_connections[i]
                del self.idle_connections[i]
                await self._close_connection(conn_info.connection_id)
                self.stats['current_idle'] -= 1

    def get_stats(self) -> dict:
        """获取连接池统计"""
        return {
            'name': self.name,
            'min_connections': self.min_connections,
            'max_connections': self.max_connections,
            'max_lifetime': self.max_lifetime,
            'is_running': self.is_running,
            **self.stats,
            'pool_utilization': self.stats['current_active'] / self.max_connections if self.max_connections > 0 else 0
        }

class MySQLConnectionPool(BaseConnectionPool):
    """MySQL连接池"""

    def __init__(self, name: str = "mysql", **kwargs):
        super().__init__(name, **kwargs)
        self.db_config = config.get('database', {})
        self.connections = {}  # connection_id -> actual connection

    async def _create_connection(self) -> Optional[str]:
        """创建MySQL连接"""
        try:
            conn = await aiomysql.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 3306),
                user=self.db_config.get('user', 'root'),
                password=self.db_config.get('password', ''),
                db=self.db_config.get('database', 'trading'),
                charset='utf8mb4'
            )

            conn_id = f"mysql_{int(time.time() * 1000000)}"
            self.connections[conn_id] = conn
            self.stats['total_created'] += 1

            logger.debug(f"创建MySQL连接: {conn_id}")
            return conn_id

        except Exception as e:
            logger.error(f"创建MySQL连接失败: {e}")
            self.stats['connection_errors'] += 1
            return None

    async def _close_connection(self, connection_id: str):
        """关闭MySQL连接"""
        if connection_id in self.connections:
            try:
                conn = self.connections[connection_id]
                conn.close()
                del self.connections[connection_id]
                self.stats['total_closed'] += 1
                logger.debug(f"关闭MySQL连接: {connection_id}")
            except Exception as e:
                logger.error(f"关闭MySQL连接失败 {connection_id}: {e}")

    async def _validate_connection(self, connection_id: str) -> bool:
        """验证MySQL连接"""
        if connection_id not in self.connections:
            return False

        try:
            conn = self.connections[connection_id]
            await conn.ping()
            return True
        except Exception:
            return False

    def get_connection(self, connection_id: str):
        """获取实际的连接对象"""
        return self.connections.get(connection_id)

class RedisConnectionPool(BaseConnectionPool):
    """Redis连接池"""

    def __init__(self, name: str = "redis", **kwargs):
        super().__init__(name, **kwargs)
        self.redis_config = config.get('redis', {})
        self.connections = {}  # connection_id -> actual connection

    async def _create_connection(self) -> Optional[str]:
        """创建Redis连接"""
        try:
            redis_url = f"redis://{self.redis_config.get('host', 'localhost')}:{self.redis_config.get('port', 6379)}/{self.redis_config.get('db', 0)}"

            conn = await aioredis.from_url(
                redis_url,
                password=self.redis_config.get('password'),
                encoding='utf-8'
            )

            conn_id = f"redis_{int(time.time() * 1000000)}"
            self.connections[conn_id] = conn
            self.stats['total_created'] += 1

            logger.debug(f"创建Redis连接: {conn_id}")
            return conn_id

        except Exception as e:
            logger.error(f"创建Redis连接失败: {e}")
            self.stats['connection_errors'] += 1
            return None

    async def _close_connection(self, connection_id: str):
        """关闭Redis连接"""
        if connection_id in self.connections:
            try:
                conn = self.connections[connection_id]
                await conn.close()
                del self.connections[connection_id]
                self.stats['total_closed'] += 1
                logger.debug(f"关闭Redis连接: {connection_id}")
            except Exception as e:
                logger.error(f"关闭Redis连接失败 {connection_id}: {e}")

    async def _validate_connection(self, connection_id: str) -> bool:
        """验证Redis连接"""
        if connection_id not in self.connections:
            return False

        try:
            conn = self.connections[connection_id]
            await conn.ping()
            return True
        except Exception:
            return False

    def get_connection(self, connection_id: str):
        """获取实际的连接对象"""
        return self.connections.get(connection_id)

class HTTPConnectionPool(BaseConnectionPool):
    """HTTP连接池"""

    def __init__(self, name: str = "http", **kwargs):
        super().__init__(name, **kwargs)
        self.sessions = {}  # connection_id -> ClientSession
        self.session_config = {
            'timeout': aiohttp.ClientTimeout(total=30),
            'connector': aiohttp.TCPConnector(
                limit=100,  # 总连接池大小
                limit_per_host=30,  # 单主机连接数限制
                ttl_dns_cache=300,  # DNS缓存TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
        }

    async def _create_connection(self) -> Optional[str]:
        """创建HTTP会话"""
        try:
            session = aiohttp.ClientSession(**self.session_config)

            conn_id = f"http_{int(time.time() * 1000000)}"
            self.sessions[conn_id] = session
            self.stats['total_created'] += 1

            logger.debug(f"创建HTTP会话: {conn_id}")
            return conn_id

        except Exception as e:
            logger.error(f"创建HTTP会话失败: {e}")
            self.stats['connection_errors'] += 1
            return None

    async def _close_connection(self, connection_id: str):
        """关闭HTTP会话"""
        if connection_id in self.sessions:
            try:
                session = self.sessions[connection_id]
                await session.close()
                del self.sessions[connection_id]
                self.stats['total_closed'] += 1
                logger.debug(f"关闭HTTP会话: {connection_id}")
            except Exception as e:
                logger.error(f"关闭HTTP会话失败 {connection_id}: {e}")

    async def _validate_connection(self, connection_id: str) -> bool:
        """验证HTTP会话"""
        if connection_id not in self.sessions:
            return False

        try:
            session = self.sessions[connection_id]
            return not session.closed
        except Exception:
            return False

    def get_session(self, connection_id: str) -> Optional[aiohttp.ClientSession]:
        """获取HTTP会话"""
        return self.sessions.get(connection_id)

class ConnectionPoolManager:
    """连接池管理器"""

    def __init__(self):
        self.pools: Dict[str, BaseConnectionPool] = {}

        # 创建默认连接池
        self._create_default_pools()

        # 监控任务
        self.monitoring_task = None
        self.is_running = False

        logger.info("连接池管理器初始化完成")

    def _create_default_pools(self):
        """创建默认连接池"""
        # MySQL连接池
        mysql_config = config.get('connection_pools', {}).get('mysql', {})
        self.pools['mysql'] = MySQLConnectionPool(
            name='mysql',
            min_connections=mysql_config.get('min_connections', 2),
            max_connections=mysql_config.get('max_connections', 10),
            max_lifetime=mysql_config.get('max_lifetime', 3600)
        )

        # Redis连接池
        redis_config = config.get('connection_pools', {}).get('redis', {})
        self.pools['redis'] = RedisConnectionPool(
            name='redis',
            min_connections=redis_config.get('min_connections', 1),
            max_connections=redis_config.get('max_connections', 5),
            max_lifetime=redis_config.get('max_lifetime', 3600)
        )

        # HTTP连接池
        http_config = config.get('connection_pools', {}).get('http', {})
        self.pools['http'] = HTTPConnectionPool(
            name='http',
            min_connections=http_config.get('min_connections', 1),
            max_connections=http_config.get('max_connections', 20),
            max_lifetime=http_config.get('max_lifetime', 300)
        )

    async def start(self):
        """启动连接池管理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动所有连接池
        for pool in self.pools.values():
            await pool.start()

        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("连接池管理器启动完成")

    async def stop(self):
        """停止连接池管理器"""
        if not self.is_running:
            return

        logger.info("正在停止连接池管理器...")
        self.is_running = False

        # 取消监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()

        # 停止所有连接池
        for pool in self.pools.values():
            await pool.stop()

        logger.info("连接池管理器已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 监控连接池状态
                for pool_name, pool in self.pools.items():
                    stats = pool.get_stats()

                    # 检查连接池健康状态
                    if stats['connection_errors'] > 10:  # 错误过多
                        logger.warning(f"连接池 {pool_name} 错误过多: {stats['connection_errors']}")

                    if stats['pool_utilization'] > 0.8:  # 使用率过高
                        logger.warning(f"连接池 {pool_name} 使用率过高: {stats['pool_utilization']:.2%}")

                await asyncio.sleep(60)  # 1分钟监控间隔

            except Exception as e:
                logger.error(f"连接池监控错误: {e}")
                await asyncio.sleep(60)

    def get_pool(self, pool_name: str) -> Optional[BaseConnectionPool]:
        """获取连接池"""
        return self.pools.get(pool_name)

    def add_pool(self, pool_name: str, pool: BaseConnectionPool):
        """添加连接池"""
        self.pools[pool_name] = pool

        if self.is_running:
            asyncio.create_task(pool.start())

    async def remove_pool(self, pool_name: str):
        """移除连接池"""
        if pool_name in self.pools:
            pool = self.pools[pool_name]
            await pool.stop()
            del self.pools[pool_name]

    # 便利方法

    async def get_mysql_connection(self):
        """获取MySQL连接"""
        pool = self.get_pool('mysql')
        if pool:
            conn_id = await pool.acquire()
            return conn_id, pool.get_connection(conn_id)
        return None, None

    async def release_mysql_connection(self, connection_id: str, broken: bool = False):
        """释放MySQL连接"""
        pool = self.get_pool('mysql')
        if pool:
            await pool.release(connection_id, broken)

    async def get_redis_connection(self):
        """获取Redis连接"""
        pool = self.get_pool('redis')
        if pool:
            conn_id = await pool.acquire()
            return conn_id, pool.get_connection(conn_id)
        return None, None

    async def release_redis_connection(self, connection_id: str, broken: bool = False):
        """释放Redis连接"""
        pool = self.get_pool('redis')
        if pool:
            await pool.release(connection_id, broken)

    async def get_http_session(self):
        """获取HTTP会话"""
        pool = self.get_pool('http')
        if pool:
            conn_id = await pool.acquire()
            return conn_id, pool.get_session(conn_id)
        return None, None

    async def release_http_session(self, connection_id: str, broken: bool = False):
        """释放HTTP会话"""
        pool = self.get_pool('http')
        if pool:
            await pool.release(connection_id, broken)

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有连接池统计"""
        return {
            'is_running': self.is_running,
            'pools': {
                pool_name: pool.get_stats()
                for pool_name, pool in self.pools.items()
            },
            'total_pools': len(self.pools)
        }

# 连接池上下文管理器

class PooledConnection:
    """连接池上下文管理器"""

    def __init__(self, manager: ConnectionPoolManager, pool_name: str):
        self.manager = manager
        self.pool_name = pool_name
        self.connection_id = None
        self.connection = None

    async def __aenter__(self):
        pool = self.manager.get_pool(self.pool_name)
        if pool:
            self.connection_id = await pool.acquire()

            if self.pool_name == 'mysql':
                self.connection = pool.get_connection(self.connection_id)
            elif self.pool_name == 'redis':
                self.connection = pool.get_connection(self.connection_id)
            elif self.pool_name == 'http':
                self.connection = pool.get_session(self.connection_id)

        return self.connection

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection_id:
            pool = self.manager.get_pool(self.pool_name)
            if pool:
                broken = exc_type is not None
                await pool.release(self.connection_id, broken)

# 全局实例
_connection_pool_manager_instance = None

def get_connection_pool_manager() -> ConnectionPoolManager:
    """获取连接池管理器实例"""
    global _connection_pool_manager_instance
    if _connection_pool_manager_instance is None:
        _connection_pool_manager_instance = ConnectionPoolManager()
    return _connection_pool_manager_instance

# 便利函数

def mysql_connection():
    """MySQL连接上下文管理器"""
    manager = get_connection_pool_manager()
    return PooledConnection(manager, 'mysql')

def redis_connection():
    """Redis连接上下文管理器"""
    manager = get_connection_pool_manager()
    return PooledConnection(manager, 'redis')

def http_session():
    """HTTP会话上下文管理器"""
    manager = get_connection_pool_manager()
    return PooledConnection(manager, 'http')
