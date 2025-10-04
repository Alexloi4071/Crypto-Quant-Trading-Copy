"""
Realtime Data Cache Manager
实时数据缓存管理器，提供高性能的内存缓存和持久化存储
支持多级缓存、数据压缩和智能过期策略
"""

import asyncio
import redis.asyncio as redis
import pickle
import lz4.frame
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
import sys
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

@dataclass

class CacheItem:
    """缓存项数据结构"""
    key: str
    data: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compressed: bool = False
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """检查是否已过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self):
        """更新访问时间和计数"""
        self.access_count += 1
        self.last_accessed = datetime.now()

class LRUCache:
    """基于LRU策略的内存缓存"""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.lock = threading.RLock()

        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0,
            'memory_usage': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]

                # 检查过期
                if item.is_expired():
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None

                # 移到末尾（最近使用）
                self.cache.move_to_end(key)
                item.touch()

                self.stats['hits'] += 1
                return item.data

            self.stats['misses'] += 1
            return None

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        with self.lock:
            try:
                # 计算过期时间
                expires_at = None
                if ttl is not None:
                    expires_at = datetime.now() + timedelta(seconds=ttl)
                elif self.default_ttl > 0:
                    expires_at = datetime.now() + timedelta(seconds=self.default_ttl)

                # 估算数据大小
                try:
                    size_bytes = len(pickle.dumps(data))
                except:
                    size_bytes = len(str(data))

                # 创建缓存项
                item = CacheItem(
                    key=key,
                    data=data,
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    size_bytes=size_bytes
                )

                # 如果已存在则更新
                if key in self.cache:
                    old_item = self.cache[key]
                    self.stats['memory_usage'] -= old_item.size_bytes

                self.cache[key] = item
                self.cache.move_to_end(key)

                self.stats['memory_usage'] += size_bytes
                self.stats['size'] = len(self.cache)

                # 检查是否需要淘汰
                self._evict_if_needed()

                return True

            except Exception as e:
                logger.error(f"设置缓存失败 {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                item = self.cache.pop(key)
                self.stats['memory_usage'] -= item.size_bytes
                self.stats['size'] = len(self.cache)
                return True
            return False

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats['memory_usage'] = 0
            self.stats['size'] = 0

    def _evict_if_needed(self):
        """根据需要淘汰旧项"""
        while len(self.cache) > self.max_size:
            # 移除最老的项
            key, item = self.cache.popitem(last=False)
            self.stats['memory_usage'] -= item.size_bytes
            self.stats['evictions'] += 1

        self.stats['size'] = len(self.cache)

    def cleanup_expired(self) -> int:
        """清理过期项"""
        expired_keys = []

        with self.lock:
            for key, item in self.cache.items():
                if item.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                item = self.cache.pop(key)
                self.stats['memory_usage'] -= item.size_bytes

            self.stats['size'] = len(self.cache)

        return len(expired_keys)

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0

            return {
                **self.stats,
                'hit_rate': hit_rate,
                'max_size': self.max_size,
                'usage_ratio': len(self.cache) / self.max_size
            }

class RealtimeDataCacheManager:
    """实时数据缓存管理器主类"""

    def __init__(self, redis_config: dict = None):
        # 内存缓存层
        self.l1_cache = LRUCache(max_size=50000, default_ttl=300)   # L1: 5万项，5分钟TTL
        self.l2_cache = LRUCache(max_size=100000, default_ttl=1800)  # L2: 10万项，30分钟TTL

        # Redis连接配置
        self.redis_config = redis_config or config.get('redis', {})
        self.redis_client = None
        self.redis_connected = False

        # 缓存策略配置
        self.cache_strategies = {
            'market_data': {
                'ttl': 60,        # 1分钟
                'level': 'l1',    # L1缓存
                'compress': False
            },
            'ohlcv_data': {
                'ttl': 300,       # 5分钟
                'level': 'l2',    # L2缓存
                'compress': True
            },
            'features': {
                'ttl': 600,       # 10分钟
                'level': 'l2',
                'compress': True
            },
            'signals': {
                'ttl': 1800,      # 30分钟
                'level': 'redis',
                'compress': True
            },
            'model_predictions': {
                'ttl': 900,       # 15分钟
                'level': 'redis',
                'compress': True
            }
        }

        # 统计信息
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_saves': 0,
            'errors': 0
        }

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 清理任务
        self.cleanup_task = None
        self.is_running = False

        logger.info("实时数据缓存管理器初始化完成")

    async def start(self):
        """启动缓存管理器"""
        if self.is_running:
            return

        try:
            # 连接Redis
            await self._connect_redis()

            self.is_running = True

            # 启动清理任务
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("缓存管理器启动完成")

        except Exception as e:
            logger.error(f"启动缓存管理器失败: {e}")
            raise

    async def stop(self):
        """停止缓存管理器"""
        if not self.is_running:
            return

        logger.info("正在停止缓存管理器...")
        self.is_running = False

        # 取消清理任务
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # 关闭Redis连接
        if self.redis_client:
            await self.redis_client.close()

        # 关闭线程池
        self.executor.shutdown(wait=True)

        logger.info("缓存管理器已停止")

    async def _connect_redis(self):
        """连接Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 1),  # 使用db 1用于缓存
                password=self.redis_config.get('password'),
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )

            # 测试连接
            await self.redis_client.ping()
            self.redis_connected = True
            logger.info("Redis缓存连接成功")

        except Exception as e:
            logger.warning(f"Redis缓存连接失败: {e}")
            self.redis_client = None
            self.redis_connected = False

    async def get(self, key: str, category: str = 'default') -> Optional[Any]:
        """获取缓存数据"""
        self.stats['requests'] += 1

        try:
            strategy = self.cache_strategies.get(category, self.cache_strategies['market_data'])
            level = strategy['level']

            # 尝试从不同缓存层获取
            data = None

            # L1缓存
            if level in ['l1', 'l2', 'redis']:
                data = self.l1_cache.get(key)
                if data is not None:
                    self.stats['cache_hits'] += 1
                    return self._decompress_if_needed(data)

            # L2缓存
            if data is None and level in ['l2', 'redis']:
                data = self.l2_cache.get(key)
                if data is not None:
                    # 提升到L1缓存
                    self.l1_cache.set(key, data, strategy['ttl'] // 2)
                    self.stats['cache_hits'] += 1
                    return self._decompress_if_needed(data)

            # Redis缓存
            if data is None and level == 'redis' and self.redis_connected:
                try:
                    redis_data = await self.redis_client.get(key)
                    if redis_data:
                        data = pickle.loads(redis_data)

                        # 提升到内存缓存
                        self.l2_cache.set(key, data, strategy['ttl'])
                        self.l1_cache.set(key, data, strategy['ttl'] // 2)

                        self.stats['cache_hits'] += 1
                        return self._decompress_if_needed(data)

                except Exception as e:
                    logger.warning(f"Redis获取失败: {e}")

            self.stats['cache_misses'] += 1
            return None

        except Exception as e:
            logger.error(f"获取缓存失败 {key}: {e}")
            self.stats['errors'] += 1
            return None

    async def set(self, key: str, data: Any, category: str = 'default', ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        try:
            strategy = self.cache_strategies.get(category, self.cache_strategies['market_data'])
            level = strategy['level']
            cache_ttl = ttl or strategy['ttl']
            compress = strategy['compress']

            # 压缩处理
            if compress:
                compressed_data = self._compress_data(data)
                if compressed_data != data:
                    self.stats['compression_saves'] += 1
                cache_data = compressed_data
            else:
                cache_data = data

            success = True

            # 设置到相应的缓存层
            if level in ['l1', 'l2', 'redis']:
                self.l1_cache.set(key, cache_data, cache_ttl)

            if level in ['l2', 'redis']:
                self.l2_cache.set(key, cache_data, cache_ttl)

            if level == 'redis' and self.redis_connected:
                try:
                    serialized = pickle.dumps(cache_data)
                    await self.redis_client.setex(key, cache_ttl, serialized)
                except Exception as e:
                    logger.warning(f"Redis设置失败: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {e}")
            self.stats['errors'] += 1
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存数据"""
        try:
            success = True

            # 从所有层删除
            self.l1_cache.delete(key)
            self.l2_cache.delete(key)

            if self.redis_connected:
                try:
                    await self.redis_client.delete(key)
                except Exception as e:
                    logger.warning(f"Redis删除失败: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {e}")
            return False

    async def clear(self, category: Optional[str] = None):
        """清空缓存"""
        try:
            if category is None:
                # 清空所有缓存
                self.l1_cache.clear()
                self.l2_cache.clear()

                if self.redis_connected:
                    await self.redis_client.flushdb()
            else:
                # 按类别清空（需要模式匹配）
                if self.redis_connected:
                    pattern = f"{category}:*"
                    keys = await self.redis_client.keys(pattern)
                    if keys:
                        await self.redis_client.delete(*keys)

            logger.info(f"缓存清空完成: {category or 'all'}")

        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    def _compress_data(self, data: Any) -> Any:
        """压缩数据"""
        try:
            # 对于大数据进行压缩
            serialized = pickle.dumps(data)

            if len(serialized) > 1024:  # 大于1KB的数据才压缩
                compressed = lz4.frame.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # 压缩率超过20%
                    return {
                        '__compressed__': True,
                        '__data__': compressed
                    }

            return data

        except Exception as e:
            logger.warning(f"数据压缩失败: {e}")
            return data

    def _decompress_if_needed(self, data: Any) -> Any:
        """根据需要解压数据"""
        try:
            if isinstance(data, dict) and data.get('__compressed__'):
                compressed_data = data['__data__']
                decompressed = lz4.frame.decompress(compressed_data)
                return pickle.loads(decompressed)

            return data

        except Exception as e:
            logger.warning(f"数据解压失败: {e}")
            return data

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                # 清理过期的内存缓存
                l1_expired = self.l1_cache.cleanup_expired()
                l2_expired = self.l2_cache.cleanup_expired()

                if l1_expired or l2_expired:
                    logger.debug(f"清理过期缓存 - L1: {l1_expired}, L2: {l2_expired}")

                await asyncio.sleep(60)  # 每分钟清理一次

            except Exception as e:
                logger.error(f"缓存清理错误: {e}")
                await asyncio.sleep(60)

    # 便利方法

    async def cache_market_data(self, symbol: str, data_type: str, data: Any) -> bool:
        """缓存市场数据"""
        key = f"market:{symbol}:{data_type}"
        return await self.set(key, data, 'market_data')

    async def get_market_data(self, symbol: str, data_type: str) -> Optional[Any]:
        """获取市场数据"""
        key = f"market:{symbol}:{data_type}"
        return await self.get(key, 'market_data')

    async def cache_ohlcv(self, symbol: str, timeframe: str, data: Any) -> bool:
        """缓存OHLCV数据"""
        key = f"ohlcv:{symbol}:{timeframe}"
        return await self.set(key, data, 'ohlcv_data')

    async def get_ohlcv(self, symbol: str, timeframe: str) -> Optional[Any]:
        """获取OHLCV数据"""
        key = f"ohlcv:{symbol}:{timeframe}"
        return await self.get(key, 'ohlcv_data')

    async def cache_features(self, symbol: str, features: dict, timestamp: datetime = None) -> bool:
        """缓存特征数据"""
        ts_str = (timestamp or datetime.now()).strftime('%Y%m%d_%H%M')
        key = f"features:{symbol}:{ts_str}"
        return await self.set(key, features, 'features')

    async def get_features(self, symbol: str, timestamp: datetime = None) -> Optional[dict]:
        """获取特征数据"""
        ts_str = (timestamp or datetime.now()).strftime('%Y%m%d_%H%M')
        key = f"features:{symbol}:{ts_str}"
        return await self.get(key, 'features')

    async def cache_signal(self, signal_id: str, signal_data: dict) -> bool:
        """缓存交易信号"""
        key = f"signal:{signal_id}"
        return await self.set(key, signal_data, 'signals')

    async def get_signal(self, signal_id: str) -> Optional[dict]:
        """获取交易信号"""
        key = f"signal:{signal_id}"
        return await self.get(key, 'signals')

    async def cache_prediction(self, symbol: str, model_name: str, prediction: dict) -> bool:
        """缓存模型预测"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        key = f"prediction:{symbol}:{model_name}:{timestamp}"
        return await self.set(key, prediction, 'model_predictions')

    async def get_latest_prediction(self, symbol: str, model_name: str) -> Optional[dict]:
        """获取最新预测"""
        # 这里简化处理，实际可能需要搜索最新的时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        key = f"prediction:{symbol}:{model_name}:{timestamp}"
        return await self.get(key, 'model_predictions')

    def get_stats(self) -> dict:
        """获取缓存管理器统计信息"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()

        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0

        return {
            'overall': {
                **self.stats,
                'cache_hit_rate': cache_hit_rate,
                'redis_connected': self.redis_connected
            },
            'l1_cache': l1_stats,
            'l2_cache': l2_stats,
            'strategies': self.cache_strategies
        }

# 全局实例
_cache_manager_instance = None

def get_cache_manager() -> RealtimeDataCacheManager:
    """获取缓存管理器实例"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = RealtimeDataCacheManager()
    return _cache_manager_instance
