"""
Cache Enhancer
缓存增强器，提供智能缓存管理和性能优化
支持多级缓存、智能预热和自适应淘汰策略
"""

import asyncio
import time
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import sys
from pathlib import Path
import weakref
import json

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

@dataclass

class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """更新访问时间"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl_seconds': self.ttl_seconds,
            'size_bytes': self.size_bytes,
            'expired': self.is_expired()
        }

@dataclass

class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0

    @property

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'total_size': self.total_size,
            'entry_count': self.entry_count
        }

class LRUCache:
    """LRU缓存实现"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory = max_memory_mb * 1024 * 1024  # 转换为字节
        self.cache = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self.cache:
                self.stats.misses += 1
                return None

            entry = self.cache[key]

            # 检查过期
            if entry.is_expired():
                del self.cache[key]
                self.stats.misses += 1
                self.stats.entry_count -= 1
                self.stats.total_size -= entry.size_bytes
                return None

            # 更新访问信息
            entry.touch()

            # 移到末尾（最近使用）
            self.cache.move_to_end(key)

            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """设置缓存值"""
        with self._lock:
            # 计算值的大小
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = sys.getsizeof(value)

            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )

            # 如果key已存在，更新
            if key in self.cache:
                old_entry = self.cache[key]
                self.stats.total_size -= old_entry.size_bytes
            else:
                self.stats.entry_count += 1

            self.cache[key] = entry
            self.stats.total_size += size_bytes

            # 移到末尾
            self.cache.move_to_end(key)

            # 检查是否需要淘汰
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                del self.cache[key]
                self.stats.entry_count -= 1
                self.stats.total_size -= entry.size_bytes
                return True
            return False

    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.stats = CacheStats()

    def _evict_if_needed(self):
        """根据需要进行淘汰"""
        # 按大小淘汰
        while (len(self.cache) > self.max_size or
               self.stats.total_size > self.max_memory):

            if not self.cache:
                break

            # 移除最久未使用的条目
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.stats.evictions += 1
            self.stats.entry_count -= 1
            self.stats.total_size -= oldest_entry.size_bytes

    def cleanup_expired(self):
        """清理过期条目"""
        with self._lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self.cache[key]
                del self.cache[key]
                self.stats.entry_count -= 1
                self.stats.total_size -= entry.size_bytes

    def get_stats(self) -> dict:
        """获取缓存统计"""
        with self._lock:
            return {
                **self.stats.to_dict(),
                'max_size': self.max_size,
                'max_memory_mb': self.max_memory / 1024 / 1024,
                'memory_usage_mb': self.stats.total_size / 1024 / 1024
            }

class RedisCache:
    """Redis缓存适配器"""

    def __init__(self, redis_client=None, key_prefix: str = "cache:"):
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def _make_key(self, key: str) -> str:
        """生成Redis key"""
        return f"{self.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.redis_client:
            self.stats.misses += 1
            return None

        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)

            if data is None:
                self.stats.misses += 1
                return None

            # 反序列化
            value = pickle.loads(data)
            self.stats.hits += 1
            return value

        except Exception as e:
            logger.error(f"Redis缓存获取失败: {e}")
            self.stats.misses += 1
            return None

    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """设置缓存值"""
        if not self.redis_client:
            return

        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)

            if ttl_seconds:
                await self.redis_client.setex(redis_key, ttl_seconds, data)
            else:
                await self.redis_client.set(redis_key, data)

        except Exception as e:
            logger.error(f"Redis缓存设置失败: {e}")

    async def delete(self, key: str) -> bool:
        """删除缓存条目"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis缓存删除失败: {e}")
            return False

    async def clear(self, pattern: str = "*"):
        """清空匹配的缓存"""
        if not self.redis_client:
            return

        try:
            search_pattern = f"{self.key_prefix}{pattern}"
            keys = await self.redis_client.keys(search_pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis缓存清空失败: {e}")

    def get_stats(self) -> dict:
        """获取缓存统计"""
        return self.stats.to_dict()

class MultiLevelCache:
    """多级缓存"""

    def __init__(self):
        # L1: 内存缓存（最快）
        self.l1_cache = LRUCache(max_size=500, max_memory_mb=50)

        # L2: 更大的内存缓存
        self.l2_cache = LRUCache(max_size=2000, max_memory_mb=200)

        # L3: Redis缓存（可选）
        self.l3_cache = None

        self.stats = {
            'l1': CacheStats(),
            'l2': CacheStats(),
            'l3': CacheStats()
        }

    def set_redis_cache(self, redis_client):
        """设置Redis缓存"""
        self.l3_cache = RedisCache(redis_client)

    async def get(self, key: str) -> Optional[Any]:
        """从多级缓存获取值"""
        # 尝试L1缓存
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats['l1'].hits += 1
            return value
        self.stats['l1'].misses += 1

        # 尝试L2缓存
        value = self.l2_cache.get(key)
        if value is not None:
            self.stats['l2'].hits += 1
            # 提升到L1
            self.l1_cache.put(key, value)
            return value
        self.stats['l2'].misses += 1

        # 尝试L3缓存（Redis）
        if self.l3_cache:
            value = await self.l3_cache.get(key)
            if value is not None:
                self.stats['l3'].hits += 1
                # 提升到L2和L1
                self.l2_cache.put(key, value)
                self.l1_cache.put(key, value)
                return value
            self.stats['l3'].misses += 1

        return None

    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """设置多级缓存值"""
        # 存储到所有级别
        self.l1_cache.put(key, value, ttl_seconds)
        self.l2_cache.put(key, value, ttl_seconds)

        if self.l3_cache:
            await self.l3_cache.put(key, value, ttl_seconds)

    async def delete(self, key: str):
        """从所有级别删除"""
        self.l1_cache.delete(key)
        self.l2_cache.delete(key)

        if self.l3_cache:
            await self.l3_cache.delete(key)

    async def clear(self):
        """清空所有级别的缓存"""
        self.l1_cache.clear()
        self.l2_cache.clear()

        if self.l3_cache:
            await self.l3_cache.clear()

    def get_stats(self) -> dict:
        """获取多级缓存统计"""
        stats = {
            'l1': self.l1_cache.get_stats(),
            'l2': self.l2_cache.get_stats()
        }

        if self.l3_cache:
            stats['l3'] = self.l3_cache.get_stats()

        # 计算总体统计
        total_hits = sum(s['hits'] for s in stats.values())
        total_misses = sum(s['misses'] for s in stats.values())

        stats['total'] = {
            'hits': total_hits,
            'misses': total_misses,
            'hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
        }

        return stats

class CacheWarmer:
    """缓存预热器"""

    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.warming_strategies = {}
        self.warming_tasks = set()

    def register_warming_strategy(self, name: str, strategy: Callable):
        """注册预热策略"""
        self.warming_strategies[name] = strategy

    async def warm_cache(self, strategy_name: str, *args, **kwargs):
        """执行缓存预热"""
        if strategy_name not in self.warming_strategies:
            logger.error(f"未知的预热策略: {strategy_name}")
            return

        strategy = self.warming_strategies[strategy_name]

        try:
            if asyncio.iscoroutinefunction(strategy):
                await strategy(self.cache, *args, **kwargs)
            else:
                strategy(self.cache, *args, **kwargs)

            logger.info(f"缓存预热完成: {strategy_name}")

        except Exception as e:
            logger.error(f"缓存预热失败 {strategy_name}: {e}")

    async def auto_warm(self):
        """自动预热"""
        for strategy_name in self.warming_strategies:
            task = asyncio.create_task(self.warm_cache(strategy_name))
            self.warming_tasks.add(task)

            # 清理完成的任务
            task.add_done_callback(self.warming_tasks.discard)

class SmartCacheManager:
    """智能缓存管理器"""

    def __init__(self):
        self.caches = {}  # cache_name -> cache_instance
        self.access_patterns = defaultdict(list)
        self.recommendations = []

    def create_cache(self, name: str, cache_type: str = "lru", **kwargs) -> Any:
        """创建缓存"""
        if cache_type == "lru":
            cache = LRUCache(**kwargs)
        elif cache_type == "multilevel":
            cache = MultiLevelCache()
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}")

        self.caches[name] = cache
        return cache

    def get_cache(self, name: str) -> Optional[Any]:
        """获取缓存实例"""
        return self.caches.get(name)

    def record_access(self, cache_name: str, key: str, hit: bool):
        """记录访问模式"""
        self.access_patterns[cache_name].append({
            'key': key,
            'timestamp': datetime.now(),
            'hit': hit
        })

        # 保持最近1000条记录
        if len(self.access_patterns[cache_name]) > 1000:
            self.access_patterns[cache_name] = self.access_patterns[cache_name][-1000:]

    def analyze_patterns(self, cache_name: str) -> Dict[str, Any]:
        """分析访问模式"""
        if cache_name not in self.access_patterns:
            return {}

        patterns = self.access_patterns[cache_name]
        if not patterns:
            return {}

        # 计算热点key
        key_counts = defaultdict(int)
        hit_rates = defaultdict(list)

        for pattern in patterns:
            key = pattern['key']
            key_counts[key] += 1
            hit_rates[key].append(pattern['hit'])

        hot_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # 计算命中率
        key_hit_rates = {}
        for key, hits in hit_rates.items():
            key_hit_rates[key] = sum(hits) / len(hits) if hits else 0

        return {
            'total_accesses': len(patterns),
            'unique_keys': len(key_counts),
            'hot_keys': hot_keys,
            'key_hit_rates': key_hit_rates,
            'overall_hit_rate': sum(p['hit'] for p in patterns) / len(patterns)
        }

    def generate_recommendations(self, cache_name: str) -> List[Dict[str, Any]]:
        """生成优化建议"""
        cache = self.get_cache(cache_name)
        if not cache:
            return []

        patterns = self.analyze_patterns(cache_name)
        recommendations = []

        # 命中率过低的建议
        if patterns.get('overall_hit_rate', 0) < 0.5:
            recommendations.append({
                'type': 'LOW_HIT_RATE',
                'description': f'缓存命中率过低 ({patterns["overall_hit_rate"]:.2%})',
                'suggestion': '考虑调整缓存策略或增加缓存大小'
            })

        # 热点key建议
        hot_keys = patterns.get('hot_keys', [])
        if hot_keys:
            top_key, access_count = hot_keys[0]
            if access_count > len(patterns.get('patterns', [])) * 0.1:  # 超过10%的访问
                recommendations.append({
                    'type': 'HOT_KEY',
                    'description': f'检测到热点key: {top_key} (访问次数: {access_count})',
                    'suggestion': '考虑为热点数据设置专门的缓存策略'
                })

        return recommendations

class CacheEnhancer:
    """缓存增强器主类"""

    def __init__(self):
        self.manager = SmartCacheManager()
        self.warmer = None
        self.default_cache = None

        # 监控任务
        self.monitoring_task = None
        self.cleanup_task = None
        self.is_running = False

        # 创建默认多级缓存
        self.default_cache = self.manager.create_cache("default", "multilevel")
        self.warmer = CacheWarmer(self.default_cache)

        logger.info("缓存增强器初始化完成")

    async def start(self):
        """启动缓存增强器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        # 自动预热
        await self.warmer.auto_warm()

        logger.info("缓存增强器启动完成")

    async def stop(self):
        """停止缓存增强器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()

        if self.cleanup_task:
            self.cleanup_task.cancel()

        logger.info("缓存增强器已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 分析所有缓存的访问模式
                for cache_name in self.manager.caches.keys():
                    patterns = self.manager.analyze_patterns(cache_name)
                    recommendations = self.manager.generate_recommendations(cache_name)

                    if recommendations:
                        for rec in recommendations:
                            logger.info(f"缓存优化建议 [{cache_name}]: {rec['description']}")

                await asyncio.sleep(300)  # 5分钟分析间隔

            except Exception as e:
                logger.error(f"缓存监控错误: {e}")
                await asyncio.sleep(300)

    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                # 清理过期缓存条目
                for cache in self.manager.caches.values():
                    if hasattr(cache, 'cleanup_expired'):
                        cache.cleanup_expired()
                    elif hasattr(cache, 'l1_cache'):  # MultiLevelCache
                        cache.l1_cache.cleanup_expired()
                        cache.l2_cache.cleanup_expired()

                await asyncio.sleep(60)  # 1分钟清理间隔

            except Exception as e:
                logger.error(f"缓存清理错误: {e}")
                await asyncio.sleep(60)

    # 便利方法

    async def get(self, key: str, cache_name: str = "default") -> Optional[Any]:
        """获取缓存值"""
        cache = self.manager.get_cache(cache_name)
        if not cache:
            return None

        if hasattr(cache, 'get') and asyncio.iscoroutinefunction(cache.get):
            value = await cache.get(key)
        else:
            value = cache.get(key)

        # 记录访问模式
        self.manager.record_access(cache_name, key, value is not None)

        return value

    async def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
                 cache_name: str = "default"):
        """设置缓存值"""
        cache = self.manager.get_cache(cache_name)
        if not cache:
            return

        if hasattr(cache, 'put') and asyncio.iscoroutinefunction(cache.put):
            await cache.put(key, value, ttl_seconds)
        else:
            cache.put(key, value, ttl_seconds)

    async def delete(self, key: str, cache_name: str = "default") -> bool:
        """删除缓存条目"""
        cache = self.manager.get_cache(cache_name)
        if not cache:
            return False

        if hasattr(cache, 'delete') and asyncio.iscoroutinefunction(cache.delete):
            return await cache.delete(key)
        else:
            return cache.delete(key)

    def cache_result(self, ttl_seconds: int = 300, cache_name: str = "default",
                    key_generator: Optional[Callable] = None):
        """缓存结果装饰器"""

        def decorator(func):
            @functools.wraps(func)

            async def async_wrapper(*args, **kwargs):
                # 生成缓存key
                if key_generator:
                    cache_key = key_generator(*args, **kwargs)
                else:
                    key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                    cache_key = hashlib.md5(key_data.encode()).hexdigest()

                # 尝试从缓存获取
                cached_result = await self.get(cache_key, cache_name)
                if cached_result is not None:
                    return cached_result

                # 执行函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # 缓存结果
                await self.put(cache_key, result, ttl_seconds, cache_name)

                return result

            @functools.wraps(func)

            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_wrapper(*args, **kwargs))

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """获取所有缓存统计"""
        stats = {}

        for cache_name, cache in self.manager.caches.items():
            if hasattr(cache, 'get_stats'):
                stats[cache_name] = cache.get_stats()

        return {
            'is_running': self.is_running,
            'cache_stats': stats,
            'access_patterns': {
                name: self.manager.analyze_patterns(name)
                for name in self.manager.caches.keys()
            },
            'recommendations': {
                name: self.manager.generate_recommendations(name)
                for name in self.manager.caches.keys()
            }
        }

# 全局实例
_cache_enhancer_instance = None

def get_cache_enhancer() -> CacheEnhancer:
    """获取缓存增强器实例"""
    global _cache_enhancer_instance
    if _cache_enhancer_instance is None:
        _cache_enhancer_instance = CacheEnhancer()
    return _cache_enhancer_instance

# 便利装饰器

def cached(ttl_seconds: int = 300, cache_name: str = "default", key_generator: Callable = None):
    """缓存装饰器"""
    enhancer = get_cache_enhancer()
    return enhancer.cache_result(ttl_seconds, cache_name, key_generator)
