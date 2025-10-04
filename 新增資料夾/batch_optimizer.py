"""
Batch Processing Optimizer
批处理优化引擎，提供智能批处理和性能优化功能
支持自适应批大小、数据聚合和负载调节
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import statistics
import sys
from pathlib import Path
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

@dataclass
class BatchMetrics:
    """批处理指标"""
    batch_id: str
    batch_size: int
    processing_time: float
    throughput: float  # items/second
    memory_usage: int  # bytes
    cpu_usage: float  # percentage
    timestamp: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0
    error_count: int = 0

@dataclass
class OptimizationProfile:
    """优化配置文件"""
    min_batch_size: int = 10
    max_batch_size: int = 1000
    target_latency_ms: float = 1000
    target_throughput: float = 100  # items/second
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 80
    adaptivity_factor: float = 0.1  # 调整幅度
    
    # 触发条件
    batch_timeout_ms: int = 5000  # 批次超时
    queue_size_threshold: int = 100  # 队列阈值

class AdaptiveBatchProcessor:
    """自适应批处理器"""
    
    def __init__(self, name: str, processor_func: Callable, profile: OptimizationProfile = None):
        self.name = name
        self.processor_func = processor_func
        self.profile = profile or OptimizationProfile()
        
        # 批处理状态
        self.current_batch_size = self.profile.min_batch_size
        self.batch_buffer = deque()
        self.batch_start_time = None
        
        # 性能历史
        self.performance_history = deque(maxlen=100)
        self.recent_metrics = deque(maxlen=20)
        
        # 统计信息
        self.stats = {
            'total_batches': 0,
            'total_items': 0,
            'total_processing_time': 0.0,
            'optimization_adjustments': 0,
            'start_time': datetime.now()
        }
        
        # 任务管理
        self.processing_task = None
        self.optimization_task = None
        self.is_running = False
        
        # 锁
        self.lock = threading.RLock()
        
        logger.info(f"自适应批处理器初始化: {name}")
    
    async def start(self):
        """启动批处理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.batch_start_time = time.time()
        
        # 启动处理和优化任务
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info(f"批处理器 {self.name} 启动完成")
    
    async def stop(self):
        """停止批处理器"""
        if not self.is_running:
            return
        
        logger.info(f"正在停止批处理器: {self.name}")
        self.is_running = False
        
        # 处理剩余批次
        if self.batch_buffer:
            await self._process_current_batch()
        
        # 取消任务
        if self.processing_task:
            self.processing_task.cancel()
        
        if self.optimization_task:
            self.optimization_task.cancel()
        
        logger.info(f"批处理器 {self.name} 已停止")
    
    async def add_item(self, item: Any) -> bool:
        """添加项到批处理"""
        with self.lock:
            self.batch_buffer.append(item)
            
            # 检查是否需要立即处理
            if (len(self.batch_buffer) >= self.current_batch_size or
                self._should_flush_batch()):
                asyncio.create_task(self._process_current_batch())
            
            return True
    
    async def add_items(self, items: List[Any]) -> int:
        """批量添加项"""
        added_count = 0
        
        with self.lock:
            for item in items:
                self.batch_buffer.append(item)
                added_count += 1
                
                if len(self.batch_buffer) >= self.current_batch_size:
                    asyncio.create_task(self._process_current_batch())
        
        return added_count
    
    def _should_flush_batch(self) -> bool:
        """检查是否应该刷新批次"""
        if not self.batch_buffer:
            return False
        
        # 超时检查
        if self.batch_start_time:
            elapsed_ms = (time.time() - self.batch_start_time) * 1000
            if elapsed_ms > self.profile.batch_timeout_ms:
                return True
        
        # 队列大小检查
        if len(self.batch_buffer) >= self.profile.queue_size_threshold:
            return True
        
        return False
    
    async def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                await asyncio.sleep(0.1)  # 100ms检查间隔
                
                if self._should_flush_batch():
                    await self._process_current_batch()
                    
            except Exception as e:
                logger.error(f"处理循环错误 {self.name}: {e}")
                await asyncio.sleep(1)
    
    async def _process_current_batch(self):
        """处理当前批次"""
        with self.lock:
            if not self.batch_buffer:
                return
            
            # 提取批次数据
            batch_size = min(len(self.batch_buffer), self.current_batch_size)
            batch_items = [self.batch_buffer.popleft() for _ in range(batch_size)]
            batch_id = f"{self.name}_{int(time.time() * 1000)}"
        
        # 记录开始时间
        start_time = time.time()
        memory_before = self._get_memory_usage()
        cpu_before = self._get_cpu_usage()
        
        success_count = 0
        error_count = 0
        
        try:
            # 执行批处理函数
            if asyncio.iscoroutinefunction(self.processor_func):
                result = await self.processor_func(batch_items)
            else:
                result = self.processor_func(batch_items)
            
            success_count = len(batch_items)
            
        except Exception as e:
            logger.error(f"批处理执行失败 {batch_id}: {e}")
            error_count = len(batch_items)
        
        # 计算指标
        end_time = time.time()
        processing_time = end_time - start_time
        memory_after = self._get_memory_usage()
        cpu_after = self._get_cpu_usage()
        
        throughput = len(batch_items) / max(processing_time, 0.001)
        success_rate = success_count / len(batch_items) if batch_items else 0
        
        metrics = BatchMetrics(
            batch_id=batch_id,
            batch_size=len(batch_items),
            processing_time=processing_time,
            throughput=throughput,
            memory_usage=memory_after - memory_before,
            cpu_usage=cpu_after - cpu_before,
            success_rate=success_rate,
            error_count=error_count
        )
        
        # 记录指标
        self.performance_history.append(metrics)
        self.recent_metrics.append(metrics)
        
        # 更新统计
        self.stats['total_batches'] += 1
        self.stats['total_items'] += len(batch_items)
        self.stats['total_processing_time'] += processing_time
        
        # 重置批次开始时间
        with self.lock:
            if self.batch_buffer:
                self.batch_start_time = time.time()
            else:
                self.batch_start_time = None
        
        logger.debug(f"处理批次 {batch_id}: {len(batch_items)} 项, {processing_time:.3f}s")
    
    async def _optimization_loop(self):
        """优化循环"""
        while self.is_running:
            try:
                await self._optimize_batch_size()
                await asyncio.sleep(30)  # 30秒优化一次
            except Exception as e:
                logger.error(f"优化循环错误 {self.name}: {e}")
                await asyncio.sleep(30)
    
    async def _optimize_batch_size(self):
        """优化批次大小"""
        if len(self.recent_metrics) < 5:
            return  # 数据不足，不优化
        
        # 分析最近的性能指标
        recent_latencies = [m.processing_time * 1000 for m in self.recent_metrics]  # 转为毫秒
        recent_throughputs = [m.throughput for m in self.recent_metrics]
        recent_cpu_usage = [m.cpu_usage for m in self.recent_metrics]
        recent_memory_usage = [m.memory_usage for m in self.recent_metrics]
        
        avg_latency = statistics.mean(recent_latencies)
        avg_throughput = statistics.mean(recent_throughputs)
        avg_cpu = statistics.mean(recent_cpu_usage)
        avg_memory = statistics.mean(recent_memory_usage) / (1024 * 1024)  # 转为MB
        
        # 当前批次大小
        old_batch_size = self.current_batch_size
        new_batch_size = self.current_batch_size
        
        # 优化决策逻辑
        should_increase = False
        should_decrease = False
        
        # 延迟优化
        if avg_latency > self.profile.target_latency_ms:
            should_decrease = True
        elif avg_latency < self.profile.target_latency_ms * 0.7:
            should_increase = True
        
        # 吞吐量优化
        if avg_throughput < self.profile.target_throughput:
            should_increase = True
        
        # 资源限制检查
        if avg_cpu > self.profile.cpu_limit_percent:
            should_decrease = True
        
        if avg_memory > self.profile.memory_limit_mb:
            should_decrease = True
        
        # 应用优化
        if should_decrease and not should_increase:
            # 减少批次大小
            adjustment = max(1, int(self.current_batch_size * self.profile.adaptivity_factor))
            new_batch_size = max(self.profile.min_batch_size, self.current_batch_size - adjustment)
            
        elif should_increase and not should_decrease:
            # 增加批次大小
            adjustment = max(1, int(self.current_batch_size * self.profile.adaptivity_factor))
            new_batch_size = min(self.profile.max_batch_size, self.current_batch_size + adjustment)
        
        # 应用新的批次大小
        if new_batch_size != old_batch_size:
            self.current_batch_size = new_batch_size
            self.stats['optimization_adjustments'] += 1
            
            logger.info(f"优化批次大小 {self.name}: {old_batch_size} -> {new_batch_size} "
                       f"(延迟: {avg_latency:.1f}ms, 吞吐量: {avg_throughput:.1f}, "
                       f"CPU: {avg_cpu:.1f}%, 内存: {avg_memory:.1f}MB)")
    
    def _get_memory_usage(self) -> int:
        """获取内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0
    
    def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        avg_processing_time = (self.stats['total_processing_time'] / 
                             max(self.stats['total_batches'], 1))
        
        avg_batch_size = (self.stats['total_items'] / 
                         max(self.stats['total_batches'], 1))
        
        # 最近性能指标
        recent_performance = {}
        if self.recent_metrics:
            recent_performance = {
                'avg_latency_ms': statistics.mean([m.processing_time * 1000 for m in self.recent_metrics]),
                'avg_throughput': statistics.mean([m.throughput for m in self.recent_metrics]),
                'avg_success_rate': statistics.mean([m.success_rate for m in self.recent_metrics])
            }
        
        return {
            'processor_name': self.name,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'current_batch_size': self.current_batch_size,
            'queue_size': len(self.batch_buffer),
            'profile': {
                'min_batch_size': self.profile.min_batch_size,
                'max_batch_size': self.profile.max_batch_size,
                'target_latency_ms': self.profile.target_latency_ms,
                'target_throughput': self.profile.target_throughput
            },
            'performance_stats': {
                'total_batches': self.stats['total_batches'],
                'total_items': self.stats['total_items'],
                'avg_processing_time': avg_processing_time,
                'avg_batch_size': avg_batch_size,
                'optimization_adjustments': self.stats['optimization_adjustments']
            },
            'recent_performance': recent_performance
        }

class BatchProcessingOptimizer:
    """批处理优化引擎主类"""
    
    def __init__(self):
        self.processors = {}  # name -> AdaptiveBatchProcessor
        self.global_stats = {
            'total_processors': 0,
            'total_items_processed': 0,
            'total_optimization_adjustments': 0,
            'start_time': datetime.now()
        }
        
        # 监控任务
        self.monitoring_task = None
        self.is_running = False
        
        logger.info("批处理优化引擎初始化完成")
    
    async def start(self):
        """启动优化引擎"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动所有处理器
        for processor in self.processors.values():
            await processor.start()
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("批处理优化引擎启动完成")
    
    async def stop(self):
        """停止优化引擎"""
        if not self.is_running:
            return
        
        logger.info("正在停止批处理优化引擎...")
        self.is_running = False
        
        # 停止监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # 停止所有处理器
        for processor in self.processors.values():
            await processor.stop()
        
        logger.info("批处理优化引擎已停止")
    
    def create_processor(self, name: str, processor_func: Callable, 
                        profile: OptimizationProfile = None) -> AdaptiveBatchProcessor:
        """创建批处理器"""
        if name in self.processors:
            return self.processors[name]
        
        processor = AdaptiveBatchProcessor(name, processor_func, profile)
        self.processors[name] = processor
        self.global_stats['total_processors'] += 1
        
        if self.is_running:
            asyncio.create_task(processor.start())
        
        logger.info(f"创建批处理器: {name}")
        return processor
    
    def get_processor(self, name: str) -> Optional[AdaptiveBatchProcessor]:
        """获取批处理器"""
        return self.processors.get(name)
    
    async def remove_processor(self, name: str):
        """移除批处理器"""
        if name in self.processors:
            processor = self.processors[name]
            await processor.stop()
            del self.processors[name]
            self.global_stats['total_processors'] -= 1
            logger.info(f"移除批处理器: {name}")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._collect_global_stats()
                await asyncio.sleep(60)  # 每分钟收集一次
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _collect_global_stats(self):
        """收集全局统计"""
        total_items = 0
        total_adjustments = 0
        
        for processor in self.processors.values():
            stats = processor.get_stats()
            total_items += stats['performance_stats']['total_items']
            total_adjustments += stats['performance_stats']['optimization_adjustments']
        
        self.global_stats['total_items_processed'] = total_items
        self.global_stats['total_optimization_adjustments'] = total_adjustments
        
        logger.info(f"全局批处理统计 - 处理器: {len(self.processors)}, "
                   f"处理项目: {total_items}, 优化调整: {total_adjustments}")
    
    def get_stats(self) -> dict:
        """获取引擎统计"""
        uptime = (datetime.now() - self.global_stats['start_time']).total_seconds()
        
        processor_stats = {
            name: processor.get_stats()
            for name, processor in self.processors.items()
        }
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'global_stats': self.global_stats,
            'processor_stats': processor_stats
        }
    
    # 便利方法
    async def process_items(self, processor_name: str, items: List[Any]) -> int:
        """处理项目列表"""
        processor = self.get_processor(processor_name)
        if not processor:
            raise ValueError(f"处理器不存在: {processor_name}")
        
        return await processor.add_items(items)
    
    async def process_item(self, processor_name: str, item: Any) -> bool:
        """处理单个项目"""
        processor = self.get_processor(processor_name)
        if not processor:
            raise ValueError(f"处理器不存在: {processor_name}")
        
        return await processor.add_item(item)

# 预定义处理器函数示例
async def default_data_processor(items: List[Any]) -> List[Any]:
    """默认数据处理器"""
    # 模拟数据处理
    await asyncio.sleep(len(items) * 0.001)  # 每项1ms
    return items

def sync_data_processor(items: List[Any]) -> List[Any]:
    """同步数据处理器"""
    # 模拟数据处理
    time.sleep(len(items) * 0.001)  # 每项1ms
    return items

# 全局实例
_batch_optimizer_instance = None

def get_batch_optimizer() -> BatchProcessingOptimizer:
    """获取批处理优化引擎实例"""
    global _batch_optimizer_instance
    if _batch_optimizer_instance is None:
        _batch_optimizer_instance = BatchProcessingOptimizer()
    return _batch_optimizer_instance