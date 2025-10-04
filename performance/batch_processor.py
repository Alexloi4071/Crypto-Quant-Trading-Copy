"""
Batch Processor
批处理优化器，提供高效的批量数据处理能力
支持自适应批大小、并行处理和智能调度
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import statistics
import sys
from pathlib import Path

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')

class BatchStatus(Enum):
    """批处理状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingStrategy(Enum):
    """处理策略"""
    SEQUENTIAL = "sequential"  # 顺序处理
    PARALLEL = "parallel"     # 并行处理
    PIPELINE = "pipeline"     # 流水线处理

@dataclass

class BatchMetrics:
    """批处理指标"""
    batch_id: str
    item_count: int
    processing_time: float
    throughput: float  # items/second
    memory_peak: int
    cpu_usage: float
    success_rate: float
    start_time: datetime
    end_time: datetime
    strategy: ProcessingStrategy

    def to_dict(self) -> dict:
        return {
            'batch_id': self.batch_id,
            'item_count': self.item_count,
            'processing_time': self.processing_time,
            'throughput': self.throughput,
            'memory_peak': self.memory_peak,
            'cpu_usage': self.cpu_usage,
            'success_rate': self.success_rate,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'strategy': self.strategy.value
        }

@dataclass

class BatchConfig:
    """批处理配置"""
    batch_id: str
    max_batch_size: int = 100
    min_batch_size: int = 10
    max_wait_time: float = 5.0  # seconds
    strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None

    # 自适应配置
    adaptive_sizing: bool = True
    target_throughput: Optional[float] = None
    target_latency: Optional[float] = None

class BatchItem(Generic[T]):
    """批处理项目"""

    def __init__(self, data: T, item_id: str = None, priority: int = 1):
        self.data = data
        self.item_id = item_id or f"item_{int(time.time() * 1000000)}"
        self.priority = priority
        self.created_at = datetime.now()
        self.retries = 0
        self.last_error = None

    def __lt__(self, other):
        return self.priority < other.priority

class Batch(Generic[T]):
    """批处理对象"""

    def __init__(self, batch_id: str, config: BatchConfig):
        self.batch_id = batch_id
        self.config = config
        self.items: List[BatchItem[T]] = []
        self.status = BatchStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[List[R]] = None
        self.error: Optional[Exception] = None
        self.metrics: Optional[BatchMetrics] = None

    def add_item(self, item: BatchItem[T]):
        """添加项目到批次"""
        if self.status != BatchStatus.PENDING:
            raise ValueError(f"Cannot add items to batch with status {self.status}")

        self.items.append(item)

    def is_ready(self) -> bool:
        """检查批次是否准备好处理"""
        if not self.items:
            return False

        # 检查批次大小
        if len(self.items) >= self.config.max_batch_size:
            return True

        # 检查等待时间
        if len(self.items) >= self.config.min_batch_size:
            wait_time = (datetime.now() - self.created_at).total_seconds()
            if wait_time >= self.config.max_wait_time:
                return True

        return False

    def get_data(self) -> List[T]:
        """获取所有数据"""
        return [item.data for item in self.items]

class BatchProcessor(Generic[T, R]):
    """批处理器"""

    def __init__(self, name: str, processor_func: Callable[[List[T]], Union[List[R], Awaitable[List[R]]]]):
        self.name = name
        self.processor_func = processor_func
        self.pending_batches: Dict[str, Batch[T]] = {}
        self.processing_batches: Dict[str, Batch[T]] = {}
        self.completed_batches: deque = deque(maxlen=1000)  # 保留历史

        # 性能监控
        self.metrics_history = deque(maxlen=1000)
        self.performance_stats = {
            'total_batches': 0,
            'total_items': 0,
            'total_processing_time': 0.0,
            'average_throughput': 0.0,
            'success_rate': 1.0,
            'adaptive_adjustments': 0
        }

        # 自适应配置
        self.current_batch_size = 50
        self.optimal_batch_size = 50
        self.size_adjustment_history = deque(maxlen=100)

        # 任务管理
        self.processing_task = None
        self.optimization_task = None
        self.is_running = False

        logger.debug(f"创建批处理器: {name}")

    async def start(self):
        """启动批处理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动处理任务
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.optimization_task = asyncio.create_task(self._optimization_loop())

        logger.info(f"批处理器 {self.name} 启动完成")

    async def stop(self):
        """停止批处理器"""
        if not self.is_running:
            return

        logger.info(f"正在停止批处理器: {self.name}")
        self.is_running = False

        # 处理剩余的批次
        for batch in list(self.pending_batches.values()):
            if batch.items:
                await self._process_batch(batch)

        # 取消任务
        if self.processing_task:
            self.processing_task.cancel()

        if self.optimization_task:
            self.optimization_task.cancel()

        logger.info(f"批处理器 {self.name} 已停止")

    async def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 检查准备好的批次
                ready_batches = [
                    batch for batch in self.pending_batches.values()
                    if batch.is_ready()
                ]

                # 处理准备好的批次
                for batch in ready_batches:
                    if batch.batch_id in self.pending_batches:
                        del self.pending_batches[batch.batch_id]
                        self.processing_batches[batch.batch_id] = batch

                        # 异步处理批次
                        asyncio.create_task(self._process_batch(batch))

                await asyncio.sleep(0.1)  # 100ms检查间隔

            except Exception as e:
                logger.error(f"处理循环错误 {self.name}: {e}")
                await asyncio.sleep(1)

    async def _process_batch(self, batch: Batch[T]):
        """处理单个批次"""
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.now()

        start_time = time.time()
        memory_before = self._get_memory_usage()

        try:
            # 根据策略处理批次
            if batch.config.strategy == ProcessingStrategy.SEQUENTIAL:
                result = await self._process_sequential(batch)
            elif batch.config.strategy == ProcessingStrategy.PARALLEL:
                result = await self._process_parallel(batch)
            elif batch.config.strategy == ProcessingStrategy.PIPELINE:
                result = await self._process_pipeline(batch)
            else:
                result = await self._process_sequential(batch)

            batch.result = result
            batch.status = BatchStatus.COMPLETED

        except Exception as e:
            logger.error(f"批处理失败 {batch.batch_id}: {e}")
            batch.error = e
            batch.status = BatchStatus.FAILED

        finally:
            batch.completed_at = datetime.now()

            # 计算指标
            end_time = time.time()
            processing_time = end_time - start_time
            memory_after = self._get_memory_usage()

            success_count = len(batch.result) if batch.result else 0
            success_rate = success_count / len(batch.items) if batch.items else 0

            metrics = BatchMetrics(
                batch_id=batch.batch_id,
                item_count=len(batch.items),
                processing_time=processing_time,
                throughput=len(batch.items) / processing_time if processing_time > 0 else 0,
                memory_peak=memory_after - memory_before,
                cpu_usage=self._get_cpu_usage(),
                success_rate=success_rate,
                start_time=batch.started_at,
                end_time=batch.completed_at,
                strategy=batch.config.strategy
            )

            batch.metrics = metrics
            self.metrics_history.append(metrics)

            # 移动到完成队列
            if batch.batch_id in self.processing_batches:
                del self.processing_batches[batch.batch_id]

            self.completed_batches.append(batch)

            # 更新性能统计
            self._update_performance_stats(metrics)

            logger.debug(f"批次处理完成 {batch.batch_id}: {len(batch.items)} 项, {processing_time:.3f}s")

    async def _process_sequential(self, batch: Batch[T]) -> List[R]:
        """顺序处理"""
        data = batch.get_data()

        if asyncio.iscoroutinefunction(self.processor_func):
            return await self.processor_func(data)
        else:
            return self.processor_func(data)

    async def _process_parallel(self, batch: Batch[T]) -> List[R]:
        """并行处理"""
        # 将批次分成更小的子批次并行处理
        chunk_size = max(1, len(batch.items) // 4)  # 分成4个子批次
        chunks = [
            batch.items[i:i + chunk_size]
            for i in range(0, len(batch.items), chunk_size)
        ]

        # 创建处理任务
        tasks = []
        for chunk in chunks:
            chunk_data = [item.data for item in chunk]

            if asyncio.iscoroutinefunction(self.processor_func):
                task = self.processor_func(chunk_data)
            else:
                task = asyncio.get_event_loop().run_in_executor(
                    None, self.processor_func, chunk_data
                )

            tasks.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        combined_result = []
        for result in results:
            if isinstance(result, Exception):
                raise result
            combined_result.extend(result)

        return combined_result

    async def _process_pipeline(self, batch: Batch[T]) -> List[R]:
        """流水线处理"""
        # 简化的流水线实现，实际可以更复杂
        return await self._process_parallel(batch)

    async def _optimization_loop(self):
        """优化循环"""
        while self.is_running:
            try:
                await self._optimize_batch_size()
                await asyncio.sleep(60)  # 1分钟优化一次
            except Exception as e:
                logger.error(f"优化循环错误 {self.name}: {e}")
                await asyncio.sleep(60)

    async def _optimize_batch_size(self):
        """优化批次大小"""
        if len(self.metrics_history) < 10:
            return  # 数据不足

        # 分析最近的性能指标
        recent_metrics = list(self.metrics_history)[-10:]

        # 计算平均吞吐量
        throughputs = [m.throughput for m in recent_metrics]
        avg_throughput = statistics.mean(throughputs)

        # 计算平均延迟
        latencies = [m.processing_time for m in recent_metrics]
        avg_latency = statistics.mean(latencies)

        # 优化决策
        old_size = self.current_batch_size

        # 如果吞吐量下降，尝试调整批次大小
        if len(self.size_adjustment_history) > 5:
            recent_throughputs = [h['throughput'] for h in list(self.size_adjustment_history)[-5:]]
            if statistics.mean(recent_throughputs) < avg_throughput * 0.9:
                # 吞吐量下降，尝试反向调整
                if self.current_batch_size > self.optimal_batch_size:
                    self.current_batch_size = max(10, int(self.current_batch_size * 0.8))
                else:
                    self.current_batch_size = min(200, int(self.current_batch_size * 1.2))
            elif avg_throughput > statistics.mean(recent_throughputs) * 1.1:
                # 吞吐量提升，继续当前方向
                self.optimal_batch_size = self.current_batch_size
        else:
            # 初始优化
            if avg_latency > 5.0:  # 延迟过高
                self.current_batch_size = max(10, int(self.current_batch_size * 0.8))
            elif avg_throughput < 10:  # 吞吐量过低
                self.current_batch_size = min(200, int(self.current_batch_size * 1.2))

        # 记录调整历史
        self.size_adjustment_history.append({
            'timestamp': datetime.now(),
            'old_size': old_size,
            'new_size': self.current_batch_size,
            'throughput': avg_throughput,
            'latency': avg_latency
        })

        if old_size != self.current_batch_size:
            self.performance_stats['adaptive_adjustments'] += 1
            logger.info(f"优化批次大小 {self.name}: {old_size} -> {self.current_batch_size}")

    def _update_performance_stats(self, metrics: BatchMetrics):
        """更新性能统计"""
        self.performance_stats['total_batches'] += 1
        self.performance_stats['total_items'] += metrics.item_count
        self.performance_stats['total_processing_time'] += metrics.processing_time

        # 更新平均吞吐量
        total_throughput = (self.performance_stats['average_throughput'] *
                           (self.performance_stats['total_batches'] - 1) +
                           metrics.throughput)
        self.performance_stats['average_throughput'] = total_throughput / self.performance_stats['total_batches']

        # 更新成功率
        total_success_rate = (self.performance_stats['success_rate'] *
                             (self.performance_stats['total_batches'] - 1) +
                             metrics.success_rate)
        self.performance_stats['success_rate'] = total_success_rate / self.performance_stats['total_batches']

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

    # 公共接口

    async def submit(self, data: T, priority: int = 1, config: BatchConfig = None) -> str:
        """提交单个项目"""
        config = config or BatchConfig(batch_id=f"batch_{int(time.time() * 1000)}")
        config.max_batch_size = self.current_batch_size  # 使用当前优化的批次大小

        item = BatchItem(data, priority=priority)

        # 查找或创建批次
        batch = None
        for b in self.pending_batches.values():
            if (len(b.items) < b.config.max_batch_size and
                b.config.strategy == config.strategy):
                batch = b
                break

        if not batch:
            batch = Batch[T](config.batch_id, config)
            self.pending_batches[batch.batch_id] = batch

        batch.add_item(item)
        return batch.batch_id

    async def submit_batch(self, data_list: List[T], config: BatchConfig = None) -> str:
        """提交批量项目"""
        config = config or BatchConfig(batch_id=f"batch_{int(time.time() * 1000)}")

        batch = Batch[T](config.batch_id, config)

        for data in data_list:
            item = BatchItem(data)
            batch.add_item(item)

        self.pending_batches[batch.batch_id] = batch
        return batch.batch_id

    def get_batch_result(self, batch_id: str) -> Optional[Batch[T]]:
        """获取批次结果"""
        # 检查处理中的批次
        if batch_id in self.processing_batches:
            return self.processing_batches[batch_id]

        # 检查完成的批次
        for batch in self.completed_batches:
            if batch.batch_id == batch_id:
                return batch

        # 检查待处理的批次
        if batch_id in self.pending_batches:
            return self.pending_batches[batch_id]

        return None

    async def wait_for_batch(self, batch_id: str, timeout: float = 60) -> Optional[Batch[T]]:
        """等待批次完成"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            batch = self.get_batch_result(batch_id)
            if batch and batch.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                return batch

            await asyncio.sleep(0.1)

        return None

    def get_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        return {
            'name': self.name,
            'is_running': self.is_running,
            'pending_batches': len(self.pending_batches),
            'processing_batches': len(self.processing_batches),
            'completed_batches': len(self.completed_batches),
            'current_batch_size': self.current_batch_size,
            'optimal_batch_size': self.optimal_batch_size,
            'performance_stats': self.performance_stats,
            'recent_metrics': [m.to_dict() for m in list(self.metrics_history)[-5:]]
        }

class BatchProcessingManager:
    """批处理管理器"""

    def __init__(self):
        self.processors: Dict[str, BatchProcessor] = {}

        # 全局统计
        self.global_stats = {
            'total_processors': 0,
            'total_batches_processed': 0,
            'total_items_processed': 0,
            'average_processing_time': 0.0,
            'start_time': datetime.now()
        }

        # 监控任务
        self.monitoring_task = None
        self.is_running = False

        logger.info("批处理管理器初始化完成")

    async def start(self):
        """启动批处理管理器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动所有处理器
        for processor in self.processors.values():
            await processor.start()

        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("批处理管理器启动完成")

    async def stop(self):
        """停止批处理管理器"""
        if not self.is_running:
            return

        logger.info("正在停止批处理管理器...")
        self.is_running = False

        # 停止监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()

        # 停止所有处理器
        for processor in self.processors.values():
            await processor.stop()

        logger.info("批处理管理器已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._collect_global_stats()
                await asyncio.sleep(60)  # 1分钟收集一次
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(60)

    async def _collect_global_stats(self):
        """收集全局统计"""
        total_batches = 0
        total_items = 0
        total_processing_time = 0.0

        for processor in self.processors.values():
            stats = processor.get_stats()
            total_batches += stats['performance_stats']['total_batches']
            total_items += stats['performance_stats']['total_items']
            total_processing_time += stats['performance_stats']['total_processing_time']

        self.global_stats.update({
            'total_batches_processed': total_batches,
            'total_items_processed': total_items,
            'average_processing_time': total_processing_time / max(total_batches, 1)
        })

    def create_processor(self, name: str, processor_func: Callable) -> BatchProcessor:
        """创建批处理器"""
        if name in self.processors:
            return self.processors[name]

        processor = BatchProcessor(name, processor_func)
        self.processors[name] = processor
        self.global_stats['total_processors'] += 1

        if self.is_running:
            asyncio.create_task(processor.start())

        logger.info(f"创建批处理器: {name}")
        return processor

    def get_processor(self, name: str) -> Optional[BatchProcessor]:
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

    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有统计信息"""
        processor_stats = {
            name: processor.get_stats()
            for name, processor in self.processors.items()
        }

        return {
            'is_running': self.is_running,
            'global_stats': self.global_stats,
            'processor_stats': processor_stats
        }

# 全局实例
_batch_processing_manager_instance = None

def get_batch_processing_manager() -> BatchProcessingManager:
    """获取批处理管理器实例"""
    global _batch_processing_manager_instance
    if _batch_processing_manager_instance is None:
        _batch_processing_manager_instance = BatchProcessingManager()
    return _batch_processing_manager_instance

# 便利装饰器

def batch_process(processor_name: str, batch_size: int = 50):
    """批处理装饰器"""

    def decorator(func):
        manager = get_batch_processing_manager()
        processor = manager.create_processor(processor_name, func)

        async def wrapper(data_list: List, **kwargs):
            config = BatchConfig(
                batch_id=f"{processor_name}_{int(time.time() * 1000)}",
                max_batch_size=batch_size
            )

            batch_id = await processor.submit_batch(data_list, config)
            batch = await processor.wait_for_batch(batch_id)

            if batch and batch.status == BatchStatus.COMPLETED:
                return batch.result
            elif batch and batch.error:
                raise batch.error
            else:
                raise TimeoutError(f"Batch processing timeout: {batch_id}")

        return wrapper
    return decorator
