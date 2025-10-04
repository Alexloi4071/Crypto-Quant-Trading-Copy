"""
Async Performance Optimizer
异步性能优化器，提供协程优化、任务调度和并发控制
自动检测和优化异步代码的性能瓶颈
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
from pathlib import Path
import functools
import weakref

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

@dataclass

class TaskMetrics:
    """任务性能指标"""
    task_id: str
    function_name: str
    execution_time: float
    memory_usage: int
    cpu_time: float
    start_time: datetime
    end_time: datetime
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'function_name': self.function_name,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_time': self.cpu_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'success': self.success,
            'error': self.error
        }

@dataclass

class OptimizationSuggestion:
    """优化建议"""
    suggestion_id: str
    function_name: str
    issue_type: str
    description: str
    impact_level: str  # HIGH, MEDIUM, LOW
    suggested_fix: str
    estimated_improvement: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'suggestion_id': self.suggestion_id,
            'function_name': self.function_name,
            'issue_type': self.issue_type,
            'description': self.description,
            'impact_level': self.impact_level,
            'suggested_fix': self.suggested_fix,
            'estimated_improvement': self.estimated_improvement
        }

class AsyncProfiler:
    """异步函数性能分析器"""

    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_tasks = {}

    def profile_async(self, func_name: str = None):
        """异步函数性能分析装饰器"""

        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)

            async def wrapper(*args, **kwargs):
                task_id = f"{name}_{int(time.time() * 1000000)}"
                start_time = datetime.now()
                start_memory = self._get_memory_usage()
                start_cpu = time.process_time()

                self.active_tasks[task_id] = {
                    'function_name': name,
                    'start_time': start_time,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }

                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None

                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise

                finally:
                    end_time = datetime.now()
                    end_memory = self._get_memory_usage()
                    end_cpu = time.process_time()

                    metrics = TaskMetrics(
                        task_id=task_id,
                        function_name=name,
                        execution_time=(end_time - start_time).total_seconds(),
                        memory_usage=end_memory - start_memory,
                        cpu_time=end_cpu - start_cpu,
                        start_time=start_time,
                        end_time=end_time,
                        success=success,
                        error=error
                    )

                    self.metrics_history[name].append(metrics)

                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]

                return result

            return wrapper
        return decorator

    def _get_memory_usage(self) -> int:
        """获取内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0

    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """获取函数性能统计"""
        metrics = list(self.metrics_history[function_name])
        if not metrics:
            return {}

        execution_times = [m.execution_time for m in metrics]
        memory_usages = [m.memory_usage for m in metrics]
        success_count = sum(1 for m in metrics if m.success)

        return {
            'function_name': function_name,
            'total_calls': len(metrics),
            'success_rate': success_count / len(metrics) if metrics else 0,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_memory_usage': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
            'last_call': metrics[-1].start_time.isoformat() if metrics else None
        }

class TaskScheduler:
    """智能任务调度器"""

    def __init__(self, max_concurrent_tasks: int = 100):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue = asyncio.PriorityQueue()
        self.running_tasks = set()
        self.task_history = deque(maxlen=10000)

        # 调度策略
        self.scheduling_enabled = True
        self.adaptive_concurrency = True
        self.current_concurrency = max_concurrent_tasks // 2

        # 性能监控
        self.scheduler_metrics = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_wait_time': 0.0,
            'avg_execution_time': 0.0,
            'concurrency_adjustments': 0
        }

    async def schedule_task(self, coro: Awaitable, priority: int = 1,
                          task_name: str = None) -> Any:
        """调度异步任务"""
        if not self.scheduling_enabled:
            return await coro

        task_name = task_name or f"task_{int(time.time() * 1000)}"
        schedule_time = datetime.now()

        # 如果当前并发数未达到限制，直接执行
        if len(self.running_tasks) < self.current_concurrency:
            return await self._execute_task(coro, task_name, schedule_time)

        # 否则加入队列等待
        future = asyncio.Future()
        await self.task_queue.put((priority, schedule_time, coro, task_name, future))
        self.scheduler_metrics['tasks_scheduled'] += 1

        return await future

    async def _execute_task(self, coro: Awaitable, task_name: str,
                          schedule_time: datetime) -> Any:
        """执行任务"""
        execution_start = datetime.now()
        wait_time = (execution_start - schedule_time).total_seconds()

        task_info = {
            'name': task_name,
            'schedule_time': schedule_time,
            'execution_start': execution_start,
            'wait_time': wait_time
        }

        self.running_tasks.add(task_name)

        try:
            result = await coro

            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds()

            task_info.update({
                'execution_end': execution_end,
                'execution_time': execution_time,
                'success': True,
                'error': None
            })

            self.scheduler_metrics['tasks_completed'] += 1
            self._update_performance_metrics(wait_time, execution_time)

            return result

        except Exception as e:
            task_info.update({
                'execution_end': datetime.now(),
                'execution_time': (datetime.now() - execution_start).total_seconds(),
                'success': False,
                'error': str(e)
            })

            self.scheduler_metrics['tasks_failed'] += 1
            raise

        finally:
            self.running_tasks.discard(task_name)
            self.task_history.append(task_info)

            # 尝试调度下一个任务
            await self._try_schedule_next()

    async def _try_schedule_next(self):
        """尝试调度下一个任务"""
        if (not self.task_queue.empty() and
            len(self.running_tasks) < self.current_concurrency):

            try:
                priority, schedule_time, coro, task_name, future = await asyncio.wait_for(
                    self.task_queue.get(), timeout=0.1
                )

                # 在后台执行任务
                asyncio.create_task(
                    self._execute_and_resolve(coro, task_name, schedule_time, future)
                )

            except asyncio.TimeoutError:
                pass

    async def _execute_and_resolve(self, coro: Awaitable, task_name: str,
                                 schedule_time: datetime, future: asyncio.Future):
        """执行任务并设置Future结果"""
        try:
            result = await self._execute_task(coro, task_name, schedule_time)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)

    def _update_performance_metrics(self, wait_time: float, execution_time: float):
        """更新性能指标"""
        total_tasks = self.scheduler_metrics['tasks_completed']

        # 更新平均等待时间
        current_avg_wait = self.scheduler_metrics['avg_wait_time']
        self.scheduler_metrics['avg_wait_time'] = (
            (current_avg_wait * (total_tasks - 1) + wait_time) / total_tasks
        )

        # 更新平均执行时间
        current_avg_exec = self.scheduler_metrics['avg_execution_time']
        self.scheduler_metrics['avg_execution_time'] = (
            (current_avg_exec * (total_tasks - 1) + execution_time) / total_tasks
        )

        # 自适应并发控制
        if self.adaptive_concurrency and total_tasks % 100 == 0:
            self._adjust_concurrency()

    def _adjust_concurrency(self):
        """调整并发数"""
        avg_wait_time = self.scheduler_metrics['avg_wait_time']

        # 如果平均等待时间过长，增加并发数
        if avg_wait_time > 1.0 and self.current_concurrency < self.max_concurrent_tasks:
            self.current_concurrency = min(
                self.max_concurrent_tasks,
                int(self.current_concurrency * 1.2)
            )
            self.scheduler_metrics['concurrency_adjustments'] += 1
            logger.info(f"增加并发数至: {self.current_concurrency}")

        # 如果等待时间很短但CPU使用率高，减少并发数
        elif avg_wait_time < 0.1 and self.current_concurrency > 10:
            self.current_concurrency = max(10, int(self.current_concurrency * 0.8))
            self.scheduler_metrics['concurrency_adjustments'] += 1
            logger.info(f"减少并发数至: {self.current_concurrency}")

    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'current_concurrency': self.current_concurrency,
            'running_tasks_count': len(self.running_tasks),
            'queued_tasks_count': self.task_queue.qsize(),
            'metrics': self.scheduler_metrics,
            'scheduling_enabled': self.scheduling_enabled,
            'adaptive_concurrency': self.adaptive_concurrency
        }

class ConcurrencyLimiter:
    """并发控制器"""

    def __init__(self, max_concurrent: int, time_window: int = 60):
        self.max_concurrent = max_concurrent
        self.time_window = time_window
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """获取并发权限"""
        async with self._lock:
            # 清理过期的请求记录
            current_time = time.time()
            while (self.request_times and
                   current_time - self.request_times[0] > self.time_window):
                self.request_times.popleft()

            # 记录当前请求时间
            self.request_times.append(current_time)

        await self.semaphore.acquire()

    def release(self):
        """释放并发权限"""
        self.semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def get_current_usage(self) -> Dict[str, Any]:
        """获取当前使用情况"""
        return {
            'max_concurrent': self.max_concurrent,
            'available_permits': self.semaphore._value,
            'active_requests': self.max_concurrent - self.semaphore._value,
            'requests_in_window': len(self.request_times),
            'time_window': self.time_window
        }

class PerformanceAnalyzer:
    """性能分析器"""

    def __init__(self):
        self.profiler = AsyncProfiler()
        self.bottleneck_detector = BottleneckDetector()
        self.suggestions = []

    def analyze_function_performance(self, function_name: str) -> List[OptimizationSuggestion]:
        """分析函数性能并生成优化建议"""
        stats = self.profiler.get_function_stats(function_name)
        if not stats:
            return []

        suggestions = []

        # 检查执行时间
        if stats['avg_execution_time'] > 1.0:
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"slow_function_{function_name}_{int(time.time())}",
                function_name=function_name,
                issue_type="SLOW_EXECUTION",
                description=f"函数平均执行时间过长: {stats['avg_execution_time']:.2f}秒",
                impact_level="HIGH",
                suggested_fix="考虑使用缓存、并行处理或优化算法",
                estimated_improvement=0.3
            ))

        # 检查成功率
        if stats['success_rate'] < 0.95:
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"low_success_{function_name}_{int(time.time())}",
                function_name=function_name,
                issue_type="LOW_SUCCESS_RATE",
                description=f"函数成功率较低: {stats['success_rate']:.2%}",
                impact_level="MEDIUM",
                suggested_fix="增加错误处理和重试机制",
                estimated_improvement=0.1
            ))

        # 检查内存使用
        if stats['avg_memory_usage'] > 100 * 1024 * 1024:  # 100MB
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"high_memory_{function_name}_{int(time.time())}",
                function_name=function_name,
                issue_type="HIGH_MEMORY_USAGE",
                description=f"函数内存使用过高: {stats['avg_memory_usage'] / 1024 / 1024:.1f}MB",
                impact_level="MEDIUM",
                suggested_fix="优化数据结构和内存分配",
                estimated_improvement=0.2
            ))

        return suggestions

class BottleneckDetector:
    """性能瓶颈检测器"""

    def __init__(self):
        self.system_metrics = deque(maxlen=1000)
        self.bottlenecks = []

    def record_system_metrics(self):
        """记录系统指标"""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            metrics = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_read_speed': disk_io.read_bytes if disk_io else 0,
                'disk_write_speed': disk_io.write_bytes if disk_io else 0,
                'network_sent': net_io.bytes_sent if net_io else 0,
                'network_recv': net_io.bytes_recv if net_io else 0
            }

            self.system_metrics.append(metrics)

        except ImportError:
            logger.warning("psutil not available, system metrics disabled")

    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """检测系统瓶颈"""
        if len(self.system_metrics) < 10:
            return []

        recent_metrics = list(self.system_metrics)[-10:]
        bottlenecks = []

        # CPU瓶颈检测
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > 80:
            bottlenecks.append({
                'type': 'CPU_BOTTLENECK',
                'severity': 'HIGH' if avg_cpu > 90 else 'MEDIUM',
                'description': f'CPU使用率过高: {avg_cpu:.1f}%',
                'suggestion': '考虑优化CPU密集型操作或增加并行处理'
            })

        # 内存瓶颈检测
        avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        if avg_memory > 85:
            bottlenecks.append({
                'type': 'MEMORY_BOTTLENECK',
                'severity': 'HIGH' if avg_memory > 95 else 'MEDIUM',
                'description': f'内存使用率过高: {avg_memory:.1f}%',
                'suggestion': '优化内存分配或增加系统内存'
            })

        return bottlenecks

class AsyncOptimizer:
    """异步性能优化器主类"""

    def __init__(self):
        self.profiler = AsyncProfiler()
        self.scheduler = TaskScheduler()
        self.analyzer = PerformanceAnalyzer()
        self.bottleneck_detector = BottleneckDetector()

        # 并发控制器集合
        self.limiters = {}

        # 优化配置
        self.optimization_enabled = True
        self.auto_optimization = True

        # 监控任务
        self.monitoring_task = None
        self.is_running = False

        logger.info("异步性能优化器初始化完成")

    async def start(self):
        """启动优化器"""
        if self.is_running:
            return

        self.is_running = True

        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("异步性能优化器启动完成")

    async def stop(self):
        """停止优化器"""
        if not self.is_running:
            return

        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()

        logger.info("异步性能优化器已停止")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 记录系统指标
                self.bottleneck_detector.record_system_metrics()

                # 检测性能瓶颈
                bottlenecks = self.bottleneck_detector.detect_bottlenecks()
                for bottleneck in bottlenecks:
                    logger.warning(f"检测到性能瓶颈: {bottleneck['description']}")

                # 自动优化
                if self.auto_optimization:
                    await self._perform_auto_optimization()

                await asyncio.sleep(30)  # 30秒监控间隔

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(30)

    async def _perform_auto_optimization(self):
        """执行自动优化"""
        # 分析所有函数的性能
        for function_name in self.profiler.metrics_history.keys():
            suggestions = self.analyzer.analyze_function_performance(function_name)

            for suggestion in suggestions:
                if suggestion.impact_level == "HIGH":
                    logger.info(f"高影响优化建议: {suggestion.description}")
                    # 这里可以实现自动优化逻辑

    def create_limiter(self, name: str, max_concurrent: int,
                      time_window: int = 60) -> ConcurrencyLimiter:
        """创建并发限制器"""
        limiter = ConcurrencyLimiter(max_concurrent, time_window)
        self.limiters[name] = limiter
        return limiter

    def get_limiter(self, name: str) -> Optional[ConcurrencyLimiter]:
        """获取并发限制器"""
        return self.limiters.get(name)

    def profile_async(self, func_name: str = None):
        """异步函数性能分析装饰器"""
        return self.profiler.profile_async(func_name)

    async def schedule_task(self, coro: Awaitable, priority: int = 1,
                          task_name: str = None) -> Any:
        """调度异步任务"""
        return await self.scheduler.schedule_task(coro, priority, task_name)

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        # 收集所有函数的性能统计
        function_stats = {}
        for func_name in self.profiler.metrics_history.keys():
            function_stats[func_name] = self.profiler.get_function_stats(func_name)

        # 收集优化建议
        all_suggestions = []
        for func_name in self.profiler.metrics_history.keys():
            suggestions = self.analyzer.analyze_function_performance(func_name)
            all_suggestions.extend(suggestions)

        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_enabled': self.optimization_enabled,
            'auto_optimization': self.auto_optimization,
            'function_stats': function_stats,
            'scheduler_stats': self.scheduler.get_stats(),
            'optimization_suggestions': [s.to_dict() for s in all_suggestions],
            'bottlenecks': self.bottleneck_detector.detect_bottlenecks(),
            'limiter_usage': {
                name: limiter.get_current_usage()
                for name, limiter in self.limiters.items()
            }
        }

# 全局实例
_async_optimizer_instance = None

def get_async_optimizer() -> AsyncOptimizer:
    """获取异步优化器实例"""
    global _async_optimizer_instance
    if _async_optimizer_instance is None:
        _async_optimizer_instance = AsyncOptimizer()
    return _async_optimizer_instance

# 便利装饰器

def profile_async(func_name: str = None):
    """性能分析装饰器"""
    optimizer = get_async_optimizer()
    return optimizer.profile_async(func_name)

async def optimized_task(coro: Awaitable, priority: int = 1, task_name: str = None) -> Any:
    """优化的任务执行"""
    optimizer = get_async_optimizer()
    return await optimizer.schedule_task(coro, priority, task_name)
