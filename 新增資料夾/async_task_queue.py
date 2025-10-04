"""
Async Task Queue System
异步任务队列系统，提供高性能的任务调度和执行功能
支持优先级队列、任务重试和负载均衡
"""

import asyncio
import pickle
from typing import Dict, List, Optional, Any, Union, Callable, Coroutine
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
import uuid
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

class TaskPriority(IntEnum):
    """任务优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None

@dataclass
class AsyncTask:
    """异步任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Union[Callable, str] = None  # 函数或函数名
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    
    # 调度信息
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # 重试配置
    max_retries: int = 3
    current_retry: int = 0
    retry_delay: int = 60  # 秒
    
    # 执行配置
    timeout: Optional[int] = None
    executor_type: str = "async"  # async, thread, process
    
    # 依赖关系
    dependencies: List[str] = field(default_factory=list)
    
    # 结果
    result: Optional[TaskResult] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready(self) -> bool:
        """检查任务是否准备执行"""
        if self.status != TaskStatus.PENDING:
            return False
        
        # 检查调度时间
        if self.scheduled_at and datetime.now() < self.scheduled_at:
            return False
        
        # 检查截止时间
        if self.deadline and datetime.now() > self.deadline:
            self.status = TaskStatus.CANCELLED
            return False
        
        return True
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return (self.status == TaskStatus.FAILED and 
                self.current_retry < self.max_retries)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'max_retries': self.max_retries,
            'current_retry': self.current_retry,
            'timeout': self.timeout,
            'executor_type': self.executor_type,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
            'result': asdict(self.result) if self.result else None
        }

class TaskWorker:
    """任务执行器"""
    
    def __init__(self, worker_id: str, worker_type: str = "async"):
        self.worker_id = worker_id
        self.worker_type = worker_type
        self.is_busy = False
        self.current_task = None
        self.processed_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # 执行器
        if worker_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=1)
        elif worker_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=1)
        else:
            self.executor = None  # 异步执行
    
    async def execute_task(self, task: AsyncTask) -> TaskResult:
        """执行任务"""
        self.is_busy = True
        self.current_task = task
        self.last_activity = datetime.now()
        
        start_time = time.time()
        result = TaskResult(task_id=task.id, status=TaskStatus.RUNNING, worker_id=self.worker_id)
        result.started_at = datetime.now()
        
        try:
            # 根据执行器类型执行任务
            if task.executor_type == "async":
                task_result = await self._execute_async(task)
            elif task.executor_type == "thread":
                task_result = await self._execute_in_thread(task)
            elif task.executor_type == "process":
                task_result = await self._execute_in_process(task)
            else:
                raise ValueError(f"不支持的执行器类型: {task.executor_type}")
            
            result.result = task_result
            result.status = TaskStatus.COMPLETED
            self.processed_count += 1
            
        except asyncio.TimeoutError:
            result.error = "任务执行超时"
            result.status = TaskStatus.FAILED
            self.error_count += 1
            
        except Exception as e:
            result.error = str(e)
            result.status = TaskStatus.FAILED
            self.error_count += 1
            logger.error(f"任务执行失败 {task.id}: {e}")
        
        finally:
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.completed_at = datetime.now()
            
            self.total_execution_time += execution_time
            self.is_busy = False
            self.current_task = None
            self.last_activity = datetime.now()
        
        return result
    
    async def _execute_async(self, task: AsyncTask) -> Any:
        """异步执行任务"""
        func = task.func
        if isinstance(func, str):
            # 如果是字符串，尝试解析为函数
            func = self._resolve_function(func)
        
        if asyncio.iscoroutinefunction(func):
            if task.timeout:
                return await asyncio.wait_for(
                    func(*task.args, **task.kwargs), 
                    timeout=task.timeout
                )
            else:
                return await func(*task.args, **task.kwargs)
        else:
            # 同步函数包装为异步
            return func(*task.args, **task.kwargs)
    
    async def _execute_in_thread(self, task: AsyncTask) -> Any:
        """在线程中执行任务"""
        func = task.func
        if isinstance(func, str):
            func = self._resolve_function(func)
        
        loop = asyncio.get_event_loop()
        
        if task.timeout:
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, func, *task.args),
                timeout=task.timeout
            )
        else:
            return await loop.run_in_executor(self.executor, func, *task.args)
    
    async def _execute_in_process(self, task: AsyncTask) -> Any:
        """在进程中执行任务"""
        func = task.func
        if isinstance(func, str):
            func = self._resolve_function(func)
        
        loop = asyncio.get_event_loop()
        
        if task.timeout:
            return await asyncio.wait_for(
                loop.run_in_executor(self.executor, func, *task.args),
                timeout=task.timeout
            )
        else:
            return await loop.run_in_executor(self.executor, func, *task.args)
    
    def _resolve_function(self, func_name: str) -> Callable:
        """解析函数名为函数对象"""
        # 这里可以实现函数名到函数对象的映射
        # 简化实现，实际应用中可以从注册表中查找
        module_name, function_name = func_name.rsplit('.', 1)
        module = __import__(module_name, fromlist=[function_name])
        return getattr(module, function_name)
    
    def get_stats(self) -> dict:
        """获取工作器统计信息"""
        uptime = (datetime.now() - self.created_at).total_seconds()
        avg_execution_time = (self.total_execution_time / max(self.processed_count, 1))
        
        return {
            'worker_id': self.worker_id,
            'worker_type': self.worker_type,
            'is_busy': self.is_busy,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count - self.error_count) / max(self.processed_count, 1),
            'avg_execution_time': avg_execution_time,
            'total_execution_time': self.total_execution_time,
            'uptime_seconds': uptime,
            'current_task_id': self.current_task.id if self.current_task else None,
            'last_activity': self.last_activity.isoformat()
        }
    
    async def cleanup(self):
        """清理工作器"""
        if self.executor:
            self.executor.shutdown(wait=True)

class AsyncTaskQueue:
    """异步任务队列主类"""
    
    def __init__(self, name: str = "default", max_workers: int = 10):
        self.name = name
        self.max_workers = max_workers
        
        # 任务存储
        self.tasks = {}  # task_id -> AsyncTask
        self.priority_queues = {
            TaskPriority.URGENT: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        
        # 工作器管理
        self.workers = {}  # worker_id -> TaskWorker
        self.available_workers = deque()
        
        # 任务依赖图
        self.dependency_graph = defaultdict(set)  # task_id -> {dependent_task_ids}
        
        # 调度器
        self.scheduler_task = None
        self.retry_task = None
        self.cleanup_task = None
        self.is_running = False
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'queue_sizes': {},
            'start_time': datetime.now()
        }
        
        # 任务结果存储
        self.task_results = {}  # task_id -> TaskResult
        self.result_ttl = 3600  # 结果保存1小时
        
        # 回调函数
        self.on_task_complete_callbacks = []
        self.on_task_error_callbacks = []
        
        logger.info(f"异步任务队列初始化: {name}")
    
    async def start(self):
        """启动任务队列"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 初始化工作器
        await self._initialize_workers()
        
        # 启动调度器
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.retry_task = asyncio.create_task(self._retry_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"任务队列 {self.name} 启动完成")
    
    async def stop(self):
        """停止任务队列"""
        if not self.is_running:
            return
        
        logger.info(f"正在停止任务队列: {self.name}")
        self.is_running = False
        
        # 取消调度任务
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        if self.retry_task:
            self.retry_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # 等待正在执行的任务完成
        busy_workers = [worker for worker in self.workers.values() if worker.is_busy]
        if busy_workers:
            logger.info(f"等待 {len(busy_workers)} 个工作器完成任务...")
            await asyncio.sleep(1)  # 给工作器一些时间完成
        
        # 清理工作器
        for worker in self.workers.values():
            await worker.cleanup()
        
        logger.info(f"任务队列 {self.name} 已停止")
    
    async def _initialize_workers(self):
        """初始化工作器"""
        for i in range(self.max_workers):
            worker_id = f"{self.name}_worker_{i}"
            worker = TaskWorker(worker_id)
            self.workers[worker_id] = worker
            self.available_workers.append(worker_id)
    
    async def submit_task(self, func: Union[Callable, str], *args, 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         name: str = "", scheduled_at: datetime = None,
                         deadline: datetime = None, timeout: int = None,
                         max_retries: int = 3, dependencies: List[str] = None,
                         executor_type: str = "async", metadata: dict = None,
                         **kwargs) -> str:
        """提交任务"""
        task = AsyncTask(
            name=name or (func.__name__ if callable(func) else str(func)),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            scheduled_at=scheduled_at,
            deadline=deadline,
            timeout=timeout,
            max_retries=max_retries,
            dependencies=dependencies or [],
            executor_type=executor_type,
            metadata=metadata or {}
        )
        
        # 添加到任务存储
        self.tasks[task.id] = task
        
        # 构建依赖图
        for dep_task_id in task.dependencies:
            self.dependency_graph[dep_task_id].add(task.id)
        
        # 如果任务准备就绪，加入队列
        if task.is_ready() and self._dependencies_satisfied(task):
            self.priority_queues[task.priority].append(task.id)
        
        # 更新统计
        self.stats['total_tasks'] += 1
        self._update_queue_stats()
        
        logger.debug(f"提交任务: {task.id} ({task.name})")
        return task.id
    
    def _dependencies_satisfied(self, task: AsyncTask) -> bool:
        """检查任务依赖是否满足"""
        for dep_task_id in task.dependencies:
            dep_task = self.tasks.get(dep_task_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def _scheduler_loop(self):
        """任务调度循环"""
        while self.is_running:
            try:
                await self._schedule_tasks()
                await asyncio.sleep(0.1)  # 100ms调度间隔
            except Exception as e:
                logger.error(f"调度循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _schedule_tasks(self):
        """调度任务"""
        if not self.available_workers:
            return
        
        # 按优先级顺序处理任务
        for priority in [TaskPriority.URGENT, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self.priority_queues[priority]
            
            while queue and self.available_workers:
                task_id = queue.popleft()
                task = self.tasks.get(task_id)
                
                if not task or not task.is_ready():
                    continue
                
                # 检查依赖
                if not self._dependencies_satisfied(task):
                    continue
                
                # 分配工作器
                worker_id = self.available_workers.popleft()
                worker = self.workers[worker_id]
                
                # 异步执行任务
                asyncio.create_task(self._execute_task_with_worker(worker, task))
    
    async def _execute_task_with_worker(self, worker: TaskWorker, task: AsyncTask):
        """使用工作器执行任务"""
        try:
            task.status = TaskStatus.RUNNING
            result = await worker.execute_task(task)
            
            # 保存结果
            task.result = result
            task.status = result.status
            self.task_results[task.id] = result
            
            # 更新统计
            if result.status == TaskStatus.COMPLETED:
                self.stats['completed_tasks'] += 1
                
                # 检查并调度依赖任务
                await self._check_dependent_tasks(task.id)
                
                # 调用完成回调
                for callback in self.on_task_complete_callbacks:
                    try:
                        await callback(task, result)
                    except Exception as e:
                        logger.error(f"任务完成回调错误: {e}")
                        
            elif result.status == TaskStatus.FAILED:
                self.stats['failed_tasks'] += 1
                
                # 调用错误回调
                for callback in self.on_task_error_callbacks:
                    try:
                        await callback(task, result)
                    except Exception as e:
                        logger.error(f"任务错误回调错误: {e}")
            
            self._update_queue_stats()
            
        finally:
            # 归还工作器
            self.available_workers.append(worker.worker_id)
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """检查并调度依赖任务"""
        dependent_task_ids = self.dependency_graph.get(completed_task_id, set())
        
        for task_id in dependent_task_ids:
            task = self.tasks.get(task_id)
            
            if (task and task.status == TaskStatus.PENDING and 
                task.is_ready() and self._dependencies_satisfied(task)):
                
                self.priority_queues[task.priority].append(task_id)
    
    async def _retry_loop(self):
        """重试循环"""
        while self.is_running:
            try:
                await self._process_retries()
                await asyncio.sleep(30)  # 30秒检查一次重试
            except Exception as e:
                logger.error(f"重试循环错误: {e}")
                await asyncio.sleep(30)
    
    async def _process_retries(self):
        """处理重试任务"""
        current_time = datetime.now()
        
        for task in self.tasks.values():
            if (task.should_retry() and 
                task.result and 
                task.result.completed_at and
                (current_time - task.result.completed_at).total_seconds() >= task.retry_delay):
                
                # 重置任务状态
                task.status = TaskStatus.PENDING
                task.current_retry += 1
                task.result = None
                
                # 重新加入队列
                if task.is_ready() and self._dependencies_satisfied(task):
                    self.priority_queues[task.priority].append(task.id)
                
                logger.info(f"重试任务: {task.id} (第{task.current_retry}次)")
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self.is_running:
            try:
                await self._cleanup_expired_results()
                await asyncio.sleep(300)  # 5分钟清理一次
            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_expired_results(self):
        """清理过期结果"""
        current_time = datetime.now()
        expired_task_ids = []
        
        for task_id, result in self.task_results.items():
            if (result.completed_at and 
                (current_time - result.completed_at).total_seconds() > self.result_ttl):
                expired_task_ids.append(task_id)
        
        for task_id in expired_task_ids:
            del self.task_results[task_id]
            if task_id in self.tasks:
                del self.tasks[task_id]
        
        if expired_task_ids:
            logger.debug(f"清理 {len(expired_task_ids)} 个过期任务结果")
    
    def _update_queue_stats(self):
        """更新队列统计"""
        self.stats['queue_sizes'] = {
            priority.name: len(queue)
            for priority, queue in self.priority_queues.items()
        }
    
    # 查询和管理方法
    def get_task(self, task_id: str) -> Optional[AsyncTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self.task_results.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.stats['cancelled_tasks'] += 1
            return True
        
        return False
    
    def get_stats(self) -> dict:
        """获取队列统计"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        worker_stats = {
            worker_id: worker.get_stats()
            for worker_id, worker in self.workers.items()
        }
        
        return {
            'queue_name': self.name,
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'max_workers': self.max_workers,
            'available_workers': len(self.available_workers),
            'task_stats': self.stats,
            'worker_stats': worker_stats,
            'current_queue_sizes': self.stats['queue_sizes']
        }
    
    # 回调注册
    def on_task_complete(self, callback: Callable):
        """注册任务完成回调"""
        self.on_task_complete_callbacks.append(callback)
    
    def on_task_error(self, callback: Callable):
        """注册任务错误回调"""
        self.on_task_error_callbacks.append(callback)

# 全局任务队列管理器
class TaskQueueManager:
    """任务队列管理器"""
    
    def __init__(self):
        self.queues = {}  # queue_name -> AsyncTaskQueue
        self.is_running = False
        
        logger.info("任务队列管理器初始化完成")
    
    async def start(self):
        """启动管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动所有队列
        for queue in self.queues.values():
            await queue.start()
        
        logger.info("任务队列管理器启动完成")
    
    async def stop(self):
        """停止管理器"""
        if not self.is_running:
            return
        
        logger.info("正在停止任务队列管理器...")
        self.is_running = False
        
        # 停止所有队列
        for queue in self.queues.values():
            await queue.stop()
        
        logger.info("任务队列管理器已停止")
    
    def create_queue(self, name: str, max_workers: int = 10) -> AsyncTaskQueue:
        """创建任务队列"""
        if name in self.queues:
            return self.queues[name]
        
        queue = AsyncTaskQueue(name, max_workers)
        self.queues[name] = queue
        
        if self.is_running:
            asyncio.create_task(queue.start())
        
        logger.info(f"创建任务队列: {name}")
        return queue
    
    def get_queue(self, name: str) -> Optional[AsyncTaskQueue]:
        """获取任务队列"""
        return self.queues.get(name)
    
    async def remove_queue(self, name: str):
        """移除任务队列"""
        if name in self.queues:
            queue = self.queues[name]
            await queue.stop()
            del self.queues[name]
            logger.info(f"移除任务队列: {name}")
    
    def get_stats(self) -> dict:
        """获取管理器统计"""
        return {
            'is_running': self.is_running,
            'queue_count': len(self.queues),
            'queue_stats': {
                name: queue.get_stats()
                for name, queue in self.queues.items()
            }
        }

# 全局实例
_task_queue_manager = None

def get_task_queue_manager() -> TaskQueueManager:
    """获取任务队列管理器实例"""
    global _task_queue_manager
    if _task_queue_manager is None:
        _task_queue_manager = TaskQueueManager()
    return _task_queue_manager

def get_default_queue() -> AsyncTaskQueue:
    """获取默认任务队列"""
    manager = get_task_queue_manager()
    return manager.create_queue("default")