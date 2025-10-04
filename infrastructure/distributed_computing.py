"""
Distributed Computing
分布式计算系统，支持多节点计算任务分发和并行处理
实现任务调度、负载均衡、容错恢复和资源管理功能
"""

import asyncio
import aiohttp
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
import socket
import pickle
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import zmq
import redis
import requests
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"          # 等待中
    RUNNING = "running"          # 运行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"      # 已取消
    TIMEOUT = "timeout"         # 超时

class NodeStatus(Enum):
    """节点状态"""
    ONLINE = "online"           # 在线
    OFFLINE = "offline"         # 离线
    BUSY = "busy"              # 忙碌
    ERROR = "error"            # 错误
    MAINTENANCE = "maintenance" # 维护中

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass

class ComputeTask:
    """计算任务"""
    task_id: str
    task_type: str
    function_name: str

    # 任务数据
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # 任务配置
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: int = 3600  # 超时时间（秒）
    retry_count: int = 3  # 重试次数

    # 资源需求
    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_required: bool = False

    # 状态信息
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # 执行信息
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0

    # 依赖关系
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'function_name': self.function_name,
            'priority': self.priority.value,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'resource_requirements': {
                'cpu_cores': self.cpu_cores,
                'memory_mb': self.memory_mb,
                'gpu_required': self.gpu_required
            },
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'assigned_node': self.assigned_node,
            'execution_time': self.execution_time,
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'has_result': self.result is not None,
            'error_message': self.error_message
        }

@dataclass

class ComputeNode:
    """计算节点"""
    node_id: str
    host: str
    port: int

    # 资源信息
    cpu_cores: int
    memory_mb: int
    gpu_count: int = 0

    # 状态信息
    status: NodeStatus = NodeStatus.OFFLINE
    last_heartbeat: datetime = field(default_factory=datetime.now)

    # 负载信息
    current_tasks: int = 0
    max_concurrent_tasks: int = 4
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    # 统计信息
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'resources': {
                'cpu_cores': self.cpu_cores,
                'memory_mb': self.memory_mb,
                'gpu_count': self.gpu_count
            },
            'status': self.status.value,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'load': {
                'current_tasks': self.current_tasks,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'cpu_usage': self.cpu_usage,
                'memory_usage': self.memory_usage
            },
            'statistics': {
                'total_tasks_completed': self.total_tasks_completed,
                'total_tasks_failed': self.total_tasks_failed,
                'average_task_time': self.average_task_time
            }
        }

class TaskQueue:
    """任务队列"""

    def __init__(self, use_redis: bool = False, redis_host: str = 'localhost', redis_port: int = 6379):
        self.use_redis = use_redis
        self.tasks = {}  # task_id -> ComputeTask
        self.pending_queue = queue.PriorityQueue()

        # Redis连接
        self.redis_client = None
        if use_redis:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_client.ping()
                logger.info("连接到Redis任务队列")
            except Exception as e:
                logger.error(f"Redis连接失败: {e}")
                self.use_redis = False

    def add_task(self, task: ComputeTask):
        """添加任务"""
        self.tasks[task.task_id] = task

        if self.use_redis and self.redis_client:
            # 存储到Redis
            self.redis_client.hset('tasks', task.task_id, pickle.dumps(task))
            self.redis_client.zadd('pending_tasks', {task.task_id: -task.priority.value})
        else:
            # 存储到本地队列（优先级队列，负数使高优先级排在前面）
            self.pending_queue.put((-task.priority.value, task.created_at.timestamp(), task.task_id))

    def get_next_task(self) -> Optional[ComputeTask]:
        """获取下一个任务"""
        if self.use_redis and self.redis_client:
            # 从Redis获取
            result = self.redis_client.zpopmin('pending_tasks', 1)
            if result:
                task_id = result[0][0]
                task_data = self.redis_client.hget('tasks', task_id)
                if task_data:
                    return pickle.loads(task_data)
        else:
            # 从本地队列获取
            try:
                _, _, task_id = self.pending_queue.get_nowait()
                return self.tasks.get(task_id)
            except queue.Empty:
                pass

        return None

    def update_task_status(self, task_id: str, status: TaskStatus, **kwargs):
        """更新任务状态"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status

            # 更新其他字段
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

            # 同步到Redis
            if self.use_redis and self.redis_client:
                self.redis_client.hset('tasks', task_id, pickle.dumps(task))

    def get_task(self, task_id: str) -> Optional[ComputeTask]:
        """获取任务"""
        if task_id in self.tasks:
            return self.tasks[task_id]

        if self.use_redis and self.redis_client:
            task_data = self.redis_client.hget('tasks', task_id)
            if task_data:
                task = pickle.loads(task_data)
                self.tasks[task_id] = task
                return task

        return None

    def get_tasks_by_status(self, status: TaskStatus) -> List[ComputeTask]:
        """按状态获取任务"""
        return [task for task in self.tasks.values() if task.status == status]

class NodeManager:
    """节点管理器"""

    def __init__(self):
        self.nodes = {}  # node_id -> ComputeNode
        self.heartbeat_timeout = 30  # 心跳超时时间（秒）

    def register_node(self, node: ComputeNode):
        """注册节点"""
        self.nodes[node.node_id] = node
        node.status = NodeStatus.ONLINE
        logger.info(f"注册计算节点: {node.node_id} ({node.host}:{node.port})")

    def update_heartbeat(self, node_id: str, cpu_usage: float = 0.0,
                        memory_usage: float = 0.0, current_tasks: int = 0):
        """更新心跳"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_heartbeat = datetime.now()
            node.cpu_usage = cpu_usage
            node.memory_usage = memory_usage
            node.current_tasks = current_tasks

            if node.status == NodeStatus.OFFLINE:
                node.status = NodeStatus.ONLINE
                logger.info(f"节点 {node_id} 重新上线")

    def check_node_health(self):
        """检查节点健康状态"""
        current_time = datetime.now()

        for node in self.nodes.values():
            time_since_heartbeat = (current_time - node.last_heartbeat).total_seconds()

            if time_since_heartbeat > self.heartbeat_timeout:
                if node.status == NodeStatus.ONLINE:
                    node.status = NodeStatus.OFFLINE
                    logger.warning(f"节点 {node.node_id} 离线")

    def get_available_nodes(self) -> List[ComputeNode]:
        """获取可用节点"""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.ONLINE and
               node.current_tasks < node.max_concurrent_tasks
        ]

    def select_best_node(self, task: ComputeTask) -> Optional[ComputeNode]:
        """选择最佳节点"""
        available_nodes = self.get_available_nodes()

        if not available_nodes:
            return None

        # 过滤满足资源需求的节点
        suitable_nodes = []
        for node in available_nodes:
            if (node.cpu_cores >= task.cpu_cores and
                node.memory_mb >= task.memory_mb and
                (not task.gpu_required or node.gpu_count > 0)):
                suitable_nodes.append(node)

        if not suitable_nodes:
            return None

        # 选择负载最低的节点
        best_node = min(suitable_nodes,
                       key=lambda n: (n.current_tasks / n.max_concurrent_tasks, n.cpu_usage))

        return best_node

class TaskScheduler:
    """任务调度器"""

    def __init__(self, task_queue: TaskQueue, node_manager: NodeManager):
        self.task_queue = task_queue
        self.node_manager = node_manager
        self.running_tasks = {}  # task_id -> (node_id, start_time)
        self.is_running = False

    async def start(self):
        """启动调度器"""
        self.is_running = True
        logger.info("任务调度器启动")

        # 启动调度循环
        asyncio.create_task(self._scheduling_loop())
        asyncio.create_task(self._health_check_loop())

    async def stop(self):
        """停止调度器"""
        self.is_running = False
        logger.info("任务调度器停止")

    async def _scheduling_loop(self):
        """调度循环"""
        while self.is_running:
            try:
                # 检查是否有待执行的任务
                task = self.task_queue.get_next_task()

                if task and task.status == TaskStatus.PENDING:
                    # 检查依赖关系
                    if self._check_dependencies(task):
                        # 选择执行节点
                        node = self.node_manager.select_best_node(task)

                        if node:
                            # 分配任务到节点
                            await self._assign_task_to_node(task, node)
                        else:
                            # 没有可用节点，任务继续等待
                            pass

                # 检查运行中的任务
                await self._check_running_tasks()

                await asyncio.sleep(1)  # 避免过度消耗CPU

            except Exception as e:
                logger.error(f"调度循环错误: {e}")
                await asyncio.sleep(5)

    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                self.node_manager.check_node_health()
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                await asyncio.sleep(10)

    def _check_dependencies(self, task: ComputeTask) -> bool:
        """检查任务依赖"""
        for dep_task_id in task.dependencies:
            dep_task = self.task_queue.get_task(dep_task_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    async def _assign_task_to_node(self, task: ComputeTask, node: ComputeNode):
        """分配任务到节点"""
        try:
            # 发送任务到节点
            success = await self._send_task_to_node(task, node)

            if success:
                # 更新任务状态
                self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.RUNNING,
                    assigned_node=node.node_id,
                    started_at=datetime.now()
                )

                # 记录运行中的任务
                self.running_tasks[task.task_id] = (node.node_id, datetime.now())

                # 更新节点状态
                node.current_tasks += 1

                logger.info(f"任务 {task.task_id} 分配到节点 {node.node_id}")
            else:
                logger.error(f"任务 {task.task_id} 分配到节点 {node.node_id} 失败")

        except Exception as e:
            logger.error(f"分配任务失败: {e}")

    async def _send_task_to_node(self, task: ComputeTask, node: ComputeNode) -> bool:
        """发送任务到节点"""
        try:
            # 构建任务数据
            task_data = {
                'task_id': task.task_id,
                'function_name': task.function_name,
                'args': task.args,
                'kwargs': task.kwargs,
                'timeout': task.timeout
            }

            # 发送HTTP请求到计算节点
            async with aiohttp.ClientSession() as session:
                url = f"http://{node.host}:{node.port}/execute_task"
                async with session.post(url, json=task_data, timeout=10) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"节点 {node.node_id} 拒绝任务: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"发送任务到节点失败: {e}")
            return False

    async def _check_running_tasks(self):
        """检查运行中的任务"""
        current_time = datetime.now()

        for task_id, (node_id, start_time) in list(self.running_tasks.items()):
            task = self.task_queue.get_task(task_id)

            if not task:
                continue

            # 检查超时
            if (current_time - start_time).total_seconds() > task.timeout:
                await self._handle_task_timeout(task_id, node_id)
                continue

            # 检查任务状态（从节点获取）
            await self._check_task_status_from_node(task_id, node_id)

    async def _handle_task_timeout(self, task_id: str, node_id: str):
        """处理任务超时"""
        logger.warning(f"任务 {task_id} 在节点 {node_id} 超时")

        # 更新任务状态
        self.task_queue.update_task_status(
            task_id,
            TaskStatus.TIMEOUT,
            completed_at=datetime.now(),
            error_message="Task timeout"
        )

        # 从运行列表中移除
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]

        # 更新节点状态
        if node_id in self.node_manager.nodes:
            self.node_manager.nodes[node_id].current_tasks -= 1

    async def _check_task_status_from_node(self, task_id: str, node_id: str):
        """从节点检查任务状态"""
        try:
            node = self.node_manager.nodes.get(node_id)
            if not node:
                return

            # 查询节点上的任务状态
            async with aiohttp.ClientSession() as session:
                url = f"http://{node.host}:{node.port}/task_status/{task_id}"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        await self._handle_task_status_update(task_id, node_id, status_data)

        except Exception as e:
            logger.error(f"检查任务状态失败: {e}")

    async def _handle_task_status_update(self, task_id: str, node_id: str, status_data: dict):
        """处理任务状态更新"""
        status = TaskStatus(status_data.get('status', 'running'))

        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            # 任务完成或失败
            self.task_queue.update_task_status(
                task_id,
                status,
                completed_at=datetime.now(),
                result=status_data.get('result'),
                error_message=status_data.get('error'),
                execution_time=status_data.get('execution_time', 0)
            )

            # 从运行列表中移除
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

            # 更新节点状态
            if node_id in self.node_manager.nodes:
                node = self.node_manager.nodes[node_id]
                node.current_tasks -= 1

                if status == TaskStatus.COMPLETED:
                    node.total_tasks_completed += 1
                else:
                    node.total_tasks_failed += 1

                # 更新平均执行时间
                exec_time = status_data.get('execution_time', 0)
                if exec_time > 0:
                    total_tasks = node.total_tasks_completed + node.total_tasks_failed
                    node.average_task_time = (
                        (node.average_task_time * (total_tasks - 1) + exec_time) / total_tasks
                    )

            logger.info(f"任务 {task_id} 在节点 {node_id} {status.value}")

class DistributedComputingSystem:
    """分布式计算系统主类"""

    def __init__(self, use_redis: bool = False):
        self.task_queue = TaskQueue(use_redis=use_redis)
        self.node_manager = NodeManager()
        self.scheduler = TaskScheduler(self.task_queue, self.node_manager)

        # 注册的函数
        self.registered_functions = {}

        # 统计信息
        self.stats = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_nodes_registered': 0,
            'average_task_completion_time': 0.0
        }

        logger.info("分布式计算系统初始化完成")

    def register_function(self, name: str, func: Callable):
        """注册可执行函数"""
        self.registered_functions[name] = func
        logger.info(f"注册函数: {name}")

    async def start_system(self):
        """启动系统"""
        await self.scheduler.start()
        logger.info("分布式计算系统启动")

    async def stop_system(self):
        """停止系统"""
        await self.scheduler.stop()
        logger.info("分布式计算系统停止")

    def submit_task(self, function_name: str, *args,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: int = 3600, **kwargs) -> str:
        """提交任务"""

        if function_name not in self.registered_functions:
            raise ValueError(f"未注册的函数: {function_name}")

        task_id = f"task_{uuid.uuid4().hex}"

        task = ComputeTask(
            task_id=task_id,
            task_type="function_call",
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )

        self.task_queue.add_task(task)
        self.stats['total_tasks_submitted'] += 1

        logger.info(f"提交任务: {task_id} ({function_name})")
        return task_id

    def submit_batch_tasks(self, tasks_config: List[Dict[str, Any]]) -> List[str]:
        """批量提交任务"""
        task_ids = []

        for config in tasks_config:
            function_name = config['function_name']
            args = config.get('args', ())
            kwargs = config.get('kwargs', {})
            priority = TaskPriority(config.get('priority', TaskPriority.NORMAL.value))
            timeout = config.get('timeout', 3600)

            task_id = self.submit_task(function_name, *args,
                                     priority=priority, timeout=timeout, **kwargs)
            task_ids.append(task_id)

        return task_ids

    def get_task_result(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """获取任务结果"""
        task = self.task_queue.get_task(task_id)

        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        # 等待任务完成
        start_time = time.time()
        while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED,
                                TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
            time.sleep(0.1)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务结果超时: {task_id}")

            # 刷新任务状态
            task = self.task_queue.get_task(task_id)

        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise Exception(f"任务执行失败: {task.error_message}")
        elif task.status == TaskStatus.TIMEOUT:
            raise TimeoutError(f"任务执行超时: {task_id}")
        else:
            raise Exception(f"任务被取消: {task_id}")

    async def get_task_result_async(self, task_id: str, timeout: Optional[int] = None) -> Any:
        """异步获取任务结果"""
        task = self.task_queue.get_task(task_id)

        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        # 异步等待任务完成
        start_time = time.time()
        while task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED,
                                TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
            await asyncio.sleep(0.1)

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务结果超时: {task_id}")

            # 刷新任务状态
            task = self.task_queue.get_task(task_id)

        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise Exception(f"任务执行失败: {task.error_message}")
        elif task.status == TaskStatus.TIMEOUT:
            raise TimeoutError(f"任务执行超时: {task_id}")
        else:
            raise Exception(f"任务被取消: {task_id}")

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.task_queue.get_task(task_id)

        if not task:
            return False

        if task.status == TaskStatus.PENDING:
            self.task_queue.update_task_status(task_id, TaskStatus.CANCELLED)
            return True
        elif task.status == TaskStatus.RUNNING:
            # 通知节点取消任务
            # TODO: 实现运行中任务的取消
            return False

        return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.task_queue.get_task(task_id)

        if not task:
            return None

        return task.to_dict()

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        # 更新统计信息
        completed_tasks = self.task_queue.get_tasks_by_status(TaskStatus.COMPLETED)
        failed_tasks = self.task_queue.get_tasks_by_status(TaskStatus.FAILED)

        self.stats['total_tasks_completed'] = len(completed_tasks)
        self.stats['total_tasks_failed'] = len(failed_tasks)
        self.stats['total_nodes_registered'] = len(self.node_manager.nodes)

        # 计算平均完成时间
        if completed_tasks:
            completion_times = [t.execution_time for t in completed_tasks if t.execution_time > 0]
            if completion_times:
                self.stats['average_task_completion_time'] = sum(completion_times) / len(completion_times)

        return {
            'system_stats': self.stats,
            'node_stats': {
                'total_nodes': len(self.node_manager.nodes),
                'online_nodes': len([n for n in self.node_manager.nodes.values() if n.status == NodeStatus.ONLINE]),
                'node_details': [node.to_dict() for node in self.node_manager.nodes.values()]
            },
            'task_stats': {
                'pending': len(self.task_queue.get_tasks_by_status(TaskStatus.PENDING)),
                'running': len(self.task_queue.get_tasks_by_status(TaskStatus.RUNNING)),
                'completed': len(completed_tasks),
                'failed': len(failed_tasks),
                'cancelled': len(self.task_queue.get_tasks_by_status(TaskStatus.CANCELLED))
            },
            'registered_functions': list(self.registered_functions.keys())
        }

# 计算节点实现

class ComputeNodeServer:
    """计算节点服务器"""

    def __init__(self, node_id: str, host: str = 'localhost', port: int = 8000):
        self.node_id = node_id
        self.host = host
        self.port = port

        # 节点资源信息
        self.cpu_cores = mp.cpu_count()
        self.memory_mb = 4096  # 假设4GB内存
        self.gpu_count = 0

        # 任务执行器
        self.executor = ProcessPoolExecutor(max_workers=self.cpu_cores)
        self.running_tasks = {}

        # 注册的函数
        self.registered_functions = {}

    def register_function(self, name: str, func: Callable):
        """注册可执行函数"""
        self.registered_functions[name] = func

    async def start_server(self):
        """启动节点服务器"""
        from aiohttp import web

        app = web.Application()
        app.router.add_post('/execute_task', self.handle_execute_task)
        app.router.add_get('/task_status/{task_id}', self.handle_task_status)
        app.router.add_post('/heartbeat', self.handle_heartbeat)

        runner = web.AppRunner(app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"计算节点 {self.node_id} 启动在 {self.host}:{self.port}")

    async def handle_execute_task(self, request):
        """处理任务执行请求"""
        from aiohttp import web

        try:
            task_data = await request.json()
            task_id = task_data['task_id']
            function_name = task_data['function_name']
            args = task_data.get('args', [])
            kwargs = task_data.get('kwargs', {})

            if function_name not in self.registered_functions:
                return web.json_response({'error': f'Unknown function: {function_name}'}, status=400)

            # 提交任务到执行器
            func = self.registered_functions[function_name]
            future = self.executor.submit(func, *args, **kwargs)

            self.running_tasks[task_id] = {
                'future': future,
                'start_time': time.time(),
                'status': 'running'
            }

            return web.json_response({'status': 'accepted'})

        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

    async def handle_task_status(self, request):
        """处理任务状态查询"""
        from aiohttp import web

        task_id = request.match_info['task_id']

        if task_id not in self.running_tasks:
            return web.json_response({'error': 'Task not found'}, status=404)

        task_info = self.running_tasks[task_id]
        future = task_info['future']

        if future.done():
            try:
                result = future.result()
                execution_time = time.time() - task_info['start_time']

                # 清理完成的任务
                del self.running_tasks[task_id]

                return web.json_response({
                    'status': 'completed',
                    'result': result,
                    'execution_time': execution_time
                })
            except Exception as e:
                # 清理failed任务
                del self.running_tasks[task_id]

                return web.json_response({
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': time.time() - task_info['start_time']
                })
        else:
            return web.json_response({
                'status': 'running',
                'execution_time': time.time() - task_info['start_time']
            })

    async def handle_heartbeat(self, request):
        """处理心跳请求"""
        from aiohttp import web
        import psutil

        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            current_tasks = len(self.running_tasks)

            return web.json_response({
                'node_id': self.node_id,
                'status': 'online',
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'current_tasks': current_tasks,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)

# 便利函数

def create_local_compute_node(node_id: str, port: int = 8000) -> ComputeNodeServer:
    """创建本地计算节点"""
    node = ComputeNodeServer(node_id, 'localhost', port)

    # 注册一些示例函数

    def compute_sum(numbers):
        return sum(numbers)

    def compute_mean(numbers):
        return sum(numbers) / len(numbers) if numbers else 0

    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    node.register_function('sum', compute_sum)
    node.register_function('mean', compute_mean)
    node.register_function('fibonacci', fibonacci)

    return node

# 全局实例
_distributed_computing_system_instance = None

def get_distributed_computing_system() -> DistributedComputingSystem:
    """获取分布式计算系统实例"""
    global _distributed_computing_system_instance
    if _distributed_computing_system_instance is None:
        _distributed_computing_system_instance = DistributedComputingSystem()
    return _distributed_computing_system_instance
