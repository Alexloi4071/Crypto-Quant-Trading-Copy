"""
Signal Hooks
信号钩子系统，为信号处理提供事件驱动的扩展机制
支持信号生成、验证、过滤和转换的钩子点
"""

from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import asyncio
import functools

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class SignalHookType(Enum):
    """信号钩子类型"""
    # 信号生成阶段
    BEFORE_SIGNAL_GENERATION = "before_signal_generation"
    AFTER_SIGNAL_GENERATION = "after_signal_generation"

    # 信号处理阶段
    SIGNAL_CREATED = "signal_created"
    SIGNAL_VALIDATED = "signal_validated"
    SIGNAL_FILTERED = "signal_filtered"
    SIGNAL_TRANSFORMED = "signal_transformed"

    # 信号聚合阶段
    SIGNALS_AGGREGATED = "signals_aggregated"
    SIGNAL_CONFLICT_DETECTED = "signal_conflict_detected"

    # 信号执行阶段
    BEFORE_SIGNAL_EXECUTION = "before_signal_execution"
    SIGNAL_EXECUTION_SUCCESS = "signal_execution_success"
    SIGNAL_EXECUTION_FAILED = "signal_execution_failed"

    # 信号状态变化
    SIGNAL_EXPIRED = "signal_expired"
    SIGNAL_CANCELLED = "signal_cancelled"
    SIGNAL_UPDATED = "signal_updated"

class HookPriority(Enum):
    """钩子优先级"""
    HIGHEST = 1
    HIGH = 10
    NORMAL = 50
    LOW = 90
    LOWEST = 100

@dataclass

class HookResult:
    """钩子执行结果"""
    hook_id: str
    plugin_id: str
    success: bool
    execution_time: float
    result_data: Any = None
    error_message: Optional[str] = None
    modified_data: Any = None  # 修改后的数据

    def to_dict(self) -> dict:
        return {
            'hook_id': self.hook_id,
            'plugin_id': self.plugin_id,
            'success': self.success,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'has_modified_data': self.modified_data is not None
        }

@dataclass

class SignalHookContext:
    """信号钩子上下文"""
    hook_type: SignalHookType
    symbol: str
    timestamp: datetime

    # 原始数据
    original_signal: Any = None
    market_data: Dict[str, Any] = field(default_factory=dict)

    # 处理数据
    current_signal: Any = None
    processing_results: List[HookResult] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: HookResult):
        """添加钩子执行结果"""
        self.processing_results.append(result)

    def get_results_by_plugin(self, plugin_id: str) -> List[HookResult]:
        """获取特定插件的结果"""
        return [r for r in self.processing_results if r.plugin_id == plugin_id]

    def has_errors(self) -> bool:
        """检查是否有错误"""
        return any(not r.success for r in self.processing_results)

    def get_error_messages(self) -> List[str]:
        """获取错误消息"""
        return [r.error_message for r in self.processing_results if not r.success and r.error_message]

class SignalHookRegistry:
    """信号钩子注册表"""

    def __init__(self):
        # 钩子存储：hook_type -> [(priority, hook_function, plugin_id)]
        self.hooks: Dict[SignalHookType, List[tuple]] = {
            hook_type: [] for hook_type in SignalHookType
        }

        # 钩子统计
        self.hook_stats: Dict[str, Dict[str, int]] = {}

        # 钩子配置
        self.hook_timeout = 5.0  # 钩子执行超时时间
        self.max_hooks_per_type = 20  # 每种类型最大钩子数

    def register_hook(self, hook_type: SignalHookType, hook_function: Callable,
                     plugin_id: str, priority: HookPriority = HookPriority.NORMAL) -> bool:
        """注册钩子"""
        try:
            # 检查钩子数量限制
            if len(self.hooks[hook_type]) >= self.max_hooks_per_type:
                logger.error(f"钩子类型 {hook_type.value} 已达到最大数量限制")
                return False

            # 验证钩子函数
            if not callable(hook_function):
                logger.error(f"钩子函数必须是可调用的: {plugin_id}")
                return False

            # 添加钩子
            hook_entry = (priority.value, hook_function, plugin_id)
            self.hooks[hook_type].append(hook_entry)

            # 按优先级排序
            self.hooks[hook_type].sort(key=lambda x: x[0])

            # 初始化统计
            hook_key = f"{plugin_id}_{hook_type.value}"
            if hook_key not in self.hook_stats:
                self.hook_stats[hook_key] = {
                    'calls': 0,
                    'successes': 0,
                    'errors': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0
                }

            logger.debug(f"注册信号钩子: {hook_type.value} <- {plugin_id} (优先级: {priority.name})")
            return True

        except Exception as e:
            logger.error(f"注册钩子失败: {e}")
            return False

    def unregister_hook(self, hook_type: SignalHookType, plugin_id: str) -> int:
        """注销钩子"""
        removed_count = 0

        # 移除指定插件的钩子
        original_hooks = self.hooks[hook_type][:]
        self.hooks[hook_type] = [
            (priority, func, pid) for priority, func, pid in original_hooks
            if pid != plugin_id
        ]

        removed_count = len(original_hooks) - len(self.hooks[hook_type])

        if removed_count > 0:
            logger.debug(f"注销钩子: {hook_type.value} <- {plugin_id} (移除 {removed_count} 个)")

        return removed_count

    def unregister_plugin_hooks(self, plugin_id: str) -> int:
        """注销插件的所有钩子"""
        total_removed = 0

        for hook_type in SignalHookType:
            removed = self.unregister_hook(hook_type, plugin_id)
            total_removed += removed

        if total_removed > 0:
            logger.info(f"注销插件 {plugin_id} 的 {total_removed} 个信号钩子")

        return total_removed

    def get_hooks(self, hook_type: SignalHookType) -> List[tuple]:
        """获取指定类型的钩子"""
        return self.hooks[hook_type][:]

    def get_hook_count(self, hook_type: SignalHookType = None) -> Union[int, Dict[str, int]]:
        """获取钩子数量"""
        if hook_type:
            return len(self.hooks[hook_type])
        else:
            return {ht.value: len(hooks) for ht, hooks in self.hooks.items()}

    def get_hook_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取钩子统计"""
        return self.hook_stats.copy()

class SignalHookExecutor:
    """信号钩子执行器"""

    def __init__(self, registry: SignalHookRegistry):
        self.registry = registry
        self.execution_history = []
        self.max_history_size = 1000

    async def execute_hooks(self, hook_type: SignalHookType,
                          context: SignalHookContext) -> SignalHookContext:
        """执行钩子"""
        hooks = self.registry.get_hooks(hook_type)

        if not hooks:
            return context

        logger.debug(f"执行 {len(hooks)} 个 {hook_type.value} 钩子")

        for priority, hook_function, plugin_id in hooks:
            try:
                result = await self._execute_single_hook(
                    hook_function, plugin_id, context
                )

                context.add_result(result)

                # 如果钩子修改了数据，更新上下文
                if result.modified_data is not None:
                    context.current_signal = result.modified_data

                # 更新统计
                self._update_hook_stats(plugin_id, hook_type, result)

            except Exception as e:
                logger.error(f"钩子执行异常 {plugin_id}: {e}")

                error_result = HookResult(
                    hook_id=f"{plugin_id}_{hook_type.value}",
                    plugin_id=plugin_id,
                    success=False,
                    execution_time=0.0,
                    error_message=str(e)
                )

                context.add_result(error_result)
                self._update_hook_stats(plugin_id, hook_type, error_result)

        # 记录执行历史
        self._record_execution_history(hook_type, context)

        return context

    async def _execute_single_hook(self, hook_function: Callable,
                                 plugin_id: str, context: SignalHookContext) -> HookResult:
        """执行单个钩子"""
        import time
        start_time = time.time()

        hook_id = f"{plugin_id}_{context.hook_type.value}"

        try:
            # 准备钩子参数
            hook_args = {
                'context': context,
                'symbol': context.symbol,
                'signal': context.current_signal,
                'market_data': context.market_data,
                'metadata': context.metadata
            }

            # 执行钩子
            if asyncio.iscoroutinefunction(hook_function):
                result_data = await asyncio.wait_for(
                    hook_function(**hook_args),
                    timeout=self.registry.hook_timeout
                )
            else:
                result_data = hook_function(**hook_args)

            execution_time = time.time() - start_time

            # 处理返回结果
            modified_data = None
            if isinstance(result_data, dict):
                if 'modified_signal' in result_data:
                    modified_data = result_data['modified_signal']
                result_data = result_data.get('result', result_data)
            elif result_data is not None and result_data != context.current_signal:
                # 如果返回了不同的数据，视为修改
                modified_data = result_data

            return HookResult(
                hook_id=hook_id,
                plugin_id=plugin_id,
                success=True,
                execution_time=execution_time,
                result_data=result_data,
                modified_data=modified_data
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return HookResult(
                hook_id=hook_id,
                plugin_id=plugin_id,
                success=False,
                execution_time=execution_time,
                error_message=f"钩子执行超时 ({self.registry.hook_timeout}s)"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return HookResult(
                hook_id=hook_id,
                plugin_id=plugin_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )

    def _update_hook_stats(self, plugin_id: str, hook_type: SignalHookType,
                          result: HookResult):
        """更新钩子统计"""
        hook_key = f"{plugin_id}_{hook_type.value}"

        if hook_key not in self.registry.hook_stats:
            self.registry.hook_stats[hook_key] = {
                'calls': 0,
                'successes': 0,
                'errors': 0,
                'total_time': 0.0,
                'avg_time': 0.0
            }

        stats = self.registry.hook_stats[hook_key]
        stats['calls'] += 1
        stats['total_time'] += result.execution_time

        if result.success:
            stats['successes'] += 1
        else:
            stats['errors'] += 1

        stats['avg_time'] = stats['total_time'] / stats['calls']

    def _record_execution_history(self, hook_type: SignalHookType,
                                context: SignalHookContext):
        """记录执行历史"""
        history_entry = {
            'hook_type': hook_type.value,
            'symbol': context.symbol,
            'timestamp': context.timestamp,
            'hooks_executed': len(context.processing_results),
            'success_count': sum(1 for r in context.processing_results if r.success),
            'error_count': sum(1 for r in context.processing_results if not r.success),
            'total_time': sum(r.execution_time for r in context.processing_results),
            'has_modifications': any(r.modified_data is not None for r in context.processing_results)
        }

        self.execution_history.append(history_entry)

        # 限制历史记录大小
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

class SignalHookManager:
    """信号钩子管理器"""

    def __init__(self):
        self.registry = SignalHookRegistry()
        self.executor = SignalHookExecutor(self.registry)

        # 预定义钩子函数
        self._setup_builtin_hooks()

        logger.info("信号钩子管理器初始化完成")

    def _setup_builtin_hooks(self):
        """设置内置钩子"""
        # 信号验证钩子
        self.registry.register_hook(
            SignalHookType.SIGNAL_CREATED,
            self._validate_signal_hook,
            "builtin_validator",
            HookPriority.HIGHEST
        )

        # 信号日志钩子
        self.registry.register_hook(
            SignalHookType.SIGNAL_CREATED,
            self._log_signal_hook,
            "builtin_logger",
            HookPriority.LOWEST
        )

    def _validate_signal_hook(self, **kwargs) -> Dict[str, Any]:
        """内置信号验证钩子"""
        context = kwargs.get('context')
        signal = kwargs.get('signal')

        if not signal:
            raise ValueError("信号为空")

        # 基础验证
        required_fields = ['symbol', 'signal_type', 'confidence', 'current_price']
        for field in required_fields:
            if not hasattr(signal, field) or getattr(signal, field) is None:
                raise ValueError(f"信号缺少必需字段: {field}")

        # 置信度验证
        if not (0 <= signal.confidence <= 1):
            raise ValueError(f"信号置信度无效: {signal.confidence}")

        # 价格验证
        if signal.current_price <= 0:
            raise ValueError(f"信号价格无效: {signal.current_price}")

        return {"result": "validation_passed"}

    def _log_signal_hook(self, **kwargs) -> Dict[str, Any]:
        """内置信号日志钩子"""
        signal = kwargs.get('signal')

        if signal:
            logger.info(f"信号创建: {signal.symbol} {signal.signal_type} "
                       f"(置信度: {signal.confidence:.2f}, 价格: {signal.current_price})")

        return {"result": "logged"}

    # 公共接口

    def register_hook(self, hook_type: SignalHookType, hook_function: Callable,
                     plugin_id: str, priority: HookPriority = HookPriority.NORMAL) -> bool:
        """注册钩子"""
        return self.registry.register_hook(hook_type, hook_function, plugin_id, priority)

    def unregister_hook(self, hook_type: SignalHookType, plugin_id: str) -> int:
        """注销钩子"""
        return self.registry.unregister_hook(hook_type, plugin_id)

    def unregister_plugin_hooks(self, plugin_id: str) -> int:
        """注销插件的所有钩子"""
        return self.registry.unregister_plugin_hooks(plugin_id)

    async def trigger_hook(self, hook_type: SignalHookType, symbol: str,
                          signal: Any = None, market_data: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None) -> SignalHookContext:
        """触发钩子"""
        context = SignalHookContext(
            hook_type=hook_type,
            symbol=symbol,
            timestamp=datetime.now(),
            original_signal=signal,
            current_signal=signal,
            market_data=market_data or {},
            metadata=metadata or {}
        )

        return await self.executor.execute_hooks(hook_type, context)

    def get_hook_stats(self) -> Dict[str, Any]:
        """获取钩子统计"""
        return {
            'registry_stats': self.registry.get_hook_stats(),
            'hook_counts': self.registry.get_hook_count(),
            'execution_history_size': len(self.executor.execution_history),
            'recent_executions': self.executor.execution_history[-10:] if self.executor.execution_history else []
        }

# 装饰器支持

def signal_hook(hook_type: SignalHookType, priority: HookPriority = HookPriority.NORMAL):
    """信号钩子装饰器"""

    def decorator(func):
        # 标记函数为信号钩子
        func._signal_hook_type = hook_type
        func._signal_hook_priority = priority

        @functools.wraps(func)

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator

# 示例钩子函数
@signal_hook(SignalHookType.SIGNAL_CREATED, HookPriority.HIGH)

def example_signal_filter_hook(**kwargs):
    """示例信号过滤钩子"""
    signal = kwargs.get('signal')

    # 过滤低置信度信号
    if signal and signal.confidence < 0.5:
        logger.debug(f"过滤低置信度信号: {signal.symbol} (置信度: {signal.confidence})")
        return {"result": "filtered", "reason": "low_confidence"}

    return {"result": "passed"}

@signal_hook(SignalHookType.BEFORE_SIGNAL_EXECUTION, HookPriority.NORMAL)

async def example_market_condition_hook(**kwargs):
    """示例市场条件检查钩子"""
    signal = kwargs.get('signal')
    market_data = kwargs.get('market_data', {})

    # 检查市场状况
    market_volatility = market_data.get('volatility', 0)

    if market_volatility > 0.5:
        logger.warning(f"市场波动性过高: {market_volatility}, 建议暂缓交易")
        return {
            "result": "market_check_warning",
            "modified_signal": None  # 取消信号
        }

    return {"result": "market_check_passed"}

# 全局实例
_signal_hook_manager_instance = None

def get_signal_hook_manager() -> SignalHookManager:
    """获取信号钩子管理器实例"""
    global _signal_hook_manager_instance
    if _signal_hook_manager_instance is None:
        _signal_hook_manager_instance = SignalHookManager()
    return _signal_hook_manager_instance
