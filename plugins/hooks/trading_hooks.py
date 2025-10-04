"""
Trading Hooks
交易钩子系统，为交易执行过程提供事件驱动的扩展机制
支持订单处理、风险管理、仓位管理等各个阶段的钩子
"""

from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
import asyncio
import functools
from decimal import Decimal

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TradingHookType(Enum):
    """交易钩子类型"""
    # 订单生命周期
    BEFORE_ORDER_CREATION = "before_order_creation"
    ORDER_CREATED = "order_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_EXPIRED = "order_expired"

    # 交易执行
    BEFORE_TRADE_EXECUTION = "before_trade_execution"
    TRADE_EXECUTED = "trade_executed"
    TRADE_FAILED = "trade_failed"

    # 仓位管理
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_MODIFIED = "position_modified"
    POSITION_RISK_EXCEEDED = "position_risk_exceeded"

    # 风险管理
    BEFORE_RISK_CHECK = "before_risk_check"
    RISK_VIOLATION_DETECTED = "risk_violation_detected"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"

    # 账户管理
    BALANCE_UPDATED = "balance_updated"
    MARGIN_CALL = "margin_call"
    ACCOUNT_SUSPENDED = "account_suspended"

    # 市场事件
    MARKET_OPENED = "market_opened"
    MARKET_CLOSED = "market_closed"
    MARKET_HALTED = "market_halted"
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"

class TradingHookPriority(Enum):
    """交易钩子优先级"""
    CRITICAL = 1      # 关键系统钩子
    HIGH = 10         # 高优先级（风险管理等）
    NORMAL = 50       # 普通优先级
    LOW = 90          # 低优先级（通知、日志等）
    MONITORING = 100  # 监控和统计

@dataclass

class TradingHookResult:
    """交易钩子执行结果"""
    hook_id: str
    plugin_id: str
    success: bool
    execution_time: float

    # 结果数据
    result_data: Any = None
    modified_order: Any = None
    modified_position: Any = None

    # 控制指令
    should_block: bool = False  # 是否阻止后续执行
    should_retry: bool = False  # 是否需要重试
    retry_delay: float = 0.0    # 重试延迟（秒）

    # 错误信息
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'hook_id': self.hook_id,
            'plugin_id': self.plugin_id,
            'success': self.success,
            'execution_time': self.execution_time,
            'should_block': self.should_block,
            'should_retry': self.should_retry,
            'retry_delay': self.retry_delay,
            'error_message': self.error_message,
            'error_code': self.error_code,
            'has_modified_order': self.modified_order is not None,
            'has_modified_position': self.modified_position is not None
        }

@dataclass

class TradingHookContext:
    """交易钩子上下文"""
    hook_type: TradingHookType
    timestamp: datetime

    # 交易相关数据
    order: Any = None
    position: Any = None
    account: Any = None

    # 市场数据
    symbol: Optional[str] = None
    current_price: Optional[float] = None
    market_data: Dict[str, Any] = field(default_factory=dict)

    # 风险数据
    risk_metrics: Dict[str, Any] = field(default_factory=dict)

    # 执行结果
    processing_results: List[TradingHookResult] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: TradingHookResult):
        """添加钩子执行结果"""
        self.processing_results.append(result)

    def should_block_execution(self) -> bool:
        """检查是否应该阻止执行"""
        return any(r.should_block for r in self.processing_results if r.success)

    def get_retry_requests(self) -> List[TradingHookResult]:
        """获取重试请求"""
        return [r for r in self.processing_results if r.success and r.should_retry]

    def has_errors(self) -> bool:
        """检查是否有错误"""
        return any(not r.success for r in self.processing_results)

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        errors = [r for r in self.processing_results if not r.success]

        return {
            'error_count': len(errors),
            'error_messages': [r.error_message for r in errors if r.error_message],
            'error_codes': [r.error_code for r in errors if r.error_code],
            'failed_plugins': [r.plugin_id for r in errors]
        }

class TradingHookRegistry:
    """交易钩子注册表"""

    def __init__(self):
        # 钩子存储
        self.hooks: Dict[TradingHookType, List[tuple]] = {
            hook_type: [] for hook_type in TradingHookType
        }

        # 钩子配置
        self.hook_timeout = 10.0  # 交易钩子超时时间（比信号钩子长）
        self.max_hooks_per_type = 30
        self.critical_hooks = set()  # 关键钩子集合

        # 统计信息
        self.hook_stats: Dict[str, Dict[str, Any]] = {}

    def register_hook(self, hook_type: TradingHookType, hook_function: Callable,
                     plugin_id: str, priority: TradingHookPriority = TradingHookPriority.NORMAL,
                     is_critical: bool = False) -> bool:
        """注册交易钩子"""
        try:
            # 检查钩子数量限制
            if len(self.hooks[hook_type]) >= self.max_hooks_per_type:
                logger.error(f"交易钩子类型 {hook_type.value} 已达到最大数量限制")
                return False

            # 验证钩子函数
            if not callable(hook_function):
                logger.error(f"钩子函数必须是可调用的: {plugin_id}")
                return False

            # 添加钩子
            hook_entry = (priority.value, hook_function, plugin_id, is_critical)
            self.hooks[hook_type].append(hook_entry)

            # 按优先级排序
            self.hooks[hook_type].sort(key=lambda x: x[0])

            # 记录关键钩子
            if is_critical:
                hook_key = f"{plugin_id}_{hook_type.value}"
                self.critical_hooks.add(hook_key)

            # 初始化统计
            hook_key = f"{plugin_id}_{hook_type.value}"
            if hook_key not in self.hook_stats:
                self.hook_stats[hook_key] = {
                    'calls': 0,
                    'successes': 0,
                    'errors': 0,
                    'blocks': 0,
                    'retries': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'is_critical': is_critical
                }

            logger.debug(f"注册交易钩子: {hook_type.value} <- {plugin_id} "
                        f"(优先级: {priority.name}, 关键: {is_critical})")
            return True

        except Exception as e:
            logger.error(f"注册交易钩子失败: {e}")
            return False

    def unregister_hook(self, hook_type: TradingHookType, plugin_id: str) -> int:
        """注销交易钩子"""
        removed_count = 0

        # 移除指定插件的钩子
        original_hooks = self.hooks[hook_type][:]
        self.hooks[hook_type] = [
            (priority, func, pid, critical) for priority, func, pid, critical in original_hooks
            if pid != plugin_id
        ]

        removed_count = len(original_hooks) - len(self.hooks[hook_type])

        # 移除关键钩子标记
        hook_key = f"{plugin_id}_{hook_type.value}"
        self.critical_hooks.discard(hook_key)

        if removed_count > 0:
            logger.debug(f"注销交易钩子: {hook_type.value} <- {plugin_id} (移除 {removed_count} 个)")

        return removed_count

    def unregister_plugin_hooks(self, plugin_id: str) -> int:
        """注销插件的所有交易钩子"""
        total_removed = 0

        for hook_type in TradingHookType:
            removed = self.unregister_hook(hook_type, plugin_id)
            total_removed += removed

        if total_removed > 0:
            logger.info(f"注销插件 {plugin_id} 的 {total_removed} 个交易钩子")

        return total_removed

    def get_hooks(self, hook_type: TradingHookType) -> List[tuple]:
        """获取指定类型的钩子"""
        return self.hooks[hook_type][:]

    def is_critical_hook(self, plugin_id: str, hook_type: TradingHookType) -> bool:
        """检查是否为关键钩子"""
        hook_key = f"{plugin_id}_{hook_type.value}"
        return hook_key in self.critical_hooks

class TradingHookExecutor:
    """交易钩子执行器"""

    def __init__(self, registry: TradingHookRegistry):
        self.registry = registry
        self.execution_history = []
        self.max_history_size = 1000
        self.failed_critical_hooks = set()  # 失败的关键钩子

    async def execute_hooks(self, hook_type: TradingHookType,
                          context: TradingHookContext) -> TradingHookContext:
        """执行交易钩子"""
        hooks = self.registry.get_hooks(hook_type)

        if not hooks:
            return context

        logger.debug(f"执行 {len(hooks)} 个 {hook_type.value} 交易钩子")

        # 分离关键钩子和普通钩子
        critical_hooks = []
        normal_hooks = []

        for priority, hook_function, plugin_id, is_critical in hooks:
            if is_critical:
                critical_hooks.append((priority, hook_function, plugin_id, is_critical))
            else:
                normal_hooks.append((priority, hook_function, plugin_id, is_critical))

        # 先执行关键钩子
        for priority, hook_function, plugin_id, is_critical in critical_hooks:
            result = await self._execute_single_hook(
                hook_function, plugin_id, context, is_critical
            )

            context.add_result(result)
            self._update_hook_stats(plugin_id, hook_type, result)

            # 关键钩子失败时的处理
            if not result.success:
                hook_key = f"{plugin_id}_{hook_type.value}"
                self.failed_critical_hooks.add(hook_key)

                logger.critical(f"关键交易钩子失败: {plugin_id} - {result.error_message}")

                # 如果关键钩子失败，可能需要中止整个流程
                if hook_type in [TradingHookType.BEFORE_RISK_CHECK,
                               TradingHookType.BEFORE_ORDER_CREATION]:
                    context.metadata['critical_hook_failed'] = True
                    return context

        # 再执行普通钩子
        for priority, hook_function, plugin_id, is_critical in normal_hooks:
            # 如果已经被标记为阻止，跳过低优先级的钩子
            if context.should_block_execution() and priority > TradingHookPriority.HIGH.value:
                continue

            result = await self._execute_single_hook(
                hook_function, plugin_id, context, is_critical
            )

            context.add_result(result)
            self._update_hook_stats(plugin_id, hook_type, result)

        # 记录执行历史
        self._record_execution_history(hook_type, context)

        return context

    async def _execute_single_hook(self, hook_function: Callable, plugin_id: str,
                                 context: TradingHookContext, is_critical: bool = False) -> TradingHookResult:
        """执行单个交易钩子"""
        import time
        start_time = time.time()

        hook_id = f"{plugin_id}_{context.hook_type.value}"

        try:
            # 准备钩子参数
            hook_args = {
                'context': context,
                'order': context.order,
                'position': context.position,
                'account': context.account,
                'symbol': context.symbol,
                'current_price': context.current_price,
                'market_data': context.market_data,
                'risk_metrics': context.risk_metrics,
                'metadata': context.metadata
            }

            # 执行钩子
            timeout = self.registry.hook_timeout * (2 if is_critical else 1)  # 关键钩子给更多时间

            if asyncio.iscoroutinefunction(hook_function):
                result_data = await asyncio.wait_for(
                    hook_function(**hook_args),
                    timeout=timeout
                )
            else:
                result_data = hook_function(**hook_args)

            execution_time = time.time() - start_time

            # 处理返回结果
            result = TradingHookResult(
                hook_id=hook_id,
                plugin_id=plugin_id,
                success=True,
                execution_time=execution_time
            )

            # 解析结果数据
            if isinstance(result_data, dict):
                result.result_data = result_data.get('result')
                result.modified_order = result_data.get('modified_order')
                result.modified_position = result_data.get('modified_position')
                result.should_block = result_data.get('should_block', False)
                result.should_retry = result_data.get('should_retry', False)
                result.retry_delay = result_data.get('retry_delay', 0.0)
            else:
                result.result_data = result_data

            # 应用修改
            if result.modified_order:
                context.order = result.modified_order
            if result.modified_position:
                context.position = result.modified_position

            return result

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return TradingHookResult(
                hook_id=hook_id,
                plugin_id=plugin_id,
                success=False,
                execution_time=execution_time,
                error_message=f"交易钩子执行超时 ({timeout}s)",
                error_code="HOOK_TIMEOUT"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TradingHookResult(
                hook_id=hook_id,
                plugin_id=plugin_id,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                error_code="HOOK_EXECUTION_ERROR"
            )

    def _update_hook_stats(self, plugin_id: str, hook_type: TradingHookType,
                          result: TradingHookResult):
        """更新钩子统计"""
        hook_key = f"{plugin_id}_{hook_type.value}"

        if hook_key not in self.registry.hook_stats:
            return

        stats = self.registry.hook_stats[hook_key]
        stats['calls'] += 1
        stats['total_time'] += result.execution_time

        if result.success:
            stats['successes'] += 1
            if result.should_block:
                stats['blocks'] += 1
            if result.should_retry:
                stats['retries'] += 1
        else:
            stats['errors'] += 1

        stats['avg_time'] = stats['total_time'] / stats['calls']

    def _record_execution_history(self, hook_type: TradingHookType,
                                context: TradingHookContext):
        """记录执行历史"""
        history_entry = {
            'hook_type': hook_type.value,
            'symbol': context.symbol,
            'timestamp': context.timestamp.isoformat(),
            'hooks_executed': len(context.processing_results),
            'success_count': sum(1 for r in context.processing_results if r.success),
            'error_count': sum(1 for r in context.processing_results if not r.success),
            'block_count': sum(1 for r in context.processing_results if r.should_block),
            'retry_count': sum(1 for r in context.processing_results if r.should_retry),
            'total_time': sum(r.execution_time for r in context.processing_results),
            'critical_hook_failed': context.metadata.get('critical_hook_failed', False)
        }

        self.execution_history.append(history_entry)

        # 限制历史记录大小
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size:]

    def get_failed_critical_hooks(self) -> set:
        """获取失败的关键钩子"""
        return self.failed_critical_hooks.copy()

    def clear_failed_critical_hooks(self):
        """清除失败的关键钩子记录"""
        self.failed_critical_hooks.clear()

class TradingHookManager:
    """交易钩子管理器"""

    def __init__(self):
        self.registry = TradingHookRegistry()
        self.executor = TradingHookExecutor(self.registry)

        # 预定义钩子
        self._setup_builtin_hooks()

        logger.info("交易钩子管理器初始化完成")

    def _setup_builtin_hooks(self):
        """设置内置钩子"""
        # 风险检查钩子
        self.registry.register_hook(
            TradingHookType.BEFORE_ORDER_CREATION,
            self._risk_check_hook,
            "builtin_risk_checker",
            TradingHookPriority.CRITICAL,
            is_critical=True
        )

        # 订单验证钩子
        self.registry.register_hook(
            TradingHookType.ORDER_CREATED,
            self._validate_order_hook,
            "builtin_order_validator",
            TradingHookPriority.HIGH,
            is_critical=True
        )

        # 交易日志钩子
        self.registry.register_hook(
            TradingHookType.TRADE_EXECUTED,
            self._log_trade_hook,
            "builtin_trade_logger",
            TradingHookPriority.MONITORING
        )

    def _risk_check_hook(self, **kwargs) -> Dict[str, Any]:
        """内置风险检查钩子"""
        context = kwargs.get('context')
        order = kwargs.get('order')
        account = kwargs.get('account')
        risk_metrics = kwargs.get('risk_metrics', {})

        # 基础风险检查
        if account and hasattr(account, 'available_balance'):
            if order and hasattr(order, 'value'):
                if order.value > account.available_balance:
                    return {
                        'result': 'risk_violation',
                        'should_block': True,
                        'error': 'insufficient_balance'
                    }

        # 检查仓位风险
        max_position_risk = risk_metrics.get('max_position_risk', 0.1)
        current_risk = risk_metrics.get('current_position_risk', 0.0)

        if current_risk > max_position_risk:
            return {
                'result': 'risk_violation',
                'should_block': True,
                'error': 'position_risk_exceeded'
            }

        return {'result': 'risk_check_passed'}

    def _validate_order_hook(self, **kwargs) -> Dict[str, Any]:
        """内置订单验证钩子"""
        order = kwargs.get('order')

        if not order:
            return {
                'result': 'validation_failed',
                'should_block': True,
                'error': 'order_is_null'
            }

        # 基础订单验证
        required_fields = ['symbol', 'quantity', 'order_type']
        for field in required_fields:
            if not hasattr(order, field) or getattr(order, field) is None:
                return {
                    'result': 'validation_failed',
                    'should_block': True,
                    'error': f'missing_field_{field}'
                }

        # 数量验证
        if hasattr(order, 'quantity') and order.quantity <= 0:
            return {
                'result': 'validation_failed',
                'should_block': True,
                'error': 'invalid_quantity'
            }

        return {'result': 'validation_passed'}

    def _log_trade_hook(self, **kwargs) -> Dict[str, Any]:
        """内置交易日志钩子"""
        context = kwargs.get('context')

        if context and context.order:
            logger.info(f"交易执行: {context.symbol} - "
                       f"类型: {getattr(context.order, 'order_type', 'unknown')} "
                       f"数量: {getattr(context.order, 'quantity', 'unknown')}")

        return {'result': 'logged'}

    # 公共接口

    def register_hook(self, hook_type: TradingHookType, hook_function: Callable,
                     plugin_id: str, priority: TradingHookPriority = TradingHookPriority.NORMAL,
                     is_critical: bool = False) -> bool:
        """注册交易钩子"""
        return self.registry.register_hook(hook_type, hook_function, plugin_id, priority, is_critical)

    def unregister_hook(self, hook_type: TradingHookType, plugin_id: str) -> int:
        """注销交易钩子"""
        return self.registry.unregister_hook(hook_type, plugin_id)

    def unregister_plugin_hooks(self, plugin_id: str) -> int:
        """注销插件的所有交易钩子"""
        return self.registry.unregister_plugin_hooks(plugin_id)

    async def trigger_hook(self, hook_type: TradingHookType,
                          order: Any = None, position: Any = None, account: Any = None,
                          symbol: str = None, current_price: float = None,
                          market_data: Dict[str, Any] = None,
                          risk_metrics: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None) -> TradingHookContext:
        """触发交易钩子"""
        context = TradingHookContext(
            hook_type=hook_type,
            timestamp=datetime.now(),
            order=order,
            position=position,
            account=account,
            symbol=symbol,
            current_price=current_price,
            market_data=market_data or {},
            risk_metrics=risk_metrics or {},
            metadata=metadata or {}
        )

        return await self.executor.execute_hooks(hook_type, context)

    def get_hook_stats(self) -> Dict[str, Any]:
        """获取钩子统计"""
        return {
            'registry_stats': self.registry.hook_stats,
            'hook_counts': {ht.value: len(hooks) for ht, hooks in self.registry.hooks.items()},
            'critical_hooks_count': len(self.registry.critical_hooks),
            'failed_critical_hooks': list(self.executor.failed_critical_hooks),
            'execution_history_size': len(self.executor.execution_history),
            'recent_executions': self.executor.execution_history[-5:]
        }

# 装饰器支持

def trading_hook(hook_type: TradingHookType,
                priority: TradingHookPriority = TradingHookPriority.NORMAL,
                is_critical: bool = False):
    """交易钩子装饰器"""

    def decorator(func):
        func._trading_hook_type = hook_type
        func._trading_hook_priority = priority
        func._trading_hook_critical = is_critical

        @functools.wraps(func)

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator

# 示例钩子函数
@trading_hook(TradingHookType.BEFORE_ORDER_CREATION,
              TradingHookPriority.HIGH, is_critical=True)

def example_position_size_check_hook(**kwargs):
    """示例仓位大小检查钩子"""
    order = kwargs.get('order')
    account = kwargs.get('account')

    if not order or not account:
        return {'result': 'missing_data'}

    # 检查单笔订单不超过账户的10%
    if hasattr(order, 'value') and hasattr(account, 'total_value'):
        max_order_size = account.total_value * 0.1

        if order.value > max_order_size:
            logger.warning(f"订单金额 {order.value} 超过账户10%限制 {max_order_size}")
            return {
                'result': 'position_size_violation',
                'should_block': True,
                'modified_order': None  # 可以修改订单或设为None来拒绝
            }

    return {'result': 'position_size_check_passed'}

@trading_hook(TradingHookType.TRADE_EXECUTED, TradingHookPriority.MONITORING)

async def example_trade_notification_hook(**kwargs):
    """示例交易通知钩子"""
    context = kwargs.get('context')

    if context and context.order:
        # 这里可以发送通知、更新数据库等
        logger.info(f"发送交易通知: {context.symbol} 交易完成")

        # 模拟异步通知操作
        await asyncio.sleep(0.1)

    return {'result': 'notification_sent'}

# 全局实例
_trading_hook_manager_instance = None

def get_trading_hook_manager() -> TradingHookManager:
    """获取交易钩子管理器实例"""
    global _trading_hook_manager_instance
    if _trading_hook_manager_instance is None:
        _trading_hook_manager_instance = TradingHookManager()
    return _trading_hook_manager_instance
