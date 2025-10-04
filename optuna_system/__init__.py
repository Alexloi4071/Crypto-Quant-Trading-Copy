"""
Optuna System - 9层量化交易优化系统

这个包提供了一个完整的9层优化架构，用于加密货币量化交易策略的参数优化。

主要组件:
- coordinator: 系统协调器，管理所有优化层
- optimizers: 各层优化器实现
- version_manager: 版本管理和结果追踪

使用示例:
    from optuna_system.coordinator import OptunaCoordinator
    
    coordinator = OptunaCoordinator(
        data_path="data", 
        config_path="config",
        symbol="BTCUSDT",
        timeframe="15m"
    )
    
    # 运行完整的9层优化
    result = coordinator.quick_complete_optimization()
"""

__version__ = "1.0.0"
__author__ = "Crypto Quant Trading System"

# 主要导出（延迟加载，避免在导入子模块时触发重型依赖）
__all__ = ['OptunaCoordinator', 'OptunaVersionManager']

def __getattr__(name):
    if name == 'OptunaCoordinator':
        from .coordinator import OptunaCoordinator  # type: ignore
        return OptunaCoordinator
    if name == 'OptunaVersionManager':
        from .version_manager import OptunaVersionManager  # type: ignore
        return OptunaVersionManager
    raise AttributeError(name)
