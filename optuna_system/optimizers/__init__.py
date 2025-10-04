"""
Optimizers - 9层优化器模块

各层优化器说明:
- Layer0: 数据清洗优化 (optuna_cleaning)
- Layer1: 标签生成优化 (optuna_label)  
- Layer2: 特征工程优化 (optuna_feature)
- Layer3: 模型参数优化 (optuna_model)
- Layer4: 交叉验证与风控优化 (optuna_cv_risk)
- Layer5: Kelly资金管理优化 (kelly_optimizer)
- Layer6: 模型集成优化 (ensemble_optimizer)
- Layer7: 多项式特征优化 (polynomial_optimizer)
- Layer8: 置信度评分优化 (confidence_optimizer)

使用示例:
    from optuna_system.optimizers.optuna_feature import FeatureOptimizer
    
    optimizer = FeatureOptimizer(
        data_path="data",
        config_path="config", 
        symbol="BTCUSDT",
        timeframe="15m"
    )
    
    result = optimizer.optimize(n_trials=100)
"""

# 核心Layer优化器 (Layer0-4)
try:
    from .optuna_cleaning import DataCleaningOptimizer
except ImportError:
    DataCleaningOptimizer = None

try:
    from .optuna_label import LabelOptimizer
except ImportError:
    LabelOptimizer = None

try:
    from .optuna_feature import FeatureOptimizer
except ImportError:
    FeatureOptimizer = None

try:
    from .optuna_model import ModelOptimizer
except ImportError:
    ModelOptimizer = None

try:
    from .optuna_cv_risk import CVRiskOptimizer
except ImportError:
    CVRiskOptimizer = None

# 高级优化器 (Layer5-8)
try:
    from .kelly_optimizer import KellyOptimizer
except ImportError:
    KellyOptimizer = None

try:
    from .ensemble_optimizer import EnsembleOptimizer
except ImportError:
    EnsembleOptimizer = None

try:
    from .polynomial_optimizer import PolynomialOptimizer
except ImportError:
    PolynomialOptimizer = None

try:
    from .confidence_optimizer import ConfidenceOptimizer
except ImportError:
    ConfidenceOptimizer = None

# 导出所有可用的优化器
__all__ = [
    # 核心层优化器
    'DataCleaningOptimizer',  # Layer0
    'LabelOptimizer',         # Layer1
    'FeatureOptimizer',       # Layer2
    'ModelOptimizer',         # Layer3
    'CVRiskOptimizer',        # Layer4
    
    # 高级优化器
    'KellyOptimizer',         # Layer5
    'EnsembleOptimizer',      # Layer6
    'PolynomialOptimizer',    # Layer7
    'ConfidenceOptimizer',    # Layer8
]

# 过滤掉无法导入的模块
__all__ = [name for name in __all__ if globals()[name] is not None]
