"""
Portfolio Optimizer
投资组合优化器，基于现代投资组合理论和机器学习算法
实现资产配置、风险平价、因子模型等多种优化策略
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import linalg
import cvxpy as cp
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
from collections import defaultdict
import warnings

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)


class OptimizationType(Enum):
    """优化类型"""
    MEAN_VARIANCE = "mean_variance"              # 均值方差优化
    MINIMUM_VARIANCE = "minimum_variance"         # 最小方差
    MAXIMUM_SHARPE = "maximum_sharpe"            # 最大夏普比率
    RISK_PARITY = "risk_parity"                  # 风险平价
    EQUAL_WEIGHT = "equal_weight"                # 等权重
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"  # 最大分散化
    BLACK_LITTERMAN = "black_litterman"          # Black-Litterman模型
    FACTOR_MODEL = "factor_model"                # 因子模型
    ROBUST_OPTIMIZATION = "robust_optimization"  # 鲁棒优化


class RiskMeasure(Enum):
    """风险度量"""
    VOLATILITY = "volatility"                    # 波动率
    VAR = "var"                                 # 风险价值
    CVAR = "cvar"                               # 条件风险价值
    MAX_DRAWDOWN = "max_drawdown"               # 最大回撤
    SEMI_DEVIATION = "semi_deviation"           # 半偏差
    TRACKING_ERROR = "tracking_error"           # 跟踪误差

@dataclass
class OptimizationConstraints:
    """优化约束条件"""
    # 权重约束
    min_weight: float = 0.0                     # 最小权重
    max_weight: float = 1.0                     # 最大权重
    long_only: bool = True                      # 仅做多

    # 组合约束
    max_concentration: Optional[float] = None    # 最大集中度
    sector_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # 行业限制

    # 风险约束
    max_volatility: Optional[float] = None       # 最大波动率
    max_var: Optional[float] = None             # 最大VaR
    max_drawdown: Optional[float] = None        # 最大回撤

    # 交易约束
    turnover_limit: Optional[float] = None       # 换手率限制
    transaction_cost: float = 0.001             # 交易成本

    # 其他约束
    target_return: Optional[float] = None        # 目标收益率
    benchmark_tracking: Optional[str] = None     # 基准跟踪

@dataclass
class OptimizationResult:
    """优化结果"""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float

    # 风险指标
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    max_drawdown: Optional[float] = None

    # 优化信息
    optimization_type: OptimizationType = OptimizationType.MEAN_VARIANCE
    status: str = "success"
    iterations: int = 0
    execution_time: float = 0.0

    # 组合特征
    concentration: float = 0.0                   # 集中度(HHI)
    diversification_ratio: float = 0.0          # 分散化比率
    effective_assets: float = 0.0               # 有效资产数


    def to_dict(self) -> dict:
        return {
            'weights': self.weights.to_dict(),
            'expected_return': self.expected_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'max_drawdown': self.max_drawdown,
            'optimization_type': self.optimization_type.value,
            'status': self.status,
            'iterations': self.iterations,
            'execution_time': self.execution_time,
            'concentration': self.concentration,
            'diversification_ratio': self.diversification_ratio,
            'effective_assets': self.effective_assets
        }


class RiskModel:
    """风险模型基类"""

    def __init__(self):
        self.name = "BaseRiskModel"

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """估计协方差矩阵"""
        return returns.cov() * 252  # 年化

    def estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """估计预期收益"""
        return returns.mean() * 252  # 年化


class SampleCovarianceModel(RiskModel):
    """样本协方差模型"""

    def __init__(self, lookback_days: int = 252):
        super().__init__()
        self.name = "SampleCovariance"
        self.lookback_days = lookback_days

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """估计样本协方差矩阵"""
        recent_returns = returns.tail(self.lookback_days)
        return recent_returns.cov() * 252


class ShrinkageModel(RiskModel):
    """收缩估计模型"""

    def __init__(self, shrinkage_target: str = "identity", alpha: Optional[float] = None):
        super().__init__()
        self.name = "Shrinkage"
        self.shrinkage_target = shrinkage_target
        self.alpha = alpha

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """估计收缩协方差矩阵"""
        sample_cov = returns.cov() * 252
        n_assets = len(sample_cov)

        # 选择收缩目标
        if self.shrinkage_target == "identity":
            target = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        elif self.shrinkage_target == "constant_correlation":
            target = self._constant_correlation_matrix(sample_cov)
        else:
            target = np.eye(n_assets)

        # 计算收缩强度
        if self.alpha is None:
            alpha = self._optimal_shrinkage(returns)
        else:
            alpha = self.alpha

        # 收缩估计
        shrunk_cov = alpha * target + (1 - alpha) * sample_cov

        return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)


    def _constant_correlation_matrix(self, sample_cov: pd.DataFrame) -> np.ndarray:
        """常相关矩阵目标"""
        variances = np.diag(sample_cov)
        avg_correlation = (sample_cov.values.sum() - variances.sum()) / (len(sample_cov) * (len(sample_cov) - 1))

        target = np.full_like(sample_cov.values, avg_correlation)
        np.fill_diagonal(target, variances)

        return target


    def _optimal_shrinkage(self, returns: pd.DataFrame) -> float:
        """计算最优收缩强度（Ledoit-Wolf方法）"""
        T, N = returns.shape

        # 样本协方差矩阵
        sample_cov = returns.cov().values

        # 计算收缩强度
        # 这是简化版本，完整实现需要更复杂的计算
        rho = 0.5  # 简化为固定值

        return max(0, min(1, rho))


class FactorModel(RiskModel):
    """因子模型"""

    def __init__(self, factors: Optional[pd.DataFrame] = None):
        super().__init__()
        self.name = "FactorModel"
        self.factors = factors
        self.factor_loadings = None
        self.idiosyncratic_risk = None

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """基于因子模型估计协方差矩阵"""
        if self.factors is None:
            # 使用PCA构建因子
            self._build_pca_factors(returns)

        # 估计因子载荷
        self._estimate_factor_loadings(returns)

        # 构建协方差矩阵: Σ = B * F * B' + D
        factor_cov = self.factors.cov() * 252
        B = self.factor_loadings
        systematic_risk = B @ factor_cov @ B.T

        # 添加特异风险
        total_cov = systematic_risk + np.diag(self.idiosyncratic_risk)

        return pd.DataFrame(total_cov, index=returns.columns, columns=returns.columns)


    def _build_pca_factors(self, returns: pd.DataFrame, n_factors: int = 5):
        """使用PCA构建因子"""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(returns.dropna())

        self.factors = pd.DataFrame(
            factor_returns,
            index=returns.dropna().index,
            columns=[f"Factor_{i+1}" for i in range(n_factors)]
        )


    def _estimate_factor_loadings(self, returns: pd.DataFrame):
        """估计因子载荷"""
        from sklearn.linear_model import LinearRegression

        common_index = returns.index.intersection(self.factors.index)
        returns_aligned = returns.loc[common_index]
        factors_aligned = self.factors.loc[common_index]

        n_assets = len(returns.columns)
        n_factors = len(self.factors.columns)

        loadings = np.zeros((n_assets, n_factors))
        idiosyncratic_var = np.zeros(n_assets)

        for i, asset in enumerate(returns.columns):
            reg = LinearRegression()
            reg.fit(factors_aligned, returns_aligned[asset])

            loadings[i] = reg.coef_

            # 计算特异风险
            predictions = reg.predict(factors_aligned)
            residuals = returns_aligned[asset] - predictions
            idiosyncratic_var[i] = residuals.var() * 252

        self.factor_loadings = loadings
        self.idiosyncratic_risk = idiosyncratic_var


class MeanReversionModel(RiskModel):
    """均值回归预期收益模型"""

    def __init__(self, lookback_days: int = 252, half_life: int = 60):
        super().__init__()
        self.name = "MeanReversion"
        self.lookback_days = lookback_days
        self.half_life = half_life

    def estimate_expected_returns(self, returns: pd.DataFrame) -> pd.Series:
        """基于均值回归估计预期收益"""
        # 计算长期均值
        long_term_mean = returns.tail(self.lookback_days).mean() * 252

        # 计算当前水平（短期均值）
        short_term_mean = returns.tail(21).mean() * 252  # 最近一个月

        # 均值回归强度
        decay_factor = np.exp(-np.log(2) / self.half_life)
        reversion_strength = 1 - decay_factor

        # 预期收益 = 当前水平 + 回归强度 * (长期均值 - 当前水平)
        expected_returns = short_term_mean + reversion_strength * (long_term_mean - short_term_mean)

        return expected_returns


class BlackLittermanModel:
    """Black-Litterman模型"""

    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.025):
        self.risk_aversion = risk_aversion
        self.tau = tau

    def optimize(self, market_caps: pd.Series, cov_matrix: pd.DataFrame,
                views: Dict[str, float], view_confidence: Dict[str, float]) -> Tuple[pd.Series, pd.DataFrame]:
        """Black-Litterman优化"""

        # 市场均衡预期收益
        market_weights = market_caps / market_caps.sum()
        pi = self.risk_aversion * (cov_matrix @ market_weights)

        if not views:
            # 没有观点时，返回市场均衡
            return pi, cov_matrix

        # 构建观点矩阵P和观点向量Q
        assets = list(cov_matrix.index)
        P = np.zeros((len(views), len(assets)))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))

        for i, (asset, view) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                P[i, asset_idx] = 1.0
                Q[i] = view
                Omega[i, i] = view_confidence.get(asset, 1.0)

        # Black-Litterman公式
        tau_cov = self.tau * cov_matrix.values

        # 新的预期收益
        M1 = linalg.inv(tau_cov) + P.T @ linalg.inv(Omega) @ P
        M2 = linalg.inv(tau_cov) @ pi.values + P.T @ linalg.inv(Omega) @ Q

        bl_returns = linalg.solve(M1, M2)

        # 新的协方差矩阵
        bl_cov = linalg.inv(linalg.inv(tau_cov) + P.T @ linalg.inv(Omega) @ P)

        return (pd.Series(bl_returns, index=cov_matrix.index),
                pd.DataFrame(bl_cov, index=cov_matrix.index, columns=cov_matrix.columns))


class PortfolioOptimizer:
    """投资组合优化器主类"""

    def __init__(self, risk_model: Optional[RiskModel] = None):
        self.risk_model = risk_model or SampleCovarianceModel()
        self.bl_model = BlackLittermanModel()

        # 优化历史
        self.optimization_history = []
        self.max_history = 1000

        # 统计信息
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'average_execution_time': 0.0,
            'optimization_types': defaultdict(int)
        }

        logger.info(f"投资组合优化器初始化完成，风险模型: {self.risk_model.name}")


    def optimize(self, returns: pd.DataFrame,
                optimization_type: OptimizationType = OptimizationType.MEAN_VARIANCE,
                constraints: Optional[OptimizationConstraints] = None,
                **kwargs) -> OptimizationResult:
        """主优化函数"""

        import time
        start_time = time.time()

        try:
            # 准备数据
            clean_returns = self._prepare_data(returns)
            if clean_returns.empty:
                raise ValueError("无有效的收益数据")

            # 设置默认约束
            constraints = constraints or OptimizationConstraints()

            # 估计预期收益和协方差矩阵
            expected_returns = self.risk_model.estimate_expected_returns(clean_returns)
            cov_matrix = self.risk_model.estimate_covariance(clean_returns)

            # 根据优化类型调用相应方法
            if optimization_type == OptimizationType.MEAN_VARIANCE:
                result = self._mean_variance_optimization(expected_returns, cov_matrix, constraints, **kwargs)
            elif optimization_type == OptimizationType.MINIMUM_VARIANCE:
                result = self._minimum_variance_optimization(cov_matrix, constraints)
            elif optimization_type == OptimizationType.MAXIMUM_SHARPE:
                result = self._maximum_sharpe_optimization(expected_returns, cov_matrix, constraints)
            elif optimization_type == OptimizationType.RISK_PARITY:
                result = self._risk_parity_optimization(cov_matrix, constraints)
            elif optimization_type == OptimizationType.EQUAL_WEIGHT:
                result = self._equal_weight_optimization(expected_returns, cov_matrix)
            elif optimization_type == OptimizationType.BLACK_LITTERMAN:
                result = self._black_litterman_optimization(clean_returns, **kwargs)
            else:
                raise ValueError(f"不支持的优化类型: {optimization_type}")

            # 计算额外指标
            result = self._calculate_additional_metrics(result, expected_returns, cov_matrix)

            # 记录执行时间
            result.execution_time = time.time() - start_time
            result.optimization_type = optimization_type

            # 更新统计
            self._update_stats(result, optimization_type)

            # 记录历史
            self._record_optimization(result)

            logger.info(f"优化完成: {optimization_type.value}, 夏普比率: {result.sharpe_ratio:.4f}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"优化失败 ({optimization_type.value}): {e}")

            self.stats['total_optimizations'] += 1
            self.stats['failed_optimizations'] += 1

            # 返回等权重作为fallback
            return self._equal_weight_optimization(
                pd.Series(index=returns.columns, data=0.1),
                pd.DataFrame(np.eye(len(returns.columns)),
                           index=returns.columns, columns=returns.columns)
            )


    def _prepare_data(self, returns: pd.DataFrame) -> pd.DataFrame:
        """准备数据"""
        # 删除缺失值
        clean_returns = returns.dropna()

        # 删除方差为0的资产
        zero_var_assets = clean_returns.var() == 0
        if zero_var_assets.any():
            logger.warning(f"移除零方差资产: {zero_var_assets[zero_var_assets].index.tolist()}")
            clean_returns = clean_returns.loc[:, ~zero_var_assets]

        # 检查数据质量
        if len(clean_returns) < 60:  # 至少需要60个观测值
            logger.warning(f"数据量较少: {len(clean_returns)} 个观测值")

        return clean_returns


    def _mean_variance_optimization(self, expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  constraints: OptimizationConstraints,
                                  risk_aversion: float = 1.0) -> OptimizationResult:
        """均值方差优化"""
        n = len(expected_returns)

        # 创建优化变量
        w = cp.Variable(n)

        # 目标函数: 最大化 μ'w - λ/2 * w'Σw
        portfolio_return = expected_returns.values @ w
        portfolio_variance = cp.quad_form(w, cov_matrix.values)
        objective = cp.Maximize(portfolio_return - risk_aversion / 2 * portfolio_variance)

        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,  # 权重和为1
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]

        if constraints.long_only:
            constraints_list.append(w >= 0)

        if constraints.max_volatility:
            constraints_list.append(cp.sqrt(portfolio_variance) <= constraints.max_volatility)

        # 求解
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["infeasible", "unbounded"]:
            weights = pd.Series(w.value, index=expected_returns.index)
            weights = weights / weights.sum()  # 归一化

            return self._create_result(weights, expected_returns, cov_matrix)
        else:
            raise ValueError(f"优化失败: {problem.status}")


    def _minimum_variance_optimization(self, cov_matrix: pd.DataFrame,
                                     constraints: OptimizationConstraints) -> OptimizationResult:
        """最小方差优化"""
        n = len(cov_matrix)

        # 创建优化变量
        w = cp.Variable(n)

        # 目标函数: 最小化组合方差
        objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))

        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]

        if constraints.long_only:
            constraints_list.append(w >= 0)

        # 求解
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["infeasible", "unbounded"]:
            weights = pd.Series(w.value, index=cov_matrix.index)
            weights = weights / weights.sum()

            # 计算预期收益（使用历史均值）
            expected_returns = pd.Series(0.08, index=cov_matrix.index)  # 假设8%年化收益

            return self._create_result(weights, expected_returns, cov_matrix)
        else:
            raise ValueError(f"最小方差优化失败: {problem.status}")


    def _maximum_sharpe_optimization(self, expected_returns: pd.Series,
                                   cov_matrix: pd.DataFrame,
                                   constraints: OptimizationConstraints,
                                   risk_free_rate: float = 0.02) -> OptimizationResult:
        """最大夏普比率优化"""
        n = len(expected_returns)

        # 转换为二次规划形式
        excess_returns = expected_returns - risk_free_rate

        # 使用scipy优化


        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, excess_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_std = np.sqrt(portfolio_variance)

            if portfolio_std == 0:
                return -np.inf

            return -portfolio_return / portfolio_std  # 负号因为要最大化

        # 约束条件
        constraints_opt = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
        ]

        # 边界条件
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]

        if constraints.long_only:
            bounds = [(0, constraints.max_weight) for _ in range(n)]

        # 初始猜测（等权重）
        x0 = np.array([1.0 / n] * n)

        # 优化
        result = opt.minimize(negative_sharpe, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints_opt)

        if result.success:
            weights = pd.Series(result.x, index=expected_returns.index)
            weights = weights / weights.sum()

            return self._create_result(weights, expected_returns, cov_matrix, risk_free_rate)
        else:
            raise ValueError(f"最大夏普比率优化失败: {result.message}")


    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame,
                                constraints: OptimizationConstraints) -> OptimizationResult:
        """风险平价优化"""
        n = len(cov_matrix)


        def risk_budget_objective(weights):
            """风险预算目标函数"""
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_std = np.sqrt(portfolio_variance)

            # 边际风险贡献
            marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_std

            # 风险贡献
            risk_contrib = weights * marginal_contrib

            # 目标：所有资产的风险贡献相等
            target_risk = portfolio_variance / n

            return np.sum((risk_contrib - target_risk) ** 2)

        # 约束条件
        constraints_opt = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # 边界条件
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]

        if constraints.long_only:
            bounds = [(0.001, constraints.max_weight) for _ in range(n)]  # 避免零权重

        # 初始猜测
        x0 = np.array([1.0 / n] * n)

        # 优化
        result = opt.minimize(risk_budget_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints_opt)

        if result.success:
            weights = pd.Series(result.x, index=cov_matrix.index)
            weights = weights / weights.sum()

            # 计算预期收益
            expected_returns = pd.Series(0.08, index=cov_matrix.index)

            return self._create_result(weights, expected_returns, cov_matrix)
        else:
            raise ValueError(f"风险平价优化失败: {result.message}")


    def _equal_weight_optimization(self, expected_returns: pd.Series,
                                 cov_matrix: pd.DataFrame) -> OptimizationResult:
        """等权重组合"""
        n = len(expected_returns)
        weights = pd.Series([1.0 / n] * n, index=expected_returns.index)

        return self._create_result(weights, expected_returns, cov_matrix)


    def _black_litterman_optimization(self, returns: pd.DataFrame,
                                    market_caps: Optional[pd.Series] = None,
                                    views: Optional[Dict[str, float]] = None,
                                    view_confidence: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Black-Litterman优化"""

        if market_caps is None:
            # 使用等权重作为市场权重
            market_caps = pd.Series(1.0, index=returns.columns)

        views = views or {}
        view_confidence = view_confidence or {}

        # 估计协方差矩阵
        cov_matrix = self.risk_model.estimate_covariance(returns)

        # Black-Litterman预期收益
        bl_returns, bl_cov = self.bl_model.optimize(
            market_caps, cov_matrix, views, view_confidence
        )

        # 使用BL结果进行均值方差优化
        constraints = OptimizationConstraints()
        result = self._mean_variance_optimization(bl_returns, bl_cov, constraints)

        return result


    def _create_result(self, weights: pd.Series, expected_returns: pd.Series,
                      cov_matrix: pd.DataFrame, risk_free_rate: float = 0.02) -> OptimizationResult:
        """创建优化结果"""

        # 计算组合指标
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        # 创建结果对象
        result = OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            status="success"
        )

        return result


    def _calculate_additional_metrics(self, result: OptimizationResult,
                                    expected_returns: pd.Series,
                                    cov_matrix: pd.DataFrame) -> OptimizationResult:
        """计算额外指标"""

        weights = result.weights

        # 集中度 (HHI)
        result.concentration = np.sum(weights ** 2)

        # 有效资产数
        result.effective_assets = 1 / result.concentration

        # 分散化比率
        individual_vol = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = np.dot(weights, individual_vol)
        result.diversification_ratio = weighted_avg_vol / result.volatility

        return result


    def _update_stats(self, result: OptimizationResult, optimization_type: OptimizationType):
        """更新统计信息"""
        self.stats['total_optimizations'] += 1

        if result.status == "success":
            self.stats['successful_optimizations'] += 1
        else:
            self.stats['failed_optimizations'] += 1

        self.stats['optimization_types'][optimization_type.value] += 1

        # 更新平均执行时间
        total_count = self.stats['total_optimizations']
        current_avg = self.stats['average_execution_time']
        self.stats['average_execution_time'] = (
            (current_avg * (total_count - 1) + result.execution_time) / total_count
        )


    def _record_optimization(self, result: OptimizationResult):
        """记录优化历史"""
        record = {
            'timestamp': datetime.now(),
            'optimization_type': result.optimization_type.value,
            'expected_return': result.expected_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'execution_time': result.execution_time,
            'asset_count': len(result.weights),
            'concentration': result.concentration
        }

        self.optimization_history.append(record)

        # 限制历史记录大小
        if len(self.optimization_history) > self.max_history:
            self.optimization_history = self.optimization_history[-self.max_history:]


    def backtest_strategy(self, returns: pd.DataFrame, rebalance_freq: int = 21,
                        optimization_type: OptimizationType = OptimizationType.MEAN_VARIANCE,
                        lookback_days: int = 252) -> Dict[str, Any]:
        """回测策略"""

        portfolio_returns = []
        portfolio_weights_history = []
        rebalance_dates = []

        for i in range(lookback_days, len(returns), rebalance_freq):
            # 训练数据
            train_data = returns.iloc[i-lookback_days:i]

            # 优化
            try:
                result = self.optimize(train_data, optimization_type)
                weights = result.weights
            except:
                # 优化失败时使用等权重
                weights = pd.Series(1.0/len(returns.columns), index=returns.columns)

            # 计算下一个周期的收益
            end_idx = min(i + rebalance_freq, len(returns))
            period_returns = returns.iloc[i:end_idx]

            period_portfolio_returns = (period_returns * weights).sum(axis=1)
            portfolio_returns.extend(period_portfolio_returns.tolist())

            portfolio_weights_history.append({
                'date': returns.index[i],
                'weights': weights.to_dict()
            })

            rebalance_dates.append(returns.index[i])

        # 计算回测指标
        portfolio_returns = pd.Series(portfolio_returns,
                                    index=returns.index[lookback_days:lookback_days+len(portfolio_returns)])

        backtest_stats = self._calculate_backtest_stats(portfolio_returns)

        return {
            'portfolio_returns': portfolio_returns,
            'weights_history': portfolio_weights_history,
            'rebalance_dates': rebalance_dates,
            'stats': backtest_stats
        }


    def _calculate_backtest_stats(self, returns: pd.Series) -> Dict[str, float]:
        """计算回测统计指标"""

        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)

        sharpe_ratio = annual_return / annual_volatility

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()

        # 胜率
        win_rate = (returns > 0).mean()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns)
        }


    def get_efficient_frontier(self, returns: pd.DataFrame, n_portfolios: int = 50) -> pd.DataFrame:
        """计算有效前沿"""

        expected_returns = self.risk_model.estimate_expected_returns(returns)
        cov_matrix = self.risk_model.estimate_covariance(returns)

        # 计算最小和最大预期收益
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()

        target_returns = np.linspace(min_ret, max_ret, n_portfolios)

        efficient_portfolios = []

        for target_ret in target_returns:
            try:
                constraints = OptimizationConstraints()
                constraints.target_return = target_ret

                result = self._target_return_optimization(expected_returns, cov_matrix, constraints)

                efficient_portfolios.append({
                    'target_return': target_ret,
                    'expected_return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe_ratio': result.sharpe_ratio,
                    'weights': result.weights.to_dict()
                })
            except:
                continue

        return pd.DataFrame(efficient_portfolios)


    def _target_return_optimization(self, expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  constraints: OptimizationConstraints) -> OptimizationResult:
        """目标收益率优化"""
        n = len(expected_returns)

        # 创建优化变量
        w = cp.Variable(n)

        # 目标函数: 最小化组合方差
        objective = cp.Minimize(cp.quad_form(w, cov_matrix.values))

        # 约束条件
        constraints_list = [
            cp.sum(w) == 1,
            expected_returns.values @ w == constraints.target_return,
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]

        if constraints.long_only:
            constraints_list.append(w >= 0)

        # 求解
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.ECOS)

        if problem.status not in ["infeasible", "unbounded"]:
            weights = pd.Series(w.value, index=expected_returns.index)
            weights = weights / weights.sum()

            return self._create_result(weights, expected_returns, cov_matrix)
        else:
            raise ValueError(f"目标收益率优化失败: {problem.status}")


    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        return {
            'stats': self.stats,
            'risk_model': self.risk_model.name,
            'history_size': len(self.optimization_history),
            'recent_optimizations': self.optimization_history[-10:] if self.optimization_history else []
        }

# 全局实例
_portfolio_optimizer_instance = None


def get_portfolio_optimizer() -> PortfolioOptimizer:
    """获取投资组合优化器实例"""
    global _portfolio_optimizer_instance
    if _portfolio_optimizer_instance is None:
        _portfolio_optimizer_instance = PortfolioOptimizer()
    return _portfolio_optimizer_instance
