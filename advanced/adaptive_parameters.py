"""
Adaptive Parameters
自适应参数系统，基于市场条件和策略性能动态调整交易参数
支持遗传算法、粒子群优化、贝叶斯优化等多种参数优化方法
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class OptimizationMethod(Enum):
    """优化方法"""
    GRID_SEARCH = "grid_search"              # 网格搜索
    RANDOM_SEARCH = "random_search"          # 随机搜索
    GENETIC_ALGORITHM = "genetic_algorithm"  # 遗传算法
    PARTICLE_SWARM = "particle_swarm"        # 粒子群优化
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"  # 贝叶斯优化
    DIFFERENTIAL_EVOLUTION = "differential_evolution"  # 差分进化
    SIMULATED_ANNEALING = "simulated_annealing"  # 模拟退火
    ADAPTIVE_LEARNING = "adaptive_learning"  # 自适应学习

class ParameterType(Enum):
    """参数类型"""
    CONTINUOUS = "continuous"    # 连续参数
    DISCRETE = "discrete"       # 离散参数
    CATEGORICAL = "categorical" # 分类参数
    BOOLEAN = "boolean"         # 布尔参数

class AdaptationTrigger(Enum):
    """适应触发条件"""
    TIME_BASED = "time_based"           # 基于时间
    PERFORMANCE_BASED = "performance_based"  # 基于性能
    MARKET_REGIME_CHANGE = "market_regime_change"  # 市场状态变化
    VOLATILITY_CHANGE = "volatility_change"  # 波动率变化
    DRAWDOWN_THRESHOLD = "drawdown_threshold"  # 回撤阈值
    MANUAL = "manual"                   # 手动触发

@dataclass

class ParameterSpec:
    """参数规格"""
    name: str
    param_type: ParameterType

    # 数值范围
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # 离散选项
    discrete_values: Optional[List[Any]] = None

    # 默认值
    default_value: Any = None

    # 约束条件
    constraints: List[Callable] = field(default_factory=list)

    # 描述
    description: str = ""

    def validate_value(self, value: Any) -> bool:
        """验证参数值"""
        if self.param_type == ParameterType.CONTINUOUS:
            return (self.min_value is None or value >= self.min_value) and \
                   (self.max_value is None or value <= self.max_value)
        elif self.param_type == ParameterType.DISCRETE:
            return value in (self.discrete_values or [])
        elif self.param_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        else:
            return True

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'param_type': self.param_type.value,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'discrete_values': self.discrete_values,
            'default_value': self.default_value,
            'description': self.description
        }

@dataclass

class OptimizationResult:
    """优化结果"""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]

    # 优化信息
    method_used: OptimizationMethod
    iterations: int
    convergence_achieved: bool

    # 统计信息
    start_time: datetime
    end_time: datetime
    evaluation_count: int

    def to_dict(self) -> dict:
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'method_used': self.method_used.value,
            'iterations': self.iterations,
            'convergence_achieved': self.convergence_achieved,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'evaluation_count': self.evaluation_count,
            'optimization_history': self.optimization_history[-10:]  # 只返回最近10次
        }

@dataclass

class AdaptationEvent:
    """适应事件"""
    timestamp: datetime
    trigger: AdaptationTrigger

    # 参数变化
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]

    # 性能变化
    old_performance: float
    expected_performance: float

    # 触发原因
    trigger_reason: str
    trigger_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'trigger': self.trigger.value,
            'parameter_changes': {
                param: {'old': self.old_parameters.get(param), 'new': self.new_parameters.get(param)}
                for param in set(list(self.old_parameters.keys()) + list(self.new_parameters.keys()))
                if self.old_parameters.get(param) != self.new_parameters.get(param)
            },
            'performance_change': {
                'old': self.old_performance,
                'expected': self.expected_performance,
                'improvement': self.expected_performance - self.old_performance
            },
            'trigger_reason': self.trigger_reason,
            'trigger_value': self.trigger_value
        }

class BaseOptimizer(ABC):
    """基础优化器"""

    def __init__(self, name: str):
        self.name = name
        self.evaluation_count = 0
        self.optimization_history = []

    @abstractmethod

    def optimize(self, objective_function: Callable, parameter_specs: List[ParameterSpec],
                max_evaluations: int = 100) -> OptimizationResult:
        """执行优化"""
        pass

    def _evaluate_parameters(self, parameters: Dict[str, Any],
                           objective_function: Callable) -> float:
        """评估参数"""
        self.evaluation_count += 1

        try:
            score = objective_function(parameters)

            # 记录评估历史
            self.optimization_history.append({
                'evaluation': self.evaluation_count,
                'parameters': parameters.copy(),
                'score': score,
                'timestamp': datetime.now()
            })

            return score

        except Exception as e:
            logger.error(f"参数评估失败: {e}")
            return -float('inf')

class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""

    def __init__(self, grid_resolution: int = 10):
        super().__init__("Grid Search")
        self.grid_resolution = grid_resolution

    def optimize(self, objective_function: Callable, parameter_specs: List[ParameterSpec],
                max_evaluations: int = 100) -> OptimizationResult:
        """网格搜索优化"""
        start_time = datetime.now()

        # 生成参数网格
        param_grids = {}
        for spec in parameter_specs:
            if spec.param_type == ParameterType.CONTINUOUS:
                param_grids[spec.name] = np.linspace(
                    spec.min_value, spec.max_value, self.grid_resolution
                )
            elif spec.param_type == ParameterType.DISCRETE:
                param_grids[spec.name] = spec.discrete_values
            elif spec.param_type == ParameterType.BOOLEAN:
                param_grids[spec.name] = [True, False]

        # 执行网格搜索
        best_score = -float('inf')
        best_parameters = {}
        evaluations = 0

        def recursive_search(param_names, current_params):
            nonlocal best_score, best_parameters, evaluations

            if evaluations >= max_evaluations:
                return

            if not param_names:
                # 评估当前参数组合
                score = self._evaluate_parameters(current_params, objective_function)
                evaluations += 1

                if score > best_score:
                    best_score = score
                    best_parameters = current_params.copy()
                return

            # 递归搜索下一个参数
            param_name = param_names[0]
            remaining_params = param_names[1:]

            for value in param_grids[param_name]:
                if evaluations >= max_evaluations:
                    break

                current_params[param_name] = value
                recursive_search(remaining_params, current_params)

        # 开始搜索
        recursive_search(list(param_grids.keys()), {})

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.optimization_history,
            method_used=OptimizationMethod.GRID_SEARCH,
            iterations=evaluations,
            convergence_achieved=evaluations < max_evaluations,
            start_time=start_time,
            end_time=datetime.now(),
            evaluation_count=evaluations
        )

class GeneticAlgorithmOptimizer(BaseOptimizer):
    """遗传算法优化器"""

    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elite_ratio: float = 0.1):
        super().__init__("Genetic Algorithm")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio

    def optimize(self, objective_function: Callable, parameter_specs: List[ParameterSpec],
                max_evaluations: int = 100) -> OptimizationResult:
        """遗传算法优化"""
        start_time = datetime.now()

        # 初始化种群
        population = self._initialize_population(parameter_specs)

        best_score = -float('inf')
        best_parameters = {}
        generation = 0
        evaluations = 0

        while evaluations < max_evaluations:
            # 评估种群
            fitness_scores = []
            for individual in population:
                if evaluations >= max_evaluations:
                    break

                score = self._evaluate_parameters(individual, objective_function)
                fitness_scores.append(score)
                evaluations += 1

                if score > best_score:
                    best_score = score
                    best_parameters = individual.copy()

            if evaluations >= max_evaluations:
                break

            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores, parameter_specs)
            generation += 1

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.optimization_history,
            method_used=OptimizationMethod.GENETIC_ALGORITHM,
            iterations=generation,
            convergence_achieved=False,  # GA通常不收敛
            start_time=start_time,
            end_time=datetime.now(),
            evaluation_count=evaluations
        )

    def _initialize_population(self, parameter_specs: List[ParameterSpec]) -> List[Dict[str, Any]]:
        """初始化种群"""
        population = []

        for _ in range(self.population_size):
            individual = {}
            for spec in parameter_specs:
                if spec.param_type == ParameterType.CONTINUOUS:
                    individual[spec.name] = np.random.uniform(spec.min_value, spec.max_value)
                elif spec.param_type == ParameterType.DISCRETE:
                    individual[spec.name] = np.random.choice(spec.discrete_values)
                elif spec.param_type == ParameterType.BOOLEAN:
                    individual[spec.name] = np.random.choice([True, False])

            population.append(individual)

        return population

    def _evolve_population(self, population: List[Dict[str, Any]],
                          fitness_scores: List[float],
                          parameter_specs: List[ParameterSpec]) -> List[Dict[str, Any]]:
        """进化种群"""
        # 排序并选择精英
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elite_count = int(self.population_size * self.elite_ratio)

        new_population = []

        # 保留精英
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())

        # 生成新个体
        while len(new_population) < self.population_size:
            # 选择父母
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)

            # 交叉
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, parameter_specs)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 变异
            child1 = self._mutate(child1, parameter_specs)
            child2 = self._mutate(child2, parameter_specs)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return new_population[:self.population_size]

    def _tournament_selection(self, population: List[Dict[str, Any]],
                            fitness_scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """锦标赛选择"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[np.argmax([fitness_scores[i] for i in indices])]
        return population[best_idx].copy()

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                  parameter_specs: List[ParameterSpec]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """交叉操作"""
        child1, child2 = parent1.copy(), parent2.copy()

        for spec in parameter_specs:
            if np.random.random() < 0.5:
                child1[spec.name], child2[spec.name] = child2[spec.name], child1[spec.name]

        return child1, child2

    def _mutate(self, individual: Dict[str, Any],
               parameter_specs: List[ParameterSpec]) -> Dict[str, Any]:
        """变异操作"""
        mutated = individual.copy()

        for spec in parameter_specs:
            if np.random.random() < self.mutation_rate:
                if spec.param_type == ParameterType.CONTINUOUS:
                    # 高斯变异
                    current_value = mutated[spec.name]
                    range_size = spec.max_value - spec.min_value
                    mutation = np.random.normal(0, range_size * 0.1)
                    new_value = np.clip(current_value + mutation, spec.min_value, spec.max_value)
                    mutated[spec.name] = new_value

                elif spec.param_type == ParameterType.DISCRETE:
                    mutated[spec.name] = np.random.choice(spec.discrete_values)

                elif spec.param_type == ParameterType.BOOLEAN:
                    mutated[spec.name] = not mutated[spec.name]

        return mutated

class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器"""

    def __init__(self, acquisition_function: str = 'ei', kappa: float = 2.576):
        super().__init__("Bayesian Optimization")
        self.acquisition_function = acquisition_function
        self.kappa = kappa
        self.gp = None

    def optimize(self, objective_function: Callable, parameter_specs: List[ParameterSpec],
                max_evaluations: int = 100) -> OptimizationResult:
        """贝叶斯优化"""
        start_time = datetime.now()

        # 只处理连续参数
        continuous_specs = [s for s in parameter_specs if s.param_type == ParameterType.CONTINUOUS]

        if not continuous_specs:
            logger.warning("贝叶斯优化仅支持连续参数")
            return self._fallback_random_search(objective_function, parameter_specs, max_evaluations)

        # 初始化高斯过程
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        # 初始随机采样
        n_initial = min(10, max_evaluations // 2)
        X_train = []
        y_train = []

        best_score = -float('inf')
        best_parameters = {}

        # 初始采样
        for _ in range(n_initial):
            params = self._random_sample(parameter_specs)
            score = self._evaluate_parameters(params, objective_function)

            # 转换为向量形式（仅连续参数）
            x = [params[spec.name] for spec in continuous_specs]
            X_train.append(x)
            y_train.append(score)

            if score > best_score:
                best_score = score
                best_parameters = params.copy()

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 贝叶斯优化循环
        for iteration in range(n_initial, max_evaluations):
            # 拟合高斯过程
            self.gp.fit(X_train, y_train)

            # 优化采集函数
            next_x = self._optimize_acquisition(continuous_specs)

            # 构建完整参数
            next_params = {}
            for i, spec in enumerate(continuous_specs):
                next_params[spec.name] = next_x[i]

            # 添加非连续参数的随机值
            for spec in parameter_specs:
                if spec.param_type == ParameterType.DISCRETE:
                    next_params[spec.name] = np.random.choice(spec.discrete_values)
                elif spec.param_type == ParameterType.BOOLEAN:
                    next_params[spec.name] = np.random.choice([True, False])

            # 评估新参数
            score = self._evaluate_parameters(next_params, objective_function)

            # 更新训练数据
            X_train = np.vstack([X_train, next_x])
            y_train = np.append(y_train, score)

            if score > best_score:
                best_score = score
                best_parameters = next_params.copy()

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.optimization_history,
            method_used=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            iterations=max_evaluations - n_initial,
            convergence_achieved=True,
            start_time=start_time,
            end_time=datetime.now(),
            evaluation_count=max_evaluations
        )

    def _random_sample(self, parameter_specs: List[ParameterSpec]) -> Dict[str, Any]:
        """随机采样"""
        params = {}
        for spec in parameter_specs:
            if spec.param_type == ParameterType.CONTINUOUS:
                params[spec.name] = np.random.uniform(spec.min_value, spec.max_value)
            elif spec.param_type == ParameterType.DISCRETE:
                params[spec.name] = np.random.choice(spec.discrete_values)
            elif spec.param_type == ParameterType.BOOLEAN:
                params[spec.name] = np.random.choice([True, False])
        return params

    def _optimize_acquisition(self, continuous_specs: List[ParameterSpec]) -> np.ndarray:
        """优化采集函数"""
        bounds = [(spec.min_value, spec.max_value) for spec in continuous_specs]

        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)

            if self.acquisition_function == 'ei':
                # Expected Improvement
                improvement = mu - np.max(self.gp.y_train_)
                Z = improvement / sigma
                ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
                return -ei[0]  # 负号因为minimize
            else:
                # Upper Confidence Bound
                return -(mu + self.kappa * sigma)[0]

        # 多次随机初始化优化
        best_x = None
        best_acq = float('inf')

        for _ in range(10):
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            result = minimize(acquisition, x0, bounds=bounds, method='L-BFGS-B')

            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        return best_x

    def _fallback_random_search(self, objective_function: Callable,
                               parameter_specs: List[ParameterSpec],
                               max_evaluations: int) -> OptimizationResult:
        """随机搜索后备方案"""
        start_time = datetime.now()

        best_score = -float('inf')
        best_parameters = {}

        for _ in range(max_evaluations):
            params = self._random_sample(parameter_specs)
            score = self._evaluate_parameters(params, objective_function)

            if score > best_score:
                best_score = score
                best_parameters = params.copy()

        return OptimizationResult(
            best_parameters=best_parameters,
            best_score=best_score,
            optimization_history=self.optimization_history,
            method_used=OptimizationMethod.RANDOM_SEARCH,
            iterations=max_evaluations,
            convergence_achieved=False,
            start_time=start_time,
            end_time=datetime.now(),
            evaluation_count=max_evaluations
        )

class AdaptiveParameterSystem:
    """自适应参数系统"""

    def __init__(self):
        self.parameter_specs = {}
        self.current_parameters = {}
        self.optimizers = {}

        # 适应历史
        self.adaptation_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)

        # 触发配置
        self.adaptation_triggers = {}

        # 统计信息
        self.stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'average_improvement': 0.0,
            'last_adaptation_time': None
        }

        # 注册默认优化器
        self._register_optimizers()

        logger.info("自适应参数系统初始化完成")

    def _register_optimizers(self):
        """注册优化器"""
        self.optimizers[OptimizationMethod.GRID_SEARCH] = GridSearchOptimizer()
        self.optimizers[OptimizationMethod.GENETIC_ALGORITHM] = GeneticAlgorithmOptimizer()
        self.optimizers[OptimizationMethod.BAYESIAN_OPTIMIZATION] = BayesianOptimizer()

    def register_parameter(self, param_spec: ParameterSpec):
        """注册参数"""
        self.parameter_specs[param_spec.name] = param_spec

        # 设置默认值
        if param_spec.default_value is not None:
            self.current_parameters[param_spec.name] = param_spec.default_value

        logger.info(f"注册参数: {param_spec.name} ({param_spec.param_type.value})")

    def set_adaptation_trigger(self, trigger: AdaptationTrigger,
                              threshold: Optional[float] = None,
                              check_interval: Optional[int] = None):
        """设置适应触发条件"""
        self.adaptation_triggers[trigger] = {
            'threshold': threshold,
            'check_interval': check_interval,
            'last_check': datetime.now()
        }

        logger.info(f"设置适应触发: {trigger.value}")

    def update_performance(self, performance_score: float, timestamp: datetime = None):
        """更新性能数据"""
        timestamp = timestamp or datetime.now()

        self.performance_history.append({
            'timestamp': timestamp,
            'score': performance_score,
            'parameters': self.current_parameters.copy()
        })

        # 检查是否需要触发适应
        self._check_adaptation_triggers(performance_score, timestamp)

    def _check_adaptation_triggers(self, current_performance: float, timestamp: datetime):
        """检查适应触发条件"""
        for trigger, config in self.adaptation_triggers.items():
            should_trigger = False
            trigger_reason = ""
            trigger_value = None

            if trigger == AdaptationTrigger.PERFORMANCE_BASED:
                if len(self.performance_history) >= 10:
                    recent_scores = [p['score'] for p in list(self.performance_history)[-10:]]
                    avg_recent = np.mean(recent_scores)

                    if config['threshold'] and avg_recent < config['threshold']:
                        should_trigger = True
                        trigger_reason = f"性能低于阈值: {avg_recent:.4f} < {config['threshold']}"
                        trigger_value = avg_recent

            elif trigger == AdaptationTrigger.TIME_BASED:
                time_since_last = (timestamp - config['last_check']).total_seconds()
                check_interval = config['check_interval'] or 3600  # 默认1小时

                if time_since_last >= check_interval:
                    should_trigger = True
                    trigger_reason = f"定时触发: {time_since_last:.0f}秒 >= {check_interval}秒"
                    config['last_check'] = timestamp

            elif trigger == AdaptationTrigger.DRAWDOWN_THRESHOLD:
                if len(self.performance_history) >= 5:
                    scores = [p['score'] for p in self.performance_history]
                    peak = max(scores)
                    drawdown = (peak - current_performance) / peak if peak > 0 else 0

                    if config['threshold'] and drawdown > config['threshold']:
                        should_trigger = True
                        trigger_reason = f"回撤超过阈值: {drawdown:.4f} > {config['threshold']}"
                        trigger_value = drawdown

            if should_trigger:
                self._trigger_adaptation(trigger, trigger_reason, trigger_value)

    def _trigger_adaptation(self, trigger: AdaptationTrigger, reason: str, value: Optional[float]):
        """触发参数适应"""
        logger.info(f"触发参数适应: {trigger.value} - {reason}")

        # 选择优化方法
        if trigger == AdaptationTrigger.PERFORMANCE_BASED:
            method = OptimizationMethod.BAYESIAN_OPTIMIZATION
        elif trigger == AdaptationTrigger.TIME_BASED:
            method = OptimizationMethod.GENETIC_ALGORITHM
        else:
            method = OptimizationMethod.GRID_SEARCH

        # 执行优化
        try:
            self.optimize_parameters(
                method=method,
                max_evaluations=20,  # 适应时使用较少的评估次数
                trigger=trigger,
                trigger_reason=reason,
                trigger_value=value
            )
        except Exception as e:
            logger.error(f"参数适应失败: {e}")

    def optimize_parameters(self, objective_function: Optional[Callable] = None,
                          method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION,
                          max_evaluations: int = 100,
                          trigger: Optional[AdaptationTrigger] = None,
                          trigger_reason: str = "",
                          trigger_value: Optional[float] = None) -> OptimizationResult:
        """优化参数"""

        if not self.parameter_specs:
            raise ValueError("没有注册的参数")

        if method not in self.optimizers:
            raise ValueError(f"不支持的优化方法: {method}")

        # 使用默认目标函数（基于历史性能）
        if objective_function is None:
            objective_function = self._default_objective_function

        old_parameters = self.current_parameters.copy()
        old_performance = self._get_current_performance()

        # 执行优化
        optimizer = self.optimizers[method]
        result = optimizer.optimize(
            objective_function,
            list(self.parameter_specs.values()),
            max_evaluations
        )

        # 更新当前参数
        if result.best_score > old_performance:
            self.current_parameters.update(result.best_parameters)

            # 记录适应事件
            adaptation_event = AdaptationEvent(
                timestamp=datetime.now(),
                trigger=trigger or AdaptationTrigger.MANUAL,
                old_parameters=old_parameters,
                new_parameters=self.current_parameters.copy(),
                old_performance=old_performance,
                expected_performance=result.best_score,
                trigger_reason=trigger_reason,
                trigger_value=trigger_value
            )

            self.adaptation_history.append(adaptation_event)

            # 更新统计
            self.stats['total_adaptations'] += 1
            self.stats['successful_adaptations'] += 1
            self.stats['last_adaptation_time'] = datetime.now()

            improvement = result.best_score - old_performance
            total_improvements = self.stats['successful_adaptations']
            current_avg = self.stats['average_improvement']
            self.stats['average_improvement'] = (
                (current_avg * (total_improvements - 1) + improvement) / total_improvements
            )

            logger.info(f"参数优化成功，性能提升: {improvement:.4f}")
        else:
            self.stats['total_adaptations'] += 1
            logger.info("参数优化未带来性能提升，保持原参数")

        return result

    def _default_objective_function(self, parameters: Dict[str, Any]) -> float:
        """默认目标函数（基于历史性能模拟）"""
        if not self.performance_history:
            return 0.0

        # 简化的目标函数：基于参数与历史最佳参数的相似度
        best_performance = max(p['score'] for p in self.performance_history)
        best_params = None

        for p in self.performance_history:
            if p['score'] == best_performance:
                best_params = p['parameters']
                break

        if not best_params:
            return np.random.normal(0, 0.1)  # 随机分数

        # 计算参数相似度
        similarity = 0.0
        param_count = 0

        for param_name, value in parameters.items():
            if param_name in best_params:
                if isinstance(value, (int, float)):
                    # 数值参数：基于差异计算相似度
                    diff = abs(value - best_params[param_name])
                    max_diff = abs(self.parameter_specs[param_name].max_value -
                                 self.parameter_specs[param_name].min_value)
                    similarity += 1 - (diff / max_diff) if max_diff > 0 else 1
                else:
                    # 分类参数：完全匹配
                    similarity += 1 if value == best_params[param_name] else 0
                param_count += 1

        if param_count > 0:
            similarity /= param_count

        # 添加随机噪声模拟真实性能变化
        return similarity * best_performance + np.random.normal(0, 0.1)

    def _get_current_performance(self) -> float:
        """获取当前性能"""
        if not self.performance_history:
            return 0.0

        # 返回最近性能的平均值
        recent_scores = [p['score'] for p in list(self.performance_history)[-5:]]
        return np.mean(recent_scores)

    def get_parameter(self, name: str) -> Any:
        """获取参数值"""
        return self.current_parameters.get(name)

    def set_parameter(self, name: str, value: Any) -> bool:
        """设置参数值"""
        if name not in self.parameter_specs:
            logger.error(f"未知参数: {name}")
            return False

        spec = self.parameter_specs[name]
        if not spec.validate_value(value):
            logger.error(f"参数值无效: {name} = {value}")
            return False

        old_value = self.current_parameters.get(name)
        self.current_parameters[name] = value

        logger.info(f"参数更新: {name} = {old_value} -> {value}")
        return True

    def get_adaptation_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """获取适应摘要"""
        cutoff_time = datetime.now() - timedelta(days=days_back)

        recent_adaptations = [
            a for a in self.adaptation_history
            if a.timestamp >= cutoff_time
        ]

        if not recent_adaptations:
            return {'message': '没有最近的适应记录'}

        # 统计分析
        trigger_counts = defaultdict(int)
        total_improvement = 0.0

        for adaptation in recent_adaptations:
            trigger_counts[adaptation.trigger.value] += 1
            total_improvement += adaptation.expected_performance - adaptation.old_performance

        return {
            'time_period_days': days_back,
            'total_adaptations': len(recent_adaptations),
            'average_improvement': total_improvement / len(recent_adaptations),
            'trigger_distribution': dict(trigger_counts),
            'current_parameters': self.current_parameters.copy(),
            'recent_adaptations': [a.to_dict() for a in recent_adaptations[-5:]]
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        return {
            'stats': self.stats,
            'registered_parameters': {
                name: spec.to_dict() for name, spec in self.parameter_specs.items()
            },
            'current_parameters': self.current_parameters.copy(),
            'active_triggers': list(self.adaptation_triggers.keys()),
            'available_optimizers': list(self.optimizers.keys()),
            'adaptation_history_size': len(self.adaptation_history),
            'performance_history_size': len(self.performance_history)
        }

# 便利函数

def create_common_parameter_specs() -> List[ParameterSpec]:
    """创建常用参数规格"""
    return [
        ParameterSpec(
            name="lookback_period",
            param_type=ParameterType.DISCRETE,
            discrete_values=[5, 10, 15, 20, 25, 30, 40, 50],
            default_value=20,
            description="回看周期"
        ),
        ParameterSpec(
            name="threshold",
            param_type=ParameterType.CONTINUOUS,
            min_value=0.001,
            max_value=0.1,
            default_value=0.02,
            description="阈值参数"
        ),
        ParameterSpec(
            name="use_volume_filter",
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            description="是否使用成交量过滤"
        ),
        ParameterSpec(
            name="risk_level",
            param_type=ParameterType.DISCRETE,
            discrete_values=["low", "medium", "high"],
            default_value="medium",
            description="风险级别"
        )
    ]

# 全局实例
_adaptive_parameter_system_instance = None

def get_adaptive_parameter_system() -> AdaptiveParameterSystem:
    """获取自适应参数系统实例"""
    global _adaptive_parameter_system_instance
    if _adaptive_parameter_system_instance is None:
        _adaptive_parameter_system_instance = AdaptiveParameterSystem()
    return _adaptive_parameter_system_instance
