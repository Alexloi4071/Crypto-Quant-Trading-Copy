"""
Model Ensemble
模型集成系统，支持多种机器学习和深度学习模型的集成
提供投票机制、权重分配和动态模型选择等高级集成策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import sys
from pathlib import Path
import pickle
import joblib
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class EnsembleMethod(Enum):
    """集成方法"""
    VOTING = "voting"                    # 投票法
    AVERAGING = "averaging"              # 平均法
    WEIGHTED_AVERAGING = "weighted_averaging"  # 加权平均
    STACKING = "stacking"               # 堆叠法
    BLENDING = "blending"               # 混合法
    BAGGING = "bagging"                 # 装袋法
    BOOSTING = "boosting"               # 提升法
    DYNAMIC_SELECTION = "dynamic_selection"  # 动态选择

class ModelType(Enum):
    """模型类型"""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    PREDICTOR = "predictor"
    STRATEGY = "strategy"

class WeightingStrategy(Enum):
    """权重策略"""
    EQUAL = "equal"                     # 等权重
    PERFORMANCE_BASED = "performance_based"  # 基于性能
    CONFIDENCE_BASED = "confidence_based"    # 基于置信度
    DYNAMIC = "dynamic"                 # 动态权重
    INVERSE_ERROR = "inverse_error"     # 反误差权重

@dataclass

class ModelMetadata:
    """模型元数据"""
    model_id: str
    model_name: str
    model_type: ModelType

    # 性能指标
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = float('inf')
    r2_score: float = 0.0

    # 模型特征
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    complexity: int = 1  # 1-10复杂度等级

    # 适用场景
    market_conditions: List[str] = field(default_factory=list)
    asset_types: List[str] = field(default_factory=list)
    time_horizons: List[str] = field(default_factory=list)

    # 状态信息
    is_trained: bool = False
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type.value,
            'performance': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'mse': self.mse,
                'r2_score': self.r2_score
            },
            'characteristics': {
                'training_time': self.training_time,
                'prediction_time': self.prediction_time,
                'memory_usage': self.memory_usage,
                'complexity': self.complexity
            },
            'applicability': {
                'market_conditions': self.market_conditions,
                'asset_types': self.asset_types,
                'time_horizons': self.time_horizons
            },
            'status': {
                'is_trained': self.is_trained,
                'last_updated': self.last_updated.isoformat(),
                'version': self.version
            }
        }

@dataclass

class EnsemblePrediction:
    """集成预测结果"""
    prediction_id: str
    ensemble_method: EnsembleMethod

    # 预测结果
    final_prediction: Union[float, int, List[float]]
    confidence: float

    # 模型贡献
    individual_predictions: Dict[str, Any] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)

    # 统计信息
    prediction_variance: float = 0.0
    consensus_level: float = 0.0  # 模型间一致性

    # 元信息
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    models_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'prediction_id': self.prediction_id,
            'ensemble_method': self.ensemble_method.value,
            'final_prediction': self.final_prediction,
            'confidence': self.confidence,
            'individual_predictions': self.individual_predictions,
            'model_weights': self.model_weights,
            'model_confidences': self.model_confidences,
            'statistics': {
                'prediction_variance': self.prediction_variance,
                'consensus_level': self.consensus_level
            },
            'metadata': {
                'timestamp': self.timestamp.isoformat(),
                'processing_time': self.processing_time,
                'models_used': self.models_used
            }
        }

class BaseModel(ABC):
    """基础模型接口"""

    def __init__(self, model_id: str, model_name: str):
        self.model_id = model_id
        self.model_name = model_name
        self.is_fitted = False
        self.metadata = ModelMetadata(model_id, model_name, self._get_model_type())

    @abstractmethod

    def _get_model_type(self) -> ModelType:
        """获取模型类型"""
        pass

    @abstractmethod

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """训练模型"""
        pass

    @abstractmethod

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """预测，返回预测结果和置信度"""
        pass

    @abstractmethod

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        predictions, _ = self.predict(X)

        if self.metadata.model_type == ModelType.CLASSIFIER:
            accuracy = accuracy_score(y, predictions)
            f1 = f1_score(y, predictions, average='weighted')

            self.metadata.accuracy = accuracy
            self.metadata.f1_score = f1

            return {
                'accuracy': accuracy,
                'f1_score': f1
            }
        else:  # REGRESSOR
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)

            self.metadata.mse = mse
            self.metadata.r2_score = r2

            return {
                'mse': mse,
                'r2_score': r2,
                'rmse': np.sqrt(mse)
            }

class SklearnModelWrapper(BaseModel):
    """Sklearn模型包装器"""

    def __init__(self, model_id: str, model_name: str, sklearn_model, model_type: ModelType):
        super().__init__(model_id, model_name)
        self.sklearn_model = sklearn_model
        self.model_type = model_type
        self.metadata.model_type = model_type

    def _get_model_type(self) -> ModelType:
        return self.model_type

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnModelWrapper':
        """训练sklearn模型"""
        import time
        start_time = time.time()

        self.sklearn_model.fit(X, y)
        self.is_fitted = True
        self.metadata.is_trained = True
        self.metadata.training_time = time.time() - start_time
        self.metadata.last_updated = datetime.now()

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """预测"""
        import time
        start_time = time.time()

        predictions = self.sklearn_model.predict(X)

        # 计算置信度
        confidence = self._calculate_confidence(X, predictions)

        self.metadata.prediction_time = time.time() - start_time

        return predictions, confidence

    def _calculate_confidence(self, X: np.ndarray, predictions: np.ndarray) -> float:
        """计算预测置信度"""
        if hasattr(self.sklearn_model, 'predict_proba'):
            # 分类器：使用最高概率作为置信度
            probabilities = self.sklearn_model.predict_proba(X)
            confidence = np.mean(np.max(probabilities, axis=1))
        elif hasattr(self.sklearn_model, 'decision_function'):
            # SVM等：使用决策函数距离
            decision_scores = self.sklearn_model.decision_function(X)
            confidence = np.mean(np.abs(decision_scores)) / (np.mean(np.abs(decision_scores)) + 1)
        else:
            # 回归器：使用预测一致性
            if len(predictions) > 1:
                confidence = 1.0 / (1.0 + np.std(predictions))
            else:
                confidence = 0.7  # 默认置信度

        return min(max(confidence, 0.0), 1.0)

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(self.sklearn_model, 'feature_importances_'):
            importances = self.sklearn_model.feature_importances_
            return {f'feature_{i}': imp for i, imp in enumerate(importances)}
        elif hasattr(self.sklearn_model, 'coef_'):
            coefficients = self.sklearn_model.coef_
            if coefficients.ndim > 1:
                coefficients = np.mean(np.abs(coefficients), axis=0)
            return {f'feature_{i}': coef for i, coef in enumerate(coefficients)}
        else:
            return {}

class WeightCalculator:
    """权重计算器"""

    def __init__(self, strategy: WeightingStrategy):
        self.strategy = strategy
        self.weight_history = defaultdict(list)

    def calculate_weights(self, models: List[BaseModel],
                         performance_data: Dict[str, Dict[str, float]] = None,
                         predictions: Dict[str, Any] = None,
                         confidences: Dict[str, float] = None) -> Dict[str, float]:
        """计算模型权重"""

        if self.strategy == WeightingStrategy.EQUAL:
            return self._equal_weights(models)
        elif self.strategy == WeightingStrategy.PERFORMANCE_BASED:
            return self._performance_based_weights(models, performance_data)
        elif self.strategy == WeightingStrategy.CONFIDENCE_BASED:
            return self._confidence_based_weights(models, confidences)
        elif self.strategy == WeightingStrategy.DYNAMIC:
            return self._dynamic_weights(models, performance_data, predictions, confidences)
        elif self.strategy == WeightingStrategy.INVERSE_ERROR:
            return self._inverse_error_weights(models, performance_data)
        else:
            return self._equal_weights(models)

    def _equal_weights(self, models: List[BaseModel]) -> Dict[str, float]:
        """等权重"""
        weight = 1.0 / len(models)
        return {model.model_id: weight for model in models}

    def _performance_based_weights(self, models: List[BaseModel],
                                 performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """基于性能的权重"""
        if not performance_data:
            return self._equal_weights(models)

        weights = {}
        total_score = 0.0

        for model in models:
            if model.model_id in performance_data:
                perf = performance_data[model.model_id]

                # 根据模型类型选择主要指标
                if model.metadata.model_type == ModelType.CLASSIFIER:
                    score = perf.get('f1_score', perf.get('accuracy', 0.5))
                else:
                    score = perf.get('r2_score', 1.0 / (1.0 + perf.get('mse', 1.0)))

                weights[model.model_id] = max(score, 0.01)  # 最小权重
                total_score += weights[model.model_id]
            else:
                weights[model.model_id] = 0.1
                total_score += 0.1

        # 标准化权重
        if total_score > 0:
            for model_id in weights:
                weights[model_id] /= total_score

        return weights

    def _confidence_based_weights(self, models: List[BaseModel],
                                confidences: Dict[str, float]) -> Dict[str, float]:
        """基于置信度的权重"""
        if not confidences:
            return self._equal_weights(models)

        weights = {}
        total_confidence = 0.0

        for model in models:
            conf = confidences.get(model.model_id, 0.5)
            weights[model.model_id] = conf
            total_confidence += conf

        # 标准化
        if total_confidence > 0:
            for model_id in weights:
                weights[model_id] /= total_confidence

        return weights

    def _dynamic_weights(self, models: List[BaseModel],
                        performance_data: Dict[str, Dict[str, float]],
                        predictions: Dict[str, Any],
                        confidences: Dict[str, float]) -> Dict[str, float]:
        """动态权重"""
        # 结合性能和置信度
        perf_weights = self._performance_based_weights(models, performance_data)
        conf_weights = self._confidence_based_weights(models, confidences)

        # 加权平均
        weights = {}
        for model in models:
            model_id = model.model_id
            weights[model_id] = 0.7 * perf_weights.get(model_id, 0) + 0.3 * conf_weights.get(model_id, 0)

        return weights

    def _inverse_error_weights(self, models: List[BaseModel],
                             performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """反误差权重"""
        if not performance_data:
            return self._equal_weights(models)

        weights = {}
        total_inv_error = 0.0

        for model in models:
            if model.model_id in performance_data:
                perf = performance_data[model.model_id]

                if model.metadata.model_type == ModelType.CLASSIFIER:
                    error = 1.0 - perf.get('accuracy', 0.5)
                else:
                    error = perf.get('mse', 1.0)

                inv_error = 1.0 / (error + 0.001)  # 避免除零
                weights[model.model_id] = inv_error
                total_inv_error += inv_error
            else:
                weights[model.model_id] = 1.0
                total_inv_error += 1.0

        # 标准化
        if total_inv_error > 0:
            for model_id in weights:
                weights[model_id] /= total_inv_error

        return weights

class EnsembleEngine:
    """集成引擎"""

    def __init__(self, ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGING):
        self.ensemble_method = ensemble_method
        self.models = {}
        self.weight_calculator = WeightCalculator(WeightingStrategy.PERFORMANCE_BASED)

        # 历史记录
        self.prediction_history = deque(maxlen=1000)
        self.performance_history = defaultdict(list)

        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'average_confidence': 0.0,
            'models_count': 0,
            'ensemble_accuracy': 0.0
        }

    def add_model(self, model: BaseModel):
        """添加模型到集成"""
        self.models[model.model_id] = model
        self.stats['models_count'] = len(self.models)
        logger.info(f"添加模型到集成: {model.model_name} ({model.model_id})")

    def remove_model(self, model_id: str):
        """从集成中移除模型"""
        if model_id in self.models:
            del self.models[model_id]
            self.stats['models_count'] = len(self.models)
            logger.info(f"从集成中移除模型: {model_id}")

    def predict(self, X: np.ndarray, feature_names: List[str] = None) -> EnsemblePrediction:
        """集成预测"""
        import time
        start_time = time.time()

        if not self.models:
            raise ValueError("集成中没有可用的模型")

        # 获取各模型的预测结果
        individual_predictions = {}
        model_confidences = {}

        for model_id, model in self.models.items():
            if model.is_fitted:
                try:
                    pred, conf = model.predict(X)
                    individual_predictions[model_id] = pred
                    model_confidences[model_id] = conf
                except Exception as e:
                    logger.error(f"模型 {model_id} 预测失败: {e}")
                    continue

        if not individual_predictions:
            raise ValueError("没有模型能够成功预测")

        # 计算模型权重
        performance_data = self._get_recent_performance()
        model_weights = self.weight_calculator.calculate_weights(
            list(self.models.values()),
            performance_data,
            individual_predictions,
            model_confidences
        )

        # 执行集成预测
        final_prediction, confidence = self._ensemble_predict(
            individual_predictions, model_weights, model_confidences
        )

        # 计算统计信息
        prediction_variance = self._calculate_prediction_variance(individual_predictions)
        consensus_level = self._calculate_consensus_level(individual_predictions)

        # 创建预测结果
        prediction_id = f"ensemble_{int(datetime.now().timestamp())}"
        result = EnsemblePrediction(
            prediction_id=prediction_id,
            ensemble_method=self.ensemble_method,
            final_prediction=final_prediction,
            confidence=confidence,
            individual_predictions=individual_predictions,
            model_weights=model_weights,
            model_confidences=model_confidences,
            prediction_variance=prediction_variance,
            consensus_level=consensus_level,
            processing_time=time.time() - start_time,
            models_used=list(individual_predictions.keys())
        )

        # 更新统计信息
        self._update_stats(result)

        # 记录历史
        self.prediction_history.append(result)

        return result

    def _ensemble_predict(self, individual_predictions: Dict[str, Any],
                         model_weights: Dict[str, float],
                         model_confidences: Dict[str, float]) -> Tuple[Any, float]:
        """执行集成预测"""

        if self.ensemble_method == EnsembleMethod.VOTING:
            return self._voting_prediction(individual_predictions, model_weights)
        elif self.ensemble_method == EnsembleMethod.AVERAGING:
            return self._averaging_prediction(individual_predictions)
        elif self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGING:
            return self._weighted_averaging_prediction(individual_predictions, model_weights)
        else:
            # 默认使用加权平均
            return self._weighted_averaging_prediction(individual_predictions, model_weights)

    def _voting_prediction(self, predictions: Dict[str, Any], weights: Dict[str, float]) -> Tuple[Any, float]:
        """投票预测"""
        # 简化的投票实现
        pred_values = list(predictions.values())

        if isinstance(pred_values[0], np.ndarray):
            # 对于数组预测，使用众数
            from scipy import stats
            if pred_values[0].dtype in [int, bool]:
                # 分类问题
                votes = np.array(pred_values)
                final_pred = stats.mode(votes, axis=0)[0][0]
                confidence = np.mean([weights.get(k, 1.0) for k in predictions.keys()])
            else:
                # 回归问题，退化为平均
                final_pred = np.mean(pred_values, axis=0)
                confidence = np.mean([weights.get(k, 1.0) for k in predictions.keys()])
        else:
            # 单值预测
            final_pred = max(set(pred_values), key=pred_values.count)
            confidence = pred_values.count(final_pred) / len(pred_values)

        return final_pred, confidence

    def _averaging_prediction(self, predictions: Dict[str, Any]) -> Tuple[Any, float]:
        """平均预测"""
        pred_values = list(predictions.values())
        final_pred = np.mean(pred_values, axis=0)

        # 置信度基于预测的一致性
        if len(pred_values) > 1:
            variance = np.var(pred_values, axis=0)
            confidence = 1.0 / (1.0 + np.mean(variance))
        else:
            confidence = 0.8

        return final_pred, confidence

    def _weighted_averaging_prediction(self, predictions: Dict[str, Any],
                                     weights: Dict[str, float]) -> Tuple[Any, float]:
        """加权平均预测"""
        weighted_sum = None
        total_weight = 0.0
        confidence_sum = 0.0

        for model_id, pred in predictions.items():
            weight = weights.get(model_id, 1.0 / len(predictions))

            if weighted_sum is None:
                weighted_sum = pred * weight
            else:
                weighted_sum += pred * weight

            total_weight += weight
            confidence_sum += weight

        if total_weight > 0:
            final_pred = weighted_sum / total_weight
            confidence = confidence_sum / len(predictions)
        else:
            final_pred = np.mean(list(predictions.values()), axis=0)
            confidence = 0.5

        return final_pred, min(confidence, 1.0)

    def _calculate_prediction_variance(self, predictions: Dict[str, Any]) -> float:
        """计算预测方差"""
        if len(predictions) < 2:
            return 0.0

        pred_values = list(predictions.values())
        try:
            variance = np.var(pred_values, axis=0)
            return float(np.mean(variance))
        except:
            return 0.0

    def _calculate_consensus_level(self, predictions: Dict[str, Any]) -> float:
        """计算模型间一致性水平"""
        if len(predictions) < 2:
            return 1.0

        pred_values = list(predictions.values())

        try:
            # 计算相关系数的平均值
            correlations = []
            for i in range(len(pred_values)):
                for j in range(i + 1, len(pred_values)):
                    corr = np.corrcoef(pred_values[i].flatten(), pred_values[j].flatten())[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            return np.mean(correlations) if correlations else 0.5
        except:
            return 0.5

    def _get_recent_performance(self) -> Dict[str, Dict[str, float]]:
        """获取最近的性能数据"""
        performance_data = {}

        for model_id, model in self.models.items():
            performance_data[model_id] = {
                'accuracy': model.metadata.accuracy,
                'f1_score': model.metadata.f1_score,
                'mse': model.metadata.mse,
                'r2_score': model.metadata.r2_score
            }

        return performance_data

    def _update_stats(self, result: EnsemblePrediction):
        """更新统计信息"""
        self.stats['total_predictions'] += 1

        # 更新平均置信度
        current_avg = self.stats['average_confidence']
        total = self.stats['total_predictions']
        self.stats['average_confidence'] = (
            (current_avg * (total - 1) + result.confidence) / total
        )

    def evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估集成性能"""
        prediction_result = self.predict(X)
        predictions = prediction_result.final_prediction

        # 根据预测类型计算指标
        if isinstance(predictions[0], (int, bool)) or len(np.unique(y)) < 10:
            # 分类问题
            accuracy = accuracy_score(y, predictions)
            f1 = f1_score(y, predictions, average='weighted')

            self.stats['ensemble_accuracy'] = accuracy

            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'consensus_level': prediction_result.consensus_level
            }
        else:
            # 回归问题
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)

            return {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'consensus_level': prediction_result.consensus_level
            }

    def get_model_rankings(self) -> List[Dict[str, Any]]:
        """获取模型排名"""
        rankings = []

        for model_id, model in self.models.items():
            ranking_info = {
                'model_id': model_id,
                'model_name': model.model_name,
                'model_type': model.metadata.model_type.value,
                'performance_score': self._calculate_performance_score(model),
                'metadata': model.metadata.to_dict()
            }
            rankings.append(ranking_info)

        # 按性能分数排序
        rankings.sort(key=lambda x: x['performance_score'], reverse=True)

        return rankings

    def _calculate_performance_score(self, model: BaseModel) -> float:
        """计算综合性能分数"""
        if model.metadata.model_type == ModelType.CLASSIFIER:
            return (model.metadata.accuracy + model.metadata.f1_score) / 2
        else:
            return model.metadata.r2_score

class ModelEnsembleSystem:
    """模型集成系统主类"""

    def __init__(self):
        self.ensembles = {}
        self.model_registry = {}

        # 系统统计
        self.system_stats = {
            'total_ensembles': 0,
            'total_models': 0,
            'total_predictions': 0,
            'average_ensemble_accuracy': 0.0
        }

        logger.info("模型集成系统初始化完成")

    def create_ensemble(self, ensemble_id: str,
                       ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGING) -> str:
        """创建新的集成"""
        ensemble = EnsembleEngine(ensemble_method)
        self.ensembles[ensemble_id] = ensemble
        self.system_stats['total_ensembles'] += 1

        logger.info(f"创建集成: {ensemble_id} (方法: {ensemble_method.value})")
        return ensemble_id

    def add_sklearn_model(self, ensemble_id: str, model_id: str, model_name: str,
                         sklearn_model, model_type: ModelType) -> bool:
        """添加sklearn模型到集成"""
        if ensemble_id not in self.ensembles:
            logger.error(f"集成 {ensemble_id} 不存在")
            return False

        try:
            wrapped_model = SklearnModelWrapper(model_id, model_name, sklearn_model, model_type)
            self.ensembles[ensemble_id].add_model(wrapped_model)
            self.model_registry[model_id] = wrapped_model
            self.system_stats['total_models'] += 1

            return True
        except Exception as e:
            logger.error(f"添加模型失败: {e}")
            return False

    def train_ensemble(self, ensemble_id: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """训练集成中的所有模型"""
        if ensemble_id not in self.ensembles:
            raise ValueError(f"集成 {ensemble_id} 不存在")

        ensemble = self.ensembles[ensemble_id]
        training_results = {}

        for model_id, model in ensemble.models.items():
            try:
                model.fit(X, y)

                # 评估模型
                eval_results = model.evaluate(X, y)
                training_results[model_id] = {
                    'success': True,
                    'training_time': model.metadata.training_time,
                    'evaluation': eval_results
                }

                logger.info(f"模型 {model_id} 训练完成")

            except Exception as e:
                logger.error(f"模型 {model_id} 训练失败: {e}")
                training_results[model_id] = {
                    'success': False,
                    'error': str(e)
                }

        return training_results

    def predict_with_ensemble(self, ensemble_id: str, X: np.ndarray,
                            feature_names: List[str] = None) -> EnsemblePrediction:
        """使用集成进行预测"""
        if ensemble_id not in self.ensembles:
            raise ValueError(f"集成 {ensemble_id} 不存在")

        ensemble = self.ensembles[ensemble_id]
        result = ensemble.predict(X, feature_names)

        self.system_stats['total_predictions'] += 1

        return result

    def evaluate_ensemble(self, ensemble_id: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """评估集成性能"""
        if ensemble_id not in self.ensembles:
            raise ValueError(f"集成 {ensemble_id} 不存在")

        ensemble = self.ensembles[ensemble_id]
        evaluation_results = ensemble.evaluate_ensemble(X, y)

        return {
            'ensemble_id': ensemble_id,
            'evaluation_results': evaluation_results,
            'model_rankings': ensemble.get_model_rankings(),
            'ensemble_stats': ensemble.stats
        }

    def compare_ensemble_methods(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               models_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较不同集成方法的性能"""

        methods_to_test = [
            EnsembleMethod.AVERAGING,
            EnsembleMethod.WEIGHTED_AVERAGING,
            EnsembleMethod.VOTING
        ]

        comparison_results = {}

        for method in methods_to_test:
            ensemble_id = f"test_{method.value}_{int(datetime.now().timestamp())}"

            try:
                # 创建集成
                self.create_ensemble(ensemble_id, method)

                # 添加模型
                for i, model_config in enumerate(models_config):
                    model_id = f"{ensemble_id}_model_{i}"
                    self.add_sklearn_model(
                        ensemble_id, model_id, model_config['name'],
                        model_config['model'], model_config['type']
                    )

                # 训练
                self.train_ensemble(ensemble_id, X_train, y_train)

                # 评估
                eval_results = self.evaluate_ensemble(ensemble_id, X_test, y_test)

                comparison_results[method.value] = eval_results

            except Exception as e:
                logger.error(f"测试方法 {method.value} 失败: {e}")
                comparison_results[method.value] = {'error': str(e)}

            finally:
                # 清理测试集成
                if ensemble_id in self.ensembles:
                    del self.ensembles[ensemble_id]

        # 确定最佳方法
        best_method = self._determine_best_method(comparison_results)

        return {
            'comparison_results': comparison_results,
            'best_method': best_method,
            'recommendation': f"推荐使用 {best_method} 方法以获得最佳性能"
        }

    def _determine_best_method(self, results: Dict[str, Any]) -> str:
        """确定最佳集成方法"""
        best_method = None
        best_score = -float('inf')

        for method, result in results.items():
            if 'error' in result:
                continue

            eval_results = result.get('evaluation_results', {})

            # 根据可用指标计算综合分数
            if 'accuracy' in eval_results:
                score = eval_results['accuracy']
            elif 'r2_score' in eval_results:
                score = eval_results['r2_score']
            else:
                continue

            if score > best_score:
                best_score = score
                best_method = method

        return best_method or "weighted_averaging"

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        ensemble_summaries = {}

        for ensemble_id, ensemble in self.ensembles.items():
            ensemble_summaries[ensemble_id] = {
                'method': ensemble.ensemble_method.value,
                'models_count': len(ensemble.models),
                'stats': ensemble.stats,
                'recent_predictions': len(ensemble.prediction_history)
            }

        return {
            'system_stats': self.system_stats,
            'ensemble_summaries': ensemble_summaries,
            'total_ensembles': len(self.ensembles),
            'model_registry_size': len(self.model_registry)
        }

# 全局实例
_model_ensemble_system_instance = None

def get_model_ensemble_system() -> ModelEnsembleSystem:
    """获取模型集成系统实例"""
    global _model_ensemble_system_instance
    if _model_ensemble_system_instance is None:
        _model_ensemble_system_instance = ModelEnsembleSystem()
    return _model_ensemble_system_instance
