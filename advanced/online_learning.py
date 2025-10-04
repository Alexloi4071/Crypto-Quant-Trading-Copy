"""
Online Learning
在线学习系统，支持流式数据的增量学习和模型动态更新
实现概念漂移检测、自适应学习率调整和模型性能监控
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
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from river import linear_model, naive_bayes, tree, drift, metrics, compose
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class LearningMode(Enum):
    """学习模式"""
    BATCH = "batch"                      # 批量学习
    INCREMENTAL = "incremental"          # 增量学习
    STREAMING = "streaming"              # 流式学习
    MINI_BATCH = "mini_batch"           # 小批量学习
    ONLINE = "online"                   # 在线学习

class DriftType(Enum):
    """概念漂移类型"""
    SUDDEN = "sudden"                   # 突然漂移
    GRADUAL = "gradual"                 # 渐进漂移
    INCREMENTAL = "incremental"         # 增量漂移
    RECURRING = "recurring"             # 循环漂移
    VIRTUAL = "virtual"                 # 虚拟漂移

class AdaptationStrategy(Enum):
    """适应策略"""
    REPLACE = "replace"                 # 替换模型
    RETRAIN = "retrain"                # 重新训练
    ENSEMBLE = "ensemble"              # 集成方法
    WEIGHTED = "weighted"              # 加权更新
    SLIDING_WINDOW = "sliding_window"  # 滑动窗口

@dataclass

class DataPoint:
    """数据点"""
    timestamp: datetime
    features: np.ndarray
    target: Optional[Union[float, int]] = None
    prediction: Optional[Union[float, int]] = None
    confidence: float = 0.0
    is_labeled: bool = False

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'features_shape': self.features.shape,
            'target': self.target,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'is_labeled': self.is_labeled
        }

@dataclass

class DriftDetectionResult:
    """漂移检测结果"""
    detection_time: datetime
    drift_type: DriftType
    drift_detected: bool

    # 漂移强度
    drift_magnitude: float
    confidence_level: float

    # 统计信息
    p_value: float
    test_statistic: float

    # 影响分析
    affected_features: List[int] = field(default_factory=list)
    performance_drop: float = 0.0

    # 建议动作
    recommended_action: AdaptationStrategy = AdaptationStrategy.RETRAIN

    def to_dict(self) -> dict:
        return {
            'detection_time': self.detection_time.isoformat(),
            'drift_type': self.drift_type.value,
            'drift_detected': self.drift_detected,
            'drift_magnitude': self.drift_magnitude,
            'confidence_level': self.confidence_level,
            'p_value': self.p_value,
            'test_statistic': self.test_statistic,
            'affected_features': self.affected_features,
            'performance_drop': self.performance_drop,
            'recommended_action': self.recommended_action.value
        }

@dataclass

class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime

    # 准确性指标
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # 回归指标
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0

    # 在线学习特有指标
    cumulative_accuracy: float = 0.0
    sliding_window_accuracy: float = 0.0
    concept_drift_count: int = 0
    adaptation_count: int = 0

    # 计算资源
    processing_time: float = 0.0
    memory_usage: float = 0.0

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'cumulative_accuracy': self.cumulative_accuracy,
            'sliding_window_accuracy': self.sliding_window_accuracy,
            'concept_drift_count': self.concept_drift_count,
            'adaptation_count': self.adaptation_count,
            'processing_time': self.processing_time,
            'memory_usage': self.memory_usage
        }

class DriftDetector:
    """概念漂移检测器基类"""

    def __init__(self, name: str):
        self.name = name
        self.detection_history = deque(maxlen=1000)
        self.is_initialized = False

    def update(self, prediction: Union[float, int], true_value: Union[float, int],
               timestamp: datetime = None) -> DriftDetectionResult:
        """更新检测器并检测漂移"""
        raise NotImplementedError

    def reset(self):
        """重置检测器"""
        self.detection_history.clear()
        self.is_initialized = False

class ADWINDriftDetector(DriftDetector):
    """ADWIN概念漂移检测器"""

    def __init__(self, delta: float = 0.002, window_size: int = 1000):
        super().__init__("ADWIN")
        self.delta = delta
        self.window_size = window_size
        self.error_window = deque(maxlen=window_size)
        self.total_samples = 0

    def update(self, prediction: Union[float, int], true_value: Union[float, int],
               timestamp: datetime = None) -> DriftDetectionResult:
        """更新ADWIN检测器"""
        timestamp = timestamp or datetime.now()

        # 计算错误
        error = 1 if prediction != true_value else 0
        self.error_window.append(error)
        self.total_samples += 1

        drift_detected = False
        drift_magnitude = 0.0
        p_value = 1.0

        if len(self.error_window) >= 10:  # 至少需要10个样本
            # 简化的ADWIN算法实现
            window_size = len(self.error_window)

            # 计算两个子窗口的错误率
            split_point = window_size // 2
            left_errors = sum(self.error_window[:split_point])
            right_errors = sum(self.error_window[split_point:])

            left_error_rate = left_errors / split_point
            right_error_rate = right_errors / (window_size - split_point)

            # 检测显著差异
            drift_magnitude = abs(left_error_rate - right_error_rate)

            if drift_magnitude > self.delta:
                drift_detected = True
                p_value = self.delta

        result = DriftDetectionResult(
            detection_time=timestamp,
            drift_type=DriftType.SUDDEN,
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude,
            confidence_level=1 - p_value,
            p_value=p_value,
            test_statistic=drift_magnitude,
            performance_drop=drift_magnitude,
            recommended_action=AdaptationStrategy.RETRAIN if drift_detected else AdaptationStrategy.WEIGHTED
        )

        self.detection_history.append(result)
        return result

class DDMDriftDetector(DriftDetector):
    """DDM (Drift Detection Method) 漂移检测器"""

    def __init__(self, alpha_warning: float = 2.0, alpha_drift: float = 3.0):
        super().__init__("DDM")
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_drift

        self.error_rate = 0.0
        self.std_error = 0.0
        self.num_samples = 0
        self.min_error_rate = float('inf')
        self.min_std_error = float('inf')

    def update(self, prediction: Union[float, int], true_value: Union[float, int],
               timestamp: datetime = None) -> DriftDetectionResult:
        """更新DDM检测器"""
        timestamp = timestamp or datetime.now()

        error = 1 if prediction != true_value else 0
        self.num_samples += 1

        # 更新错误率
        self.error_rate = ((self.num_samples - 1) * self.error_rate + error) / self.num_samples

        # 更新标准差
        if self.num_samples > 1:
            self.std_error = np.sqrt(self.error_rate * (1 - self.error_rate) / self.num_samples)

        # 更新最小值
        if self.error_rate + self.std_error < self.min_error_rate + self.min_std_error:
            self.min_error_rate = self.error_rate
            self.min_std_error = self.std_error

        # 检测漂移
        drift_detected = False
        drift_type = DriftType.GRADUAL
        recommended_action = AdaptationStrategy.WEIGHTED

        if self.num_samples > 30:  # 需要足够的样本
            current_level = self.error_rate + self.std_error
            min_level = self.min_error_rate + self.min_std_error

            if current_level > min_level + self.alpha_drift * self.min_std_error:
                drift_detected = True
                drift_type = DriftType.SUDDEN
                recommended_action = AdaptationStrategy.REPLACE
            elif current_level > min_level + self.alpha_warning * self.min_std_error:
                drift_detected = True
                drift_type = DriftType.GRADUAL
                recommended_action = AdaptationStrategy.RETRAIN

        drift_magnitude = abs(self.error_rate - self.min_error_rate)

        result = DriftDetectionResult(
            detection_time=timestamp,
            drift_type=drift_type,
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude,
            confidence_level=0.95 if drift_detected else 0.5,
            p_value=0.05 if drift_detected else 0.5,
            test_statistic=current_level - min_level if self.num_samples > 30 else 0,
            performance_drop=drift_magnitude,
            recommended_action=recommended_action
        )

        self.detection_history.append(result)
        return result

class OnlineLearner:
    """在线学习器基类"""

    def __init__(self, name: str, learning_mode: LearningMode):
        self.name = name
        self.learning_mode = learning_mode
        self.is_fitted = False

        # 性能追踪
        self.performance_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=1000)

        # 统计信息
        self.total_samples = 0
        self.correct_predictions = 0
        self.adaptation_count = 0

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'OnlineLearner':
        """增量学习"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测，返回预测结果和置信度"""
        raise NotImplementedError

    def adapt(self, drift_result: DriftDetectionResult):
        """根据漂移检测结果进行适应"""
        raise NotImplementedError

class SklearnOnlineLearner(OnlineLearner):
    """基于Sklearn的在线学习器"""

    def __init__(self, name: str, sklearn_model, learning_mode: LearningMode = LearningMode.INCREMENTAL):
        super().__init__(name, learning_mode)
        self.model = sklearn_model
        self.classes_ = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnOnlineLearner':
        """增量训练"""
        import time
        start_time = time.time()

        try:
            if hasattr(self.model, 'partial_fit'):
                if not self.is_fitted and hasattr(self, 'classes_'):
                    # 第一次训练时需要指定类别
                    unique_classes = np.unique(y)
                    self.model.partial_fit(X, y, classes=unique_classes)
                    self.classes_ = unique_classes
                else:
                    self.model.partial_fit(X, y)

                self.is_fitted = True
                self.total_samples += len(X)

                # 记录性能
                processing_time = time.time() - start_time
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    processing_time=processing_time
                )
                self.performance_history.append(metrics)

            else:
                logger.warning(f"模型 {self.name} 不支持增量学习")

        except Exception as e:
            logger.error(f"增量训练失败: {e}")

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        predictions = self.model.predict(X)

        # 计算置信度
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            confidences = np.max(probabilities, axis=1)
        elif hasattr(self.model, 'decision_function'):
            decision_scores = self.model.decision_function(X)
            confidences = np.abs(decision_scores) / (np.abs(decision_scores) + 1)
        else:
            confidences = np.ones(len(predictions)) * 0.8

        return predictions, confidences

    def adapt(self, drift_result: DriftDetectionResult):
        """适应策略"""
        if not drift_result.drift_detected:
            return

        self.adaptation_count += 1

        if drift_result.recommended_action == AdaptationStrategy.REPLACE:
            # 重置模型
            self.model = self.model.__class__(**self.model.get_params())
            self.is_fitted = False
            logger.info(f"模型 {self.name} 已重置")

        elif drift_result.recommended_action == AdaptationStrategy.RETRAIN:
            # 降低学习率或增加正则化
            if hasattr(self.model, 'learning_rate') and hasattr(self.model, 'set_params'):
                current_lr = getattr(self.model, 'learning_rate', 0.01)
                new_lr = current_lr * 0.8
                self.model.set_params(learning_rate=new_lr)
                logger.info(f"模型 {self.name} 学习率调整为 {new_lr}")

class RiverOnlineLearner(OnlineLearner):
    """基于River的在线学习器"""

    def __init__(self, name: str, river_model, learning_mode: LearningMode = LearningMode.STREAMING):
        super().__init__(name, learning_mode)
        self.model = river_model
        self.metric = metrics.Accuracy() if hasattr(river_model, 'predict_one') else metrics.MAE()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'RiverOnlineLearner':
        """增量训练"""
        import time
        start_time = time.time()

        try:
            for i in range(len(X)):
                x_dict = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}

                # 预测然后学习
                if self.is_fitted:
                    y_pred = self.model.predict_one(x_dict)
                    self.metric.update(y[i], y_pred)

                # 学习
                self.model.learn_one(x_dict, y[i])
                self.is_fitted = True

            self.total_samples += len(X)

            # 记录性能
            processing_time = time.time() - start_time
            metrics_obj = PerformanceMetrics(
                timestamp=datetime.now(),
                processing_time=processing_time,
                accuracy=self.metric.get() if hasattr(self.metric, 'get') else 0.0
            )
            self.performance_history.append(metrics_obj)

        except Exception as e:
            logger.error(f"River模型训练失败: {e}")

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")

        predictions = []
        confidences = []

        for i in range(len(X)):
            x_dict = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}

            if hasattr(self.model, 'predict_proba_one'):
                pred_proba = self.model.predict_proba_one(x_dict)
                if pred_proba:
                    pred = max(pred_proba, key=pred_proba.get)
                    conf = pred_proba[pred]
                else:
                    pred = 0
                    conf = 0.5
            else:
                pred = self.model.predict_one(x_dict)
                conf = 0.8

            predictions.append(pred)
            confidences.append(conf)

        return np.array(predictions), np.array(confidences)

    def adapt(self, drift_result: DriftDetectionResult):
        """适应策略"""
        if not drift_result.drift_detected:
            return

        self.adaptation_count += 1

        # River模型通常自动适应，这里可以调整学习率
        if hasattr(self.model, 'learning_rate'):
            if drift_result.drift_magnitude > 0.1:
                # 大幅漂移，增加学习率
                self.model.learning_rate = min(self.model.learning_rate * 1.5, 0.1)
            else:
                # 小幅漂移，稍微增加学习率
                self.model.learning_rate = min(self.model.learning_rate * 1.2, 0.05)

        logger.info(f"River模型 {self.name} 已适应概念漂移")

class OnlineLearningSystem:
    """在线学习系统"""

    def __init__(self):
        self.learners = {}
        self.drift_detectors = {}
        self.data_stream = deque(maxlen=10000)

        # 系统配置
        self.window_size = 1000
        self.drift_detection_enabled = True
        self.auto_adaptation_enabled = True

        # 统计信息
        self.system_stats = {
            'total_samples_processed': 0,
            'total_predictions_made': 0,
            'drift_detections': 0,
            'model_adaptations': 0,
            'average_processing_time': 0.0
        }

        logger.info("在线学习系统初始化完成")

    def add_learner(self, learner_id: str, learner: OnlineLearner):
        """添加学习器"""
        self.learners[learner_id] = learner
        logger.info(f"添加学习器: {learner_id} ({learner.name})")

    def add_drift_detector(self, detector_id: str, detector: DriftDetector):
        """添加漂移检测器"""
        self.drift_detectors[detector_id] = detector
        logger.info(f"添加漂移检测器: {detector_id} ({detector.name})")

    def process_data_point(self, data_point: DataPoint) -> Dict[str, Any]:
        """处理单个数据点"""
        import time
        start_time = time.time()

        results = {}

        # 添加到数据流
        self.data_stream.append(data_point)

        # 进行预测
        predictions = {}
        confidences = {}

        for learner_id, learner in self.learners.items():
            if learner.is_fitted:
                try:
                    pred, conf = learner.predict(data_point.features.reshape(1, -1))
                    predictions[learner_id] = pred[0]
                    confidences[learner_id] = conf[0]

                    data_point.prediction = pred[0]
                    data_point.confidence = conf[0]

                except Exception as e:
                    logger.error(f"预测失败 {learner_id}: {e}")

        # 漂移检测
        drift_results = {}
        if data_point.is_labeled and self.drift_detection_enabled:
            for detector_id, detector in self.drift_detectors.items():
                for learner_id, prediction in predictions.items():
                    try:
                        drift_result = detector.update(
                            prediction, data_point.target, data_point.timestamp
                        )
                        drift_results[f"{detector_id}_{learner_id}"] = drift_result

                        if drift_result.drift_detected:
                            self.system_stats['drift_detections'] += 1

                            # 自动适应
                            if self.auto_adaptation_enabled:
                                self.learners[learner_id].adapt(drift_result)
                                self.system_stats['model_adaptations'] += 1

                    except Exception as e:
                        logger.error(f"漂移检测失败 {detector_id}: {e}")

        # 增量学习
        if data_point.is_labeled:
            for learner_id, learner in self.learners.items():
                try:
                    learner.partial_fit(
                        data_point.features.reshape(1, -1),
                        np.array([data_point.target])
                    )
                except Exception as e:
                    logger.error(f"增量学习失败 {learner_id}: {e}")

        # 更新统计
        processing_time = time.time() - start_time
        self.system_stats['total_samples_processed'] += 1
        if predictions:
            self.system_stats['total_predictions_made'] += len(predictions)

        # 更新平均处理时间
        total_samples = self.system_stats['total_samples_processed']
        current_avg = self.system_stats['average_processing_time']
        self.system_stats['average_processing_time'] = (
            (current_avg * (total_samples - 1) + processing_time) / total_samples
        )

        results = {
            'data_point': data_point.to_dict(),
            'predictions': predictions,
            'confidences': confidences,
            'drift_results': {k: v.to_dict() for k, v in drift_results.items()},
            'processing_time': processing_time
        }

        return results

    def process_batch(self, X: np.ndarray, y: np.ndarray = None,
                     timestamps: List[datetime] = None) -> List[Dict[str, Any]]:
        """批量处理数据"""
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(seconds=i) for i in range(len(X))]

        results = []

        for i in range(len(X)):
            data_point = DataPoint(
                timestamp=timestamps[i],
                features=X[i],
                target=y[i] if y is not None else None,
                is_labeled=y is not None
            )

            result = self.process_data_point(data_point)
            results.append(result)

        return results

    def evaluate_learners(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """评估学习器性能"""
        evaluation_results = {}

        for learner_id, learner in self.learners.items():
            if learner.is_fitted:
                try:
                    predictions, confidences = learner.predict(X_test)

                    # 计算指标
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                    # 判断是分类还是回归
                    is_classification = len(np.unique(y_test)) < 10 and np.all(y_test == y_test.astype(int))

                    if is_classification:
                        metrics = {
                            'accuracy': accuracy_score(y_test, predictions),
                            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0),
                            'average_confidence': np.mean(confidences)
                        }
                    else:
                        metrics = {
                            'mse': mean_squared_error(y_test, predictions),
                            'mae': mean_absolute_error(y_test, predictions),
                            'r2_score': r2_score(y_test, predictions),
                            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                            'average_confidence': np.mean(confidences)
                        }

                    evaluation_results[learner_id] = metrics

                except Exception as e:
                    logger.error(f"评估学习器 {learner_id} 失败: {e}")
                    evaluation_results[learner_id] = {'error': str(e)}

        return evaluation_results

    def get_drift_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """获取漂移检测摘要"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        drift_summary = {
            'time_window_hours': hours_back,
            'detectors': {},
            'total_drifts_detected': 0,
            'drift_types_distribution': defaultdict(int),
            'adaptation_actions': defaultdict(int)
        }

        for detector_id, detector in self.drift_detectors.items():
            recent_detections = [
                d for d in detector.detection_history
                if d.detection_time >= cutoff_time
            ]

            detected_drifts = [d for d in recent_detections if d.drift_detected]

            drift_summary['detectors'][detector_id] = {
                'total_detections': len(recent_detections),
                'drifts_detected': len(detected_drifts),
                'average_magnitude': np.mean([d.drift_magnitude for d in detected_drifts]) if detected_drifts else 0,
                'recent_drifts': [d.to_dict() for d in detected_drifts[-5:]]
            }

            drift_summary['total_drifts_detected'] += len(detected_drifts)

            for drift in detected_drifts:
                drift_summary['drift_types_distribution'][drift.drift_type.value] += 1
                drift_summary['adaptation_actions'][drift.recommended_action.value] += 1

        return drift_summary

    def get_learner_performance_trends(self, learner_id: str, hours_back: int = 24) -> Dict[str, Any]:
        """获取学习器性能趋势"""
        if learner_id not in self.learners:
            return {'error': f'学习器 {learner_id} 不存在'}

        learner = self.learners[learner_id]
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        recent_performance = [
            p for p in learner.performance_history
            if p.timestamp >= cutoff_time
        ]

        if not recent_performance:
            return {'message': '没有最近的性能数据'}

        # 计算趋势
        accuracy_trend = [p.accuracy for p in recent_performance if p.accuracy > 0]
        processing_time_trend = [p.processing_time for p in recent_performance]

        return {
            'learner_id': learner_id,
            'time_window_hours': hours_back,
            'total_samples': learner.total_samples,
            'adaptation_count': learner.adaptation_count,
            'performance_trend': {
                'accuracy': {
                    'current': accuracy_trend[-1] if accuracy_trend else 0,
                    'average': np.mean(accuracy_trend) if accuracy_trend else 0,
                    'trend_direction': 'improving' if len(accuracy_trend) > 1 and accuracy_trend[-1] > accuracy_trend[0] else 'stable'
                },
                'processing_time': {
                    'current': processing_time_trend[-1] if processing_time_trend else 0,
                    'average': np.mean(processing_time_trend),
                    'trend_direction': 'faster' if len(processing_time_trend) > 1 and processing_time_trend[-1] < processing_time_trend[0] else 'stable'
                }
            },
            'recent_performance_data': [p.to_dict() for p in recent_performance[-10:]]
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        learner_summaries = {}
        for learner_id, learner in self.learners.items():
            learner_summaries[learner_id] = {
                'name': learner.name,
                'learning_mode': learner.learning_mode.value,
                'is_fitted': learner.is_fitted,
                'total_samples': learner.total_samples,
                'adaptation_count': learner.adaptation_count,
                'performance_history_size': len(learner.performance_history)
            }

        detector_summaries = {}
        for detector_id, detector in self.drift_detectors.items():
            detector_summaries[detector_id] = {
                'name': detector.name,
                'detection_history_size': len(detector.detection_history),
                'recent_drifts': len([d for d in detector.detection_history[-100:] if d.drift_detected])
            }

        return {
            'system_stats': self.system_stats,
            'learners': learner_summaries,
            'drift_detectors': detector_summaries,
            'data_stream_size': len(self.data_stream),
            'configuration': {
                'window_size': self.window_size,
                'drift_detection_enabled': self.drift_detection_enabled,
                'auto_adaptation_enabled': self.auto_adaptation_enabled
            }
        }

# 便利函数

def create_sklearn_online_learners() -> List[SklearnOnlineLearner]:
    """创建常用的Sklearn在线学习器"""
    learners = [
        SklearnOnlineLearner(
            "SGD_Classifier",
            SGDClassifier(loss='log', learning_rate='adaptive', eta0=0.01),
            LearningMode.INCREMENTAL
        ),
        SklearnOnlineLearner(
            "SGD_Regressor",
            SGDRegressor(learning_rate='adaptive', eta0=0.01),
            LearningMode.INCREMENTAL
        ),
        SklearnOnlineLearner(
            "Passive_Aggressive",
            PassiveAggressiveClassifier(),
            LearningMode.INCREMENTAL
        ),
        SklearnOnlineLearner(
            "Multinomial_NB",
            MultinomialNB(),
            LearningMode.INCREMENTAL
        )
    ]
    return learners

def create_river_online_learners() -> List[RiverOnlineLearner]:
    """创建常用的River在线学习器"""
    try:
        learners = [
            RiverOnlineLearner(
                "River_LogisticRegression",
                linear_model.LogisticRegression(),
                LearningMode.STREAMING
            ),
            RiverOnlineLearner(
                "River_LinearRegression",
                linear_model.LinearRegression(),
                LearningMode.STREAMING
            ),
            RiverOnlineLearner(
                "River_NaiveBayes",
                naive_bayes.GaussianNB(),
                LearningMode.STREAMING
            ),
            RiverOnlineLearner(
                "River_HoeffdingTree",
                tree.HoeffdingTreeClassifier(),
                LearningMode.STREAMING
            )
        ]
        return learners
    except ImportError:
        logger.warning("River库未安装，无法创建River学习器")
        return []

# 全局实例
_online_learning_system_instance = None

def get_online_learning_system() -> OnlineLearningSystem:
    """获取在线学习系统实例"""
    global _online_learning_system_instance
    if _online_learning_system_instance is None:
        _online_learning_system_instance = OnlineLearningSystem()
    return _online_learning_system_instance
