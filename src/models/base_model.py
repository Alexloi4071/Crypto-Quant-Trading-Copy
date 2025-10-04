"""
基礎模型類
為所有機器學習模型提供統一的接口和共同功能
支持模型版本管理、序列化、評估等核心功能
"""

import joblib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from config.settings import config
from src.utils.logger import setup_logger

warnings.filterwarnings('ignore')

logger = setup_logger(__name__)


class ModelMetrics:
    """模型評估指標類"""

    def __init__(self):
        self.metrics = {}
        self.timestamp = datetime.now()

    def add_metric(self, name: str, value: float, description: str = ""):
        """添加評估指標"""
        self.metrics[name] = {
            'value': value,
            'description': description,
            'timestamp': self.timestamp
        }

    def get_metric(self, name: str) -> Optional[float]:
        """獲取指標值"""
        return self.metrics.get(name, {}).get('value')

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }

    def __str__(self) -> str:
        """字符串表示"""
        lines = []
        for name, data in self.metrics.items():
            lines.append(f"{name}: {data['value']:.4f}")
        return "\n".join(lines)


class BaseModel(ABC):
    """機器學習模型基類"""

    def __init__(self, symbol: str, timeframe: str, version: str = "1.0.0", **kwargs):
        """
        初始化基礎模型

        Args:
            symbol: 交易品種
            timeframe: 時間框架
            version: 模型版本
            **kwargs: 其他參數
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.version = version
        self.model_name = self.__class__.__name__

        # 模型狀態
        self.is_fitted = False
        self.feature_names = []
        self.target_name = ""
        self.model_type = "unknown"  # classification, regression

        # 模型參數
        self.hyperparameters = kwargs
        self.best_params = {}

        # 評估結果
        self.training_metrics = ModelMetrics()
        self.validation_metrics = ModelMetrics()
        self.test_metrics = ModelMetrics()

        # 元數據
        self.metadata = {
            'created_at': datetime.now(),
            'trained_at': None,
            'data_shape': None,
            'training_time': None,
            'feature_count': 0,
            'model_size': 0
        }

        # 模型對象
        self.model = None

        logger.info(f"初始化 {self.model_name} 模型: {symbol}_{timeframe}_v{version}")

    @abstractmethod
    def _create_model(self) -> Any:
        """創建模型實例（由子類實現）"""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'BaseModel':
        """訓練模型（由子類實現）"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """預測（由子類實現）"""
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """預測概率（分類模型需實現）"""
        if self.model_type != "classification":
            raise NotImplementedError("預測概率僅適用於分類模型")
        return np.array([])

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """獲取特徵重要性（如果模型支持）"""
        return None

    def evaluate(self, X: pd.DataFrame, y: pd.Series, dataset_type: str = "test") -> ModelMetrics:
        """評估模型性能"""
        try:
            logger.debug(f"評估 {dataset_type} 集性能")

            if not self.is_fitted:
                raise ValueError("模型尚未訓練，無法評估")

            # 預測
            y_pred = self.predict(X)

            # 創建評估指標對象
            if dataset_type == "training":
                metrics = self.training_metrics
            elif dataset_type == "validation":
                metrics = self.validation_metrics
            else:
                metrics = self.test_metrics

            # 根據模型類型計算指標
            if self.model_type == "classification":
                metrics = self._evaluate_classification(y, y_pred, X, metrics)
            elif self.model_type == "regression":
                metrics = self._evaluate_regression(y, y_pred, metrics)

            logger.debug(f"{dataset_type} 集評估完成")
            return metrics

        except Exception as e:
            logger.error(f"模型評估失敗: {e}")
            return ModelMetrics()

    def _evaluate_classification(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X: pd.DataFrame,
        metrics: ModelMetrics,
    ) -> ModelMetrics:
        """評估分類模型"""
        try:
            # 基礎分類指標
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            metrics.add_metric('accuracy', accuracy, '準確率')
            metrics.add_metric('precision', precision, '精確率')
            metrics.add_metric('recall', recall, '召回率')
            metrics.add_metric('f1_score', f1, 'F1分數')

            # AUC指標 (如果支持概率預測)
            try:
                y_proba = self.predict_proba(X)
                if y_proba.size > 0:
                    if len(np.unique(y_true)) == 2:  # 二分類
                        auc = roc_auc_score(y_true, y_proba[:, 1])
                    else:  # 多分類
                        auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                    metrics.add_metric('auc', auc, 'AUC分數')
            except Exception:
                pass

            # 交易相關指標
            if len(np.unique(y_true)) <= 3:  # 假設是交易信號分類 (-1, 0, 1)
                signal_accuracy = self._calculate_signal_accuracy(y_true, y_pred)
                metrics.add_metric('signal_accuracy', signal_accuracy, '信號準確率')

            return metrics

        except Exception as e:
            logger.error(f"分類評估失敗: {e}")
            return metrics

    def _evaluate_regression(
        self, y_true: pd.Series, y_pred: np.ndarray, metrics: ModelMetrics
    ) -> ModelMetrics:
        """評估回歸模型"""
        try:
            # 基礎回歸指標
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics.add_metric('mse', mse, '均方誤差')
            metrics.add_metric('rmse', rmse, '均方根誤差')
            metrics.add_metric('mae', mae, '平均絕對誤差')
            metrics.add_metric('r2_score', r2, 'R²分數')

            # 相對誤差
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            metrics.add_metric('mape', mape, '平均絕對百分比誤差')

            # 方向準確率 (回歸預測價格方向的準確率)
            if len(y_true) > 1:
                true_direction = np.sign(y_true.diff().fillna(0))
                pred_direction = np.sign(pd.Series(y_pred).diff().fillna(0))
                direction_accuracy = (true_direction == pred_direction).mean()
                metrics.add_metric('direction_accuracy', direction_accuracy, '方向預測準確率')

            return metrics

        except Exception as e:
            logger.error(f"回歸評估失敗: {e}")
            return metrics

    def _calculate_signal_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """計算交易信號準確率"""
        try:
            # 只考慮非中性信號的準確率
            non_neutral_mask = (y_true != 0) & (y_pred != 0)
            if non_neutral_mask.sum() == 0:
                return 0.0

            non_neutral_accuracy = (y_true[non_neutral_mask] == y_pred[non_neutral_mask]).mean()
            return non_neutral_accuracy

        except Exception as e:
            logger.error(f"信號準確率計算失敗: {e}")
            return 0.0

    def save_model(self, directory: Optional[Path] = None) -> Path:
        """保存模型"""
        try:
            if directory is None:
                directory = config.get_model_path(self.symbol, self.timeframe, self.version)

            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

            # 模型文件路徑
            model_file = directory / (
                f"{self.model_name}_{self.symbol}_{self.timeframe}_v{self.version}.joblib"
            )
            metadata_file = directory / (
                f"{self.model_name}_{self.symbol}_{self.timeframe}_v{self.version}_metadata.json"
            )

            # 保存模型對象
            if self.model is not None:
                joblib.dump(self.model, model_file)
                self.metadata['model_size'] = model_file.stat().st_size

            # 保存元數據
            metadata = self._prepare_metadata_for_save()
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"模型已保存到: {model_file}")
            return model_file

        except Exception as e:
            logger.error(f"保存模型失敗: {e}")
            raise

    def load_model(self, model_path: Path) -> 'BaseModel':
        """加載模型"""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            # 加載模型對象
            self.model = joblib.load(model_path)
            self.is_fitted = True

            # 嘗試加載元數據
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    loaded_metadata = json.load(f)
                self._load_metadata_from_dict(loaded_metadata)

            logger.info(f"模型已從 {model_path} 加載")
            return self

        except Exception as e:
            logger.error(f"加載模型失敗: {e}")
            raise

    def _prepare_metadata_for_save(self) -> Dict:
        """準備保存的元數據"""
        return {
            'model_info': {
                'name': self.model_name,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'version': self.version,
                'model_type': self.model_type
            },
            'training_info': self.metadata,
            'hyperparameters': self.hyperparameters,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'metrics': {
                'training': self.training_metrics.to_dict(),
                'validation': self.validation_metrics.to_dict(),
                'test': self.test_metrics.to_dict()
            }
        }

    def _load_metadata_from_dict(self, metadata: Dict):
        """從字典加載元數據"""
        try:
            model_info = metadata.get('model_info', {})
            self.symbol = model_info.get('symbol', self.symbol)
            self.timeframe = model_info.get('timeframe', self.timeframe)
            self.version = model_info.get('version', self.version)
            self.model_type = model_info.get('model_type', self.model_type)

            self.metadata.update(metadata.get('training_info', {}))
            self.hyperparameters = metadata.get('hyperparameters', {})
            self.best_params = metadata.get('best_params', {})
            self.feature_names = metadata.get('feature_names', [])
            self.target_name = metadata.get('target_name', '')

            # 加載評估指標
            metrics_data = metadata.get('metrics', {})
            if 'training' in metrics_data:
                self.training_metrics.metrics = metrics_data['training'].get('metrics', {})
            if 'validation' in metrics_data:
                self.validation_metrics.metrics = metrics_data['validation'].get('metrics', {})
            if 'test' in metrics_data:
                self.test_metrics.metrics = metrics_data['test'].get('metrics', {})

        except Exception as e:
            logger.warning(f"部分元數據加載失敗: {e}")

    def get_model_summary(self) -> Dict:
        """獲取模型摘要信息"""
        return {
            'model_name': self.model_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'version': self.version,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names),
            'training_metrics': {
                name: data['value'] for name, data in self.training_metrics.metrics.items()
            },
            'validation_metrics': {
                name: data['value'] for name, data in self.validation_metrics.metrics.items()
            },
            'test_metrics': {
                name: data['value'] for name, data in self.test_metrics.metrics.items()
            },
            'metadata': self.metadata
        }

    def compare_with_baseline(self, baseline_accuracy: float = 0.5) -> Dict:
        """與基線模型比較"""
        results = {}

        for dataset_type, metrics in [
            ('training', self.training_metrics),
            ('validation', self.validation_metrics),
            ('test', self.test_metrics)
        ]:
            accuracy = metrics.get_metric('accuracy')
            if accuracy is not None:
                improvement = accuracy - baseline_accuracy
                results[dataset_type] = {
                    'accuracy': accuracy,
                    'baseline': baseline_accuracy,
                    'improvement': improvement,
                    'improvement_pct': (improvement / baseline_accuracy) * 100
                }

        return results

    def validate_input_data(self, X: pd.DataFrame, y: pd.Series = None) -> bool:
        """驗證輸入數據"""
        try:
            # 檢查數據類型
            if not isinstance(X, pd.DataFrame):
                logger.error("輸入特徵必須是DataFrame")
                return False

            if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
                logger.error("目標變量必須是Series或ndarray")
                return False

            # 檢查數據形狀
            if X.empty:
                logger.error("輸入數據為空")
                return False

            # 檢查特徵一致性 (如果模型已訓練)
            if self.is_fitted and self.feature_names:
                missing_features = set(self.feature_names) - set(X.columns)
                if missing_features:
                    logger.error(f"缺失特徵: {missing_features}")
                    return False

                extra_features = set(X.columns) - set(self.feature_names)
                if extra_features:
                    logger.warning(f"額外特徵將被忽略: {extra_features}")

            # 檢查缺失值
            if X.isnull().any().any():
                null_features = X.columns[X.isnull().any()].tolist()
                logger.warning(f"特徵包含缺失值: {null_features}")

            # 檢查無限值
            if np.isinf(X.values).any():
                logger.warning("特徵包含無限值")

            return True

        except Exception as e:
            logger.error(f"數據驗證失敗: {e}")
            return False

    def get_prediction_confidence(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """獲取預測信心度（如果模型支持）"""
        try:
            if self.model_type == "classification":
                probas = self.predict_proba(X)
                if probas.size > 0:
                    # 返回最高概率作為信心度
                    return np.max(probas, axis=1)

            return None

        except Exception as e:
            logger.error(f"計算預測信心度失敗: {e}")
            return None

    def feature_importance_analysis(self) -> Optional[pd.DataFrame]:
        """特徵重要性分析"""
        try:
            importance = self.get_feature_importance()
            if importance is None:
                return None

            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in importance.items()
            ]).sort_values('importance', ascending=False)

            return importance_df

        except Exception as e:
            logger.error(f"特徵重要性分析失敗: {e}")
            return None

    def __str__(self) -> str:
        """字符串表示"""
        status = "已訓練" if self.is_fitted else "未訓練"
        return f"{self.model_name}({self.symbol}_{self.timeframe}_v{self.version}) - {status}"

    def __repr__(self) -> str:
        """詳細字符串表示"""
        return (f"{self.model_name}("
                f"symbol='{self.symbol}', "
                f"timeframe='{self.timeframe}', "
                f"version='{self.version}', "
                f"fitted={self.is_fitted}, "
                f"features={len(self.feature_names)})")


class ModelFactory:
    """模型工廠類"""

    _model_registry = {}

    @classmethod
    def register_model(cls, model_name: str, model_class: type):
        """註冊模型類"""
        cls._model_registry[model_name.lower()] = model_class
        logger.info(f"註冊模型: {model_name}")

    @classmethod
    def create_model(cls, model_name: str, symbol: str, timeframe: str, version: str = "1.0.0", **kwargs) -> BaseModel:
        """創建模型實例"""
        model_name_lower = model_name.lower()

        if model_name_lower not in cls._model_registry:
            available_models = list(cls._model_registry.keys())
            raise ValueError(f"未知模型 '{model_name}'，可用模型: {available_models}")

        model_class = cls._model_registry[model_name_lower]
        return model_class(symbol=symbol, timeframe=timeframe, version=version, **kwargs)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """獲取可用模型列表"""
        return list(cls._model_registry.keys())


class ModelValidator:
    """模型驗證器"""

    def __init__(self, min_accuracy: float = 0.6, min_samples: int = 100):
        self.min_accuracy = min_accuracy
        self.min_samples = min_samples

    def validate_model_performance(self, model: BaseModel) -> Dict[str, bool]:
        """驗證模型性能"""
        results = {
            'training_performance': False,
            'validation_performance': False,
            'overfitting_check': True,
            'overall': False
        }

        try:
            # 檢查訓練性能
            train_acc = model.training_metrics.get_metric('accuracy')
            if train_acc and train_acc >= self.min_accuracy:
                results['training_performance'] = True

            # 檢查驗證性能
            val_acc = model.validation_metrics.get_metric('accuracy')
            if val_acc and val_acc >= self.min_accuracy:
                results['validation_performance'] = True

            # 檢查過擬合
            if train_acc and val_acc:
                if train_acc - val_acc > 0.1:  # 如果訓練準確率比驗證準確率高10%以上
                    results['overfitting_check'] = False

            # 總體評估
            results['overall'] = (
                results['training_performance']
                and results['validation_performance']
                and results['overfitting_check']
            )

            return results

        except Exception as e:
            logger.error(f"模型性能驗證失敗: {e}")
            return results
