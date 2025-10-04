"""
Model Training and Management Module
Comprehensive ML model training, validation, and management system
Supports multiple model types with hyperparameter optimization
"""

import pandas as pd
import numpy as np
import joblib
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings

# ML libraries
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.version_manager import VersionManager
from src.models.base_model import BaseModel  # Import BaseModel from base_model.py
from src.models.random_forest_model import RandomForestModel

warnings.filterwarnings('ignore')

logger = setup_logger(__name__)


class LightGBMModel(BaseModel):
    """LightGBM model implementation"""

    def __init__(self, symbol: str, timeframe: str, version: str, **params):
        super().__init__(symbol, timeframe, version, **params)
        self.model_name = "LightGBM"
        self.model_type = "classification"  # Set model type

        # Initialize attributes that BaseModel doesn't have
        self.scaler = None
        self.label_encoder = None

        # Default parameters
        self.default_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        # Update with custom parameters
        self.params = {**self.default_params, **params}
        self.training_history = {}
        self.performance_metrics = {}

    def _create_model(self):
        """Create LightGBM model instance (required by BaseModel)"""
        # LightGBM doesn't need a pre-created model instance
        # The model is created during training
        return None

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_data: Tuple = None, **kwargs):
        """Train LightGBM model"""
        try:
            logger.info(f"Training LightGBM model for {self.symbol}_{self.timeframe}_{self.version}")

            # Store feature names
            self.feature_names = X.columns.tolist()

            # Encode labels to ensure they are 0, 1, 2
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_scaled, label=y_encoded)

            valid_data = None
            if validation_data:
                X_val, y_val = validation_data
                y_val_encoded = self.label_encoder.transform(y_val)
                X_val_scaled = pd.DataFrame(
                    self.scaler.transform(X_val),
                    columns=X_val.columns,
                    index=X_val.index
                )
                valid_data = lgb.Dataset(X_val_scaled, label=y_val_encoded)

            # Training parameters
            train_params = self.params.copy()
            train_params.update(kwargs)

            # Train model
            self.model = lgb.train(
                train_params,
                train_data,
                valid_sets=[valid_data] if valid_data else None,
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(period=100)
                ] if valid_data else None
            )

            # Store training history
            self.training_history = {
                'training_samples': len(X),
                'features_count': len(self.feature_names),
                'training_date': datetime.now().isoformat(),
                'model_params': self.params
            }

            logger.info("LightGBM model training completed")
            return self

        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")

            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[self.feature_names]),
                columns=self.feature_names,
                index=X.index
            )

            # Get predictions
            predictions = self.model.predict(X_scaled)

            # Convert to class predictions
            class_predictions = np.argmax(predictions, axis=1)

            # Decode labels back to original values
            return self.label_encoder.inverse_transform(class_predictions)

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")

            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.transform(X[self.feature_names]),
                columns=self.feature_names,
                index=X.index
            )

            return self.model.predict(X_scaled)

        except Exception as e:
            logger.error(f"Error predicting probabilities: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            return {}

        importance_scores = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_names, importance_scores))

    def save_model(self) -> str:
        """Save model to disk"""
        try:
            model_path = config.get_model_path(self.symbol, self.timeframe, self.version)
            model_path.mkdir(parents=True, exist_ok=True)

            # Save LightGBM model
            model_file = model_path / 'lgb_model.txt'
            self.model.save_model(str(model_file))

            # Save preprocessing components
            joblib.dump(self.scaler, model_path / 'scaler.pkl')
            joblib.dump(self.label_encoder, model_path / 'label_encoder.pkl')

            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'version': self.version,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'model_params': self.params
            }

            with open(model_path / 'model_config.yaml', 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)

            # Save feature importance
            feature_importance = self.get_feature_importance()
            with open(model_path / 'feature_importance.json', 'w') as f:
                json.dump(feature_importance, f, indent=2)

            logger.info(f"Model saved to {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: str):
        """Load model from disk"""
        try:
            model_path = Path(model_path)

            # Load LightGBM model
            self.model = lgb.Booster(model_file=str(model_path / 'lgb_model.txt'))

            # Load preprocessing components
            self.scaler = joblib.load(model_path / 'scaler.pkl')
            self.label_encoder = joblib.load(model_path / 'label_encoder.pkl')

            # Load metadata
            with open(model_path / 'model_config.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
                self.feature_names = metadata['feature_names']
                self.training_history = metadata['training_history']
                self.performance_metrics = metadata['performance_metrics']
                self.params = metadata['model_params']

            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


class ModelManager:
    """Model management and orchestration"""

    def __init__(self):
        self.version_manager = VersionManager()
        # Use consistent keys matching CLI (e.g., 'random_forest')
        self.available_models = {
            'lightgbm': LightGBMModel,
            'random_forest': RandomForestModel,
        }

    def create_model(self, model_type: str, symbol: str, timeframe: str,
                     version: str = None, use_optimal_params: bool = True, **params) -> BaseModel:
        """Create a new model instance"""
        if model_type.lower() not in self.available_models:
            raise ValueError(f"Model type {model_type} not available")

        if self.available_models[model_type.lower()] is None:
            raise ValueError(f"Model type {model_type} requires additional dependencies")

        if version is None:
            version = self.version_manager.get_next_version(symbol, timeframe)

        # 自動讀取優化參數
        if use_optimal_params and not params:
            optimal_params = self._load_optimal_params(symbol, timeframe)
            if optimal_params:
                params.update(optimal_params)
                logger.info(f"✅ 已加載優化參數: {len(optimal_params)} 個參數")
            else:
                logger.warning(f"⚠️ 未找到優化參數，使用默認參數")

        model_class = self.available_models[model_type.lower()]
        return model_class(symbol, timeframe, version, **params)

    def _load_optimal_params(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """加載優化參數文件"""
        try:
            from pathlib import Path
            import json
            
            # 構建參數文件路徑
            params_file = Path(f"results/optimal_params/{symbol}/{timeframe}/model/model_latest.json")
            
            if not params_file.exists():
                logger.warning(f"優化參數文件不存在: {params_file}")
                return {}
            
            # 讀取參數文件
            with open(params_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            optimal_params = data.get('best_params', {})
            logger.info(f"📊 從 {params_file} 加載了 {len(optimal_params)} 個優化參數")
            
            return optimal_params
            
        except Exception as e:
            logger.error(f"讀取優化參數失敗: {e}")
            return {}

    def train_model(self, model: BaseModel, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> BaseModel:
        """Train a model with validation"""
        try:
            logger.info(f"Starting training for {model.model_name} model")

            # Handle class imbalance with SMOTE
            if len(np.unique(y_train)) > 1:
                smote = SMOTE(random_state=42, k_neighbors=min(5, len(y_train)//10))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

                # Convert back to DataFrame/Series
                X_train_balanced = pd.DataFrame(
                    X_train_balanced,
                    columns=X_train.columns
                )
                y_train_balanced = pd.Series(y_train_balanced)

                logger.info(f"Applied SMOTE: {len(X_train)} -> {len(X_train_balanced)} samples")
            else:
                X_train_balanced, y_train_balanced = X_train, y_train

            # Train model
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            model.fit(X_train_balanced, y_train_balanced, validation_data=validation_data)

            # Evaluate model
            if X_val is not None and y_val is not None:
                performance = self.evaluate_model(model, X_val, y_val)
                model.performance_metrics = performance

            logger.info("Model training completed successfully")
            return model

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def evaluate_model(self, model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Classification metrics
            report = classification_report(y_test, y_pred, output_dict=True)

            # ROC AUC (multi-class)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except Exception:
                roc_auc = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            performance = {
                'accuracy': report['accuracy'],
                'macro_avg_precision': report['macro avg']['precision'],
                'macro_avg_recall': report['macro avg']['recall'],
                'macro_avg_f1_score': report['macro avg']['f1-score'],
                'weighted_avg_precision': report['weighted avg']['precision'],
                'weighted_avg_recall': report['weighted avg']['recall'],
                'weighted_avg_f1_score': report['weighted avg']['f1-score'],
                'roc_auc_ovr': roc_auc,
                'confusion_matrix': cm.tolist(),
                'class_report': report
            }

            logger.info(f"Model evaluation completed. Accuracy: {performance['accuracy']:.4f}")
            return performance

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    def cross_validate_model(self, model_type: str, X: pd.DataFrame, y: pd.Series,
                             symbol: str, timeframe: str, cv_folds: int = 5, **params) -> Dict:
        """Perform time series cross validation"""
        try:
            logger.info(f"Starting cross validation for {model_type} model")

            # Time series split
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = []
            fold_performances = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"Training fold {fold + 1}/{cv_folds}")

                # Split data
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                # Create and train model
                temp_version = f"CV_fold_{fold}"
                temp_model = self.create_model(model_type, symbol, timeframe, temp_version, **params)
                temp_model = self.train_model(temp_model, X_train_fold, y_train_fold, X_val_fold, y_val_fold)

                # Evaluate fold
                fold_performance = temp_model.performance_metrics
                fold_performances.append(fold_performance)
                cv_scores.append(fold_performance.get('accuracy', 0.0))

            # Aggregate results
            cv_results = {
                'mean_accuracy': np.mean(cv_scores),
                'std_accuracy': np.std(cv_scores),
                'cv_scores': cv_scores,
                'fold_performances': fold_performances
            }

            logger.info(
                f"Cross validation completed. Mean accuracy: {cv_results['mean_accuracy']:.4f} "
                f"± {cv_results['std_accuracy']:.4f}"
            )
            return cv_results

        except Exception as e:
            logger.error(f"Error in cross validation: {e}")
            return {}

    def load_model(self, model_path: str) -> BaseModel:
        """Load a trained model from disk"""
        try:
            model_path = Path(model_path)

            # Load metadata to determine model type
            with open(model_path / 'model_config.yaml', 'r') as f:
                metadata = yaml.safe_load(f)

            raw_name = str(metadata.get('model_name', '')).lower()
            # Normalize aliases to internal keys
            alias_map = {
                'lightgbm': 'lightgbm',
                'lightgbmmodel': 'lightgbm',
                'random_forest': 'random_forest',
                'randomforest': 'random_forest',
                'randomforestmodel': 'random_forest',
                'lstm': 'lstm',
                'lstmmodel': 'lstm',
            }
            model_key = alias_map.get(raw_name, raw_name)

            if model_key == 'lstm':
                # Lazy import to avoid TF overhead when unused
                from src.models.lstm_model import LSTMModel  # type: ignore
                model_class = LSTMModel
            else:
                if model_key not in self.available_models:
                    raise ValueError(f"Unknown model type: {raw_name}")
                model_class = self.available_models[model_key]

            # Create model instance
            model = model_class(
                metadata['symbol'],
                metadata['timeframe'],
                metadata['version']
            )

            # Load model
            model.load_model(str(model_path))

            logger.info(f"Loaded {model_key} model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

# Usage example functions


def train_lightgbm_model(
    symbol: str,
    timeframe: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    **params
) -> LightGBMModel:
    """Train a LightGBM model"""
    manager = ModelManager()
    model = manager.create_model('lightgbm', symbol, timeframe, **params)
    return manager.train_model(model, X_train, y_train, X_val, y_val)


def load_trained_model(model_path: str) -> BaseModel:
    """Load a previously trained model"""
    manager = ModelManager()
    return manager.load_model(model_path)
