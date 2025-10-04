"""
Model Optimizer Module
Specialized hyperparameter optimization for different model types
Supports LightGBM, Random Forest, XGBoost, and neural networks
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import optuna
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ML Libraries
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import os
    # 抑制TensorFlow警告和優化提示
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from sklearn.svm import SVC
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False

from config.settings import config
from src.utils.logger import setup_logger, get_optimization_logger
from src.utils.helpers import timing_decorator

logger = setup_logger(__name__)


class ModelOptimizer:
    """Specialized model hyperparameter optimizer"""

    def __init__(self):
        self.available_models = {
            'lightgbm': self._optimize_lightgbm,
            'randomforest': self._optimize_randomforest,
            'gradientboosting': self._optimize_gradientboosting,
        }

        if XGBOOST_AVAILABLE:
            self.available_models['xgboost'] = self._optimize_xgboost

        if TENSORFLOW_AVAILABLE:
            self.available_models['lstm'] = self._optimize_lstm
            self.available_models['mlp'] = self._optimize_mlp

        if SVM_AVAILABLE:
            self.available_models['svm'] = self._optimize_svm

        self.optimization_history = {}
        self.best_models = {}

    @timing_decorator
    def optimize_model(self, model_type: str, X: pd.DataFrame, y: pd.Series,
                      symbol: str = "Unknown", timeframe: str = "1h",
                      n_trials: int = 100, timeout: int = 3600,
                      cv_folds: int = 5, scoring: str = 'f1_weighted') -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model type

        Args:
            model_type: Type of model to optimize
            X: Feature DataFrame
            y: Target Series
            symbol: Trading symbol
            timeframe: Time frame
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric to optimize

        Returns:
            Optimization results including best parameters and scores
        """
        try:
            logger.info(f"Starting {model_type} optimization for {symbol}_{timeframe}")

            if model_type not in self.available_models:
                raise ValueError(f"Model type {model_type} not available. "
                               f"Available models: {list(self.available_models.keys())}")

            # Clean data
            X_clean, y_clean = self._clean_optimization_data(X, y)

            if len(X_clean) < 100:
                logger.warning(f"Insufficient data for optimization: {len(X_clean)} samples")
                return {}

            # Create optimization study
            study_name = f"{model_type}_{symbol}_{timeframe}_{int(datetime.now().timestamp())}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )

            # Setup optimization logger
            opt_logger = get_optimization_logger(symbol, timeframe)

            # Define objective function


            def objective(trial):
                return self.available_models[model_type](
                    trial, X_clean, y_clean, cv_folds, scoring, opt_logger
                )

            # Run optimization
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

            # Compile results
            results = {
                'model_type': model_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'best_params': study.best_params,
                'best_score': study.best_value,
                'best_trial': study.best_trial.number,
                'n_trials': len(study.trials),
                'optimization_time': sum(
                    [(t.datetime_complete - t.datetime_start).total_seconds()
                     for t in study.trials if t.datetime_complete and t.datetime_start]
                ),
                'study_name': study_name
            }

            # Log results
            opt_logger.log_best_params(results['best_params'], results['best_score'])
            logger.info(f"{model_type} optimization completed. Best score: {results['best_score']:.4f}")

            # Store results
            self.optimization_history[f"{symbol}_{timeframe}_{model_type}"] = results

            return results

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return {}


    def _clean_optimization_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data for optimization"""
        try:
            # Remove rows with missing targets
            valid_mask = ~y.isna()
            X_clean = X.loc[valid_mask].copy()
            y_clean = y.loc[valid_mask].copy()

            # Handle infinite values
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

            # Fill missing values
            numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())

            # Remove constant columns
            constant_cols = X_clean.columns[X_clean.nunique() <= 1].tolist()
            if constant_cols:
                X_clean = X_clean.drop(columns=constant_cols)
                logger.info(f"Removed {len(constant_cols)} constant features")

            return X_clean, y_clean

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return X, y


    def _optimize_lightgbm(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                          cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize LightGBM hyperparameters"""
        try:
            # Define hyperparameter search space
            params = {
                'objective': 'multiclass' if len(np.unique(y)) > 2 else 'binary',
                'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else None,
                'metric': 'multi_logloss' if len(np.unique(y)) > 2 else 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'verbosity': -1,
                'random_state': 42,
                'n_jobs': -1
            }

            # Remove num_class for binary classification
            if params['objective'] == 'binary':
                params.pop('num_class')

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create datasets
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val)

                # Train model
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(30),
                        lgb.log_evaluation(0)
                    ]
                )

                # Predict and evaluate
                if len(np.unique(y)) > 2:
                    y_pred = model.predict(X_val)
                    y_pred_class = np.argmax(y_pred, axis=1)
                else:
                    y_pred_prob = model.predict(X_val)
                    y_pred_class = (y_pred_prob > 0.5).astype(int)

                if scoring == 'f1_weighted':
                    score = f1_score(y_val, y_pred_class, average='weighted')
                elif scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred_class)
                elif scoring == 'precision':
                    score = precision_score(y_val, y_pred_class, average='weighted')
                elif scoring == 'recall':
                    score = recall_score(y_val, y_pred_class, average='weighted')
                else:
                    score = f1_score(y_val, y_pred_class, average='weighted')

                scores.append(score)

                # Report intermediate result for pruning
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"LightGBM trial failed: {e}")
            return 0.0


    def _optimize_randomforest(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                             cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize Random Forest hyperparameters"""
        try:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.8]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
                'random_state': 42,
                'n_jobs': -1
            }

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_val)

                if scoring == 'f1_weighted':
                    score = f1_score(y_val, y_pred, average='weighted')
                elif scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = f1_score(y_val, y_pred, average='weighted')

                scores.append(score)

                # Report for pruning
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"Random Forest trial failed: {e}")
            return 0.0


    def _optimize_xgboost(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                         cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize XGBoost hyperparameters"""
        if not XGBOOST_AVAILABLE:
            return 0.0

        try:
            params = {
                'objective': 'multi:softprob' if len(np.unique(y)) > 2 else 'binary:logistic',
                'eval_metric': 'mlogloss' if len(np.unique(y)) > 2 else 'logloss',
                'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else None,
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

            # Remove num_class for binary classification
            if params['objective'] == 'binary:logistic':
                params.pop('num_class')

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=30,
                    verbose=False
                )

                # Predict and evaluate
                y_pred = model.predict(X_val)

                if scoring == 'f1_weighted':
                    score = f1_score(y_val, y_pred, average='weighted')
                elif scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = f1_score(y_val, y_pred, average='weighted')

                scores.append(score)

                # Report for pruning
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"XGBoost trial failed: {e}")
            return 0.0


    def _optimize_gradientboosting(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                                 cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize Gradient Boosting hyperparameters"""
        try:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                'random_state': 42
            }

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                model = GradientBoostingClassifier(**params)
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_val)

                if scoring == 'f1_weighted':
                    score = f1_score(y_val, y_pred, average='weighted')
                elif scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = f1_score(y_val, y_pred, average='weighted')

                scores.append(score)

                # Report for pruning
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"Gradient Boosting trial failed: {e}")
            return 0.0


    def _optimize_lstm(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                      cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize LSTM hyperparameters"""
        if not TENSORFLOW_AVAILABLE:
            return 0.0

        try:
            # LSTM hyperparameters
            sequence_length = trial.suggest_int('sequence_length', 30, 120)
            lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128)
            lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 64)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            dense_units = trial.suggest_int('dense_units', 16, 64)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

            # Prepare data for LSTM
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Create sequences


            def create_sequences(X_data, y_data, seq_len):
                X_seq, y_seq = [], []
                for i in range(seq_len, len(X_data)):
                    X_seq.append(X_data[i-seq_len:i])
                    y_seq.append(y_data.iloc[i])
                return np.array(X_seq), np.array(y_seq)

            if len(X_scaled) < sequence_length + 100:
                return 0.0  # Not enough data for LSTM

            X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)

            # Convert labels to categorical for multi-class
            if len(np.unique(y)) > 2:
                y_seq = tf.keras.utils.to_categorical(y_seq)
                output_units = len(np.unique(y))
                loss_function = 'categorical_crossentropy'
            else:
                output_units = 1
                loss_function = 'binary_crossentropy'

            # Time series cross-validation for LSTM
            scores = []
            n_folds = min(3, cv_folds)  # Reduce folds for LSTM due to computational cost

            for fold in range(n_folds):
                # Simple train/test split for each fold
                test_size = len(X_seq) // n_folds
                test_start = fold * test_size
                test_end = test_start + test_size

                X_train = np.concatenate([X_seq[:test_start], X_seq[test_end:]])
                y_train = np.concatenate([y_seq[:test_start], y_seq[test_end:]])
                X_test = X_seq[test_start:test_end]
                y_test = y_seq[test_start:test_end]

                if len(X_train) < 50 or len(X_test) < 10:
                    continue

                # Build model
                model = Sequential([
                    LSTM(lstm_units_1, return_sequences=True,
                         input_shape=(sequence_length, X.shape[1])),
                    Dropout(dropout_rate),
                    LSTM(lstm_units_2),
                    Dropout(dropout_rate),
                    Dense(dense_units, activation='relu'),
                    Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
                ])

                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss=loss_function,
                    metrics=['accuracy']
                )

                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=20,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(patience=3)
                    ]
                )

                # Get best validation score
                score = max(history.history['val_accuracy'])
                scores.append(score)

                # Clean up
                del model
                tf.keras.backend.clear_session()

            if not scores:
                return 0.0

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except Exception as e:
            logger.debug(f"LSTM trial failed: {e}")
            return 0.0


    def _optimize_mlp(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                     cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize Multi-Layer Perceptron hyperparameters"""
        if not TENSORFLOW_AVAILABLE:
            return 0.0

        try:
            # MLP hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 5)
            first_layer_units = trial.suggest_int('first_layer_units', 32, 256)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

            # Prepare data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Convert labels
            if len(np.unique(y)) > 2:
                y_cat = tf.keras.utils.to_categorical(y)
                output_units = len(np.unique(y))
                loss_function = 'categorical_crossentropy'
            else:
                y_cat = y.values
                output_units = 1
                loss_function = 'binary_crossentropy'

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=min(3, cv_folds))
            scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_cat[train_idx], y_cat[val_idx]

                # Build model
                model = Sequential()
                model.add(Dense(first_layer_units, activation='relu', input_shape=(X.shape[1],)))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))

                # Add hidden layers with decreasing units
                units = first_layer_units
                for layer in range(n_layers - 1):
                    units = max(16, units // 2)
                    model.add(Dense(units, activation='relu'))
                    model.add(BatchNormalization())
                    model.add(Dropout(dropout_rate))

                # Output layer
                model.add(Dense(output_units,
                               activation='sigmoid' if output_units == 1 else 'softmax'))

                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss=loss_function,
                    metrics=['accuracy']
                )

                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True)
                    ]
                )

                # Get best validation score
                score = max(history.history['val_accuracy'])
                scores.append(score)

                # Clean up
                del model
                tf.keras.backend.clear_session()

            if not scores:
                return 0.0

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except Exception as e:
            logger.debug(f"MLP trial failed: {e}")
            return 0.0


    def _optimize_svm(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                     cv_folds: int, scoring: str, opt_logger) -> float:
        """Optimize SVM hyperparameters"""
        if not SVM_AVAILABLE:
            return 0.0

        try:
            # SVM hyperparameters
            C = trial.suggest_float('C', 0.01, 100, log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

            if kernel == 'poly':
                degree = trial.suggest_int('degree', 2, 5)
            else:
                degree = 3

            params = {
                'C': C,
                'kernel': kernel,
                'gamma': gamma,
                'degree': degree,
                'random_state': 42,
                'probability': True
            }

            # Scale features for SVM
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Cross-validation with limited data due to SVM computational cost
            tscv = TimeSeriesSplit(n_splits=min(3, cv_folds))
            scores = []

            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Limit training samples for performance
                if len(X_train) > 1000:
                    sample_idx = np.random.choice(len(X_train), 1000, replace=False)
                    X_train = X_train[sample_idx]
                    y_train = y_train.iloc[sample_idx]

                # Train model
                model = SVC(**params)
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_val)

                if scoring == 'f1_weighted':
                    score = f1_score(y_val, y_pred, average='weighted')
                elif scoring == 'accuracy':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = f1_score(y_val, y_pred, average='weighted')

                scores.append(score)

            if not scores:
                return 0.0

            mean_score = np.mean(scores)
            opt_logger.log_trial(trial.number, trial.params, mean_score)

            return mean_score

        except Exception as e:
            logger.debug(f"SVM trial failed: {e}")
            return 0.0


    def compare_models(self, X: pd.DataFrame, y: pd.Series,
                      models_to_compare: List[str] = None,
                      n_trials_per_model: int = 50,
                      symbol: str = "Unknown", timeframe: str = "1h") -> Dict[str, Any]:
        """Compare multiple models and return the best one"""
        try:
            if models_to_compare is None:
                models_to_compare = ['lightgbm', 'randomforest', 'gradientboosting']

            results = {}

            for model_type in models_to_compare:
                if model_type in self.available_models:
                    logger.info(f"Optimizing {model_type}...")
                    result = self.optimize_model(
                        model_type, X, y, symbol, timeframe,
                        n_trials=n_trials_per_model, timeout=1800
                    )

                    if result:
                        results[model_type] = result

            if not results:
                return {}

            # Find best model
            best_model = max(results.items(), key=lambda x: x[1]['best_score'])

            comparison_results = {
                'best_model': best_model[0],
                'best_score': best_model[1]['best_score'],
                'all_results': results,
                'comparison_date': datetime.now().isoformat()
            }

            logger.info(f"Model comparison completed. Best: {best_model[0]} "
                       f"(score: {best_model[1]['best_score']:.4f})")

            return comparison_results

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {}


    def save_optimization_results(self, symbol: str, timeframe: str, version: str,
                                results: Dict[str, Any]) -> str:
        """Save optimization results"""
        try:
            results_path = config.get_processed_path(symbol, timeframe, version)
            results_path.mkdir(parents=True, exist_ok=True)

            # Save detailed results
            results_file = results_path / 'model_optimization_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Model optimization results saved to {results_path}")
            return str(results_path)

        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
            return ""

# Convenience functions


def optimize_lightgbm_for_symbol(X: pd.DataFrame, y: pd.Series,
                                symbol: str, timeframe: str,
                                n_trials: int = 100) -> Dict[str, Any]:
    """Quick LightGBM optimization"""
    optimizer = ModelOptimizer()
    return optimizer.optimize_model('lightgbm', X, y, symbol, timeframe, n_trials)


def compare_tree_models(X: pd.DataFrame, y: pd.Series,
                       symbol: str = "Unknown", timeframe: str = "1h") -> Dict[str, Any]:
    """Compare tree-based models"""
    optimizer = ModelOptimizer()
    models = ['lightgbm', 'randomforest', 'gradientboosting']
    if XGBOOST_AVAILABLE:
        models.append('xgboost')

    return optimizer.compare_models(X, y, models, 30, symbol, timeframe)

# Usage example
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target
    y = pd.Series((X.sum(axis=1) > 0).astype(int))

    # Test model optimization
    optimizer = ModelOptimizer()

    # Optimize single model
    results = optimizer.optimize_model('lightgbm', X, y, 'TEST', '1h', n_trials=10)
    print(f"LightGBM optimization: {results['best_score']:.4f}")

    # Compare models
    comparison = optimizer.compare_models(X, y, ['lightgbm', 'randomforest'],
                                        n_trials_per_model=5)
    print(f"Best model: {comparison.get('best_model', 'None')}")

    print("Model optimization example completed!")
