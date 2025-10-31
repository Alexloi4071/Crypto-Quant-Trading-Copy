# -*- coding: utf-8 -*-
"""
模型超參數優化器 (第3層) - 多模型集成版本
分別優化LightGBM、XGBoost、CatBoost並保存所有預測結果

阶段A修复：集成Financial-First目标函数
- 从ML主导(90%)转向金融主导(98%)
- Sharpe从8%提升到40%
- F1从40%降低到2%
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

from optuna_system.utils.io_utils import read_dataframe
from optuna_system.utils.financial_objectives import FinancialFirstObjective

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - 可選依賴
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

warnings.filterwarnings('ignore')


class ModelOptimizer:
    """模型超參數優化器 - 第3層優化（多模型集成版本）"""

    def __init__(self, data_path: str, config_path: str = "configs/",
                 symbol: str = "BTCUSDT", timeframe: str = "15m",
                 results_path: str = None):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        
        # 設置結果保存路徑
        if results_path:
            self.results_path = Path(results_path)
        else:
            self.results_path = Path("optuna_system/results") / f"{symbol}_{timeframe}"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.symbol = symbol
        self.timeframe = timeframe

        # 使用集中日誌
        self.logger = logging.getLogger(__name__)
        
        # 定義要訓練的模型（完整5模型方案）
        self.models_to_train = ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'extratrees']

    def get_available_models(self) -> List[str]:
        available_models: List[str] = []
        if 'lightgbm' in self.models_to_train:
            available_models.append('lightgbm')
        if 'xgboost' in self.models_to_train and xgb is not None:
            available_models.append('xgboost')
        if 'catboost' in self.models_to_train and CatBoostClassifier is not None:
            available_models.append('catboost')
        if 'randomforest' in self.models_to_train:
            available_models.append('randomforest')
        if 'extratrees' in self.models_to_train:
            available_models.append('extratrees')
        return available_models

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """使用與Layer2相同的數據加載方式"""
        try:
            feature_candidates = [
                self.config_path / f"selected_features_{self.timeframe}.parquet",
                self.config_path / f"selected_features_{self.timeframe}.pkl",
                self.config_path / "selected_features.parquet",
                self.config_path / "selected_features.pkl"
            ]
            feature_file = next((p for p in feature_candidates if p.exists()), None)
            if feature_file is None:
                self.logger.error("特徵文件不存在於 configs 目錄，請先執行 Layer2")
                return pd.DataFrame(), pd.Series(), None

            features_df = read_dataframe(feature_file)
            # 防洩漏：物化特徵集可能含 label，Layer3 載入後先移除
            if 'label' in features_df.columns:
                features_df = features_df.drop(columns=['label'])
                self.logger.warning("Dropped target column 'label' from features_df to prevent leakage")
            self.logger.info(f"成功加載特徵文件: {feature_file}")

            label_candidates = [
                self.config_path / f"labels_{self.timeframe}.parquet",
                self.config_path / f"labels_{self.timeframe}.pkl",
                self.config_path / "labels.parquet",
                self.config_path / "labels.pkl"
            ]
            label_file = next((p for p in label_candidates if p.exists()), None)
            if label_file is None:
                self.logger.error("標籤文件不存在於 configs 目錄，請先執行 Layer1")
                return pd.DataFrame(), pd.Series(), None

            labels_df = read_dataframe(label_file)
            labels = labels_df['label'] if 'label' in labels_df.columns else labels_df.iloc[:, 0]
            self.logger.info(f"成功加載標籤文件: {label_file}")

            cleaned_candidates = [
                self.config_path / f"cleaned_ohlcv_{self.timeframe}.parquet",
                self.config_path / f"cleaned_ohlcv_{self.timeframe}.pkl",
                Path("data/processed/cleaned") / f"{self.symbol}_{self.timeframe}" / "cleaned_ohlcv.pkl",
                Path("data/processed/cleaned") / f"{self.symbol}_{self.timeframe}" / "cleaned_ohlcv.parquet"
            ]
            cleaned_file = next((p for p in cleaned_candidates if p.exists()), None)
            returns_series = None
            if cleaned_file is not None and cleaned_file.exists():
                ohlcv_df = read_dataframe(cleaned_file)
                if 'close' in ohlcv_df.columns:
                    returns_series = ohlcv_df['close'].pct_change()
                    self.logger.info(f"成功計算收益率")

            # 對齊索引
            common_index = features_df.index.intersection(labels.index)
            if len(common_index) == 0:
                self.logger.error("特徵和標籤索引沒有交集")
                return pd.DataFrame(), pd.Series(), None

            features_df = features_df.loc[common_index]
            labels = labels.loc[common_index]
            if returns_series is not None:
                returns_series = returns_series.loc[returns_series.index.intersection(common_index)]

            self.logger.info(f"數據加載完成: {features_df.shape[0]} 樣本, {features_df.shape[1]} 特徵")
            self.logger.info(f"標籤分佈: {labels.value_counts().to_dict()}")

            return features_df, labels, returns_series

        except Exception as e:
            self.logger.error(f"數據加載失敗: {e}")
            return pd.DataFrame(), pd.Series(), None

    def suggest_model_params(self, trial: optuna.Trial, model_type: str) -> Dict:
        """
        為指定模型類型生成超參數
        
        阶段B修复：扩展参数搜索范围（针对15分钟crypto交易优化）
        - max_depth: 3-10 → 6-15 (更深的树捕捉复杂模式)
        - learning_rate: 0.03-0.1 → 0.01-0.15 (更大的探索空间)
        """

        if model_type == 'lightgbm':
            # 阶段B：扩展max_depth范围（3-10 → 6-15）
            max_depth = trial.suggest_int('max_depth', 6, 15)
            # 動態計算 num_leaves 的上下界，避免 low > high 的無效區間
            upper_num_leaves = min((2 ** max_depth) - 1, 256)
            lower_num_leaves = 16 if upper_num_leaves >= 16 else max(2, upper_num_leaves)
            num_leaves = trial.suggest_int('num_leaves', lower_num_leaves, upper_num_leaves)
            return {
                'type': 'lightgbm',
                'params': {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                    'verbosity': -1,
                    'seed': 42,
                    'num_leaves': num_leaves,
                    'max_depth': max_depth,
                    # 阶段B：扩展learning_rate范围（0.03-0.1 → 0.01-0.15）
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 400, 1200),  # 阶段B：扩展上限
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    # 阶段B：crypto 24/7交易，数据充足，扩展bagging_fraction范围
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 120),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 5.0),
                    'class_weight': 'balanced',
                    'num_threads': 0,
                    'deterministic': True
                }
            }

        elif model_type == 'xgboost' and xgb is not None:
            # 阶段B：扩展max_depth范围（3-10 → 6-15）
            max_depth = trial.suggest_int('max_depth', 6, 15)
            return {
                'type': 'xgboost',
                'params': {
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'tree_method': trial.suggest_categorical('tree_method', ['hist', 'approx']),
                    'max_depth': max_depth,
                    # 阶段B：扩展eta范围（0.03-0.1 → 0.01-0.15）
                    'eta': trial.suggest_float('eta', 0.01, 0.15, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 400, 1200),  # 阶段B：扩展上限
                    # 阶段B：crypto 24/7交易，扩展subsample范围
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'lambda': trial.suggest_float('lambda', 0.0, 5.0),
                    'alpha': trial.suggest_float('alpha', 0.0, 5.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'seed': 42
                }
            }

        elif model_type == 'catboost' and CatBoostClassifier is not None:
            # 阶段B：扩展depth范围（4-10 → 6-12）
            depth = trial.suggest_int('depth', 6, 12)
            return {
                'type': 'catboost',
                'params': {
                    'loss_function': 'MultiClass',
                    'eval_metric': 'TotalF1',
                    'iterations': trial.suggest_int('iterations', 500, 1500),
                    # 阶段B：扩展learning_rate范围（0.02-0.3 → 0.01-0.3）
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': depth,
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 6.0),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
                    'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'verbose': False,
                    'random_seed': 42
                }
            }

        elif model_type == 'randomforest':
            return {
                'type': 'randomforest',
                'params': {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1,
                    'bootstrap': True
                }
            }

        elif model_type == 'extratrees':
            return {
                'type': 'extratrees',
                'params': {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1,
                    'bootstrap': False
                }
            }
        
        else:
            raise ValueError(f"不支持的模型類型或缺少依賴: {model_type}")

    def fit_and_predict(self, model_info: Dict, X_train: pd.DataFrame, y_train: pd.Series, 
                       X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """訓練模型並生成預測"""
        model_type = model_info['type']
        params = model_info['params']

        try:
            if model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)

            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)

            elif model_type == 'catboost':
                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test).flatten()
                proba = model.predict_proba(X_test)

            elif model_type == 'randomforest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)

            elif model_type == 'extratrees':
                from sklearn.ensemble import ExtraTreesClassifier
                model = ExtraTreesClassifier(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)

            else:
                raise ValueError(f"不支持的模型類型: {model_type}")

            return preds, proba

        except Exception as e:
            self.logger.error(f"模型訓練失敗 ({model_type}): {e}")
            # 返回默認預測
            preds = np.ones(len(X_test))
            proba = np.ones((len(X_test), 3)) / 3
            return preds, proba

    def purged_walk_forward_cv(self, X: pd.DataFrame, y: pd.Series, model_info: Dict,
                               n_splits: int = 4, embargo_pct: float = 0.02,
                               purge_pct: float = 0.01,
                               returns: Optional[pd.Series] = None) -> Dict:
        """實現帶 purged & embargo 的 Walk-Forward CV"""

        total_len = len(X)
        split_size = max(1, total_len // (n_splits + 1))
        embargo = max(1, int(total_len * embargo_pct))
        purge = max(1, int(total_len * purge_pct))

        metrics = {
            'f1_weighted': [],
            'f1_macro': [],
            'accuracy': [],
            'precision_weighted': [],
            'recall_weighted': [],
            'auc_macro': [],
            'logloss': [],
            'strategy_return': [],
            'sharpe': [],
            'win_rate': []
        }

        for i in range(n_splits):
            train_end = (i + 1) * split_size
            train_end_purged = max(purge, train_end - purge)

            test_start = min(total_len, train_end + embargo)
            test_end = min(total_len, test_start + split_size)

            if test_end - test_start < 20 or train_end_purged <= purge:
                continue

            train_idx = list(range(purge, train_end_purged))
            test_idx = list(range(test_start, test_end))

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            preds, proba = self.fit_and_predict(model_info, X_train, y_train, X_test)

            metrics['f1_weighted'].append(f1_score(y_test, preds, average='weighted'))
            metrics['f1_macro'].append(f1_score(y_test, preds, average='macro'))
            metrics['accuracy'].append(accuracy_score(y_test, preds))
            metrics['precision_weighted'].append(precision_score(y_test, preds, average='weighted', zero_division=0))
            metrics['recall_weighted'].append(recall_score(y_test, preds, average='weighted', zero_division=0))
            metrics['logloss'].append(log_loss(y_test, proba))

            try:
                metrics['auc_macro'].append(roc_auc_score(y_test, proba, multi_class='ovo', average='macro'))
            except Exception:
                metrics['auc_macro'].append(0.5)

            if returns is not None:
                aligned_idx = returns.index.intersection(y_test.index)
                if len(aligned_idx) > 10:
                    positions = pd.Series(preds, index=y_test.index).reindex(aligned_idx).fillna(1) - 1
                    strategy_ret = positions * returns.reindex(aligned_idx).shift(-1)
                    strategy_ret = strategy_ret.dropna()
                    if len(strategy_ret) > 0:
                        metrics['strategy_return'].append(strategy_ret.sum())
                        std = strategy_ret.std()
                        sharpe = (strategy_ret.mean() / std * np.sqrt(252)) if std > 0 else 0
                        metrics['sharpe'].append(sharpe)
                        metrics['win_rate'].append((strategy_ret > 0).mean())

        summary = {f'{k}_mean': np.mean(v) if v else 0 for k, v in metrics.items()}
        summary.update({f'{k}_std': np.std(v) if v else 0 for k, v in metrics.items()})
        summary['folds'] = len(metrics['f1_weighted'])
        
        # ==============================
        # 阶段A修复：使用Financial-First目标函数（完整替代）
        # ✅ 按MD文件计划：删除旧的ML主导(90% ML)，使用新的金融主导(98% 金融)
        # 参考：OPTUNA_4_STAGE_FIX_PLAN.md 第62-86行
        # ==============================
        
        # 创建Financial-First目标函数（金融指标98%，ML指标2%）
        objective = FinancialFirstObjective(risk_free_rate=0.02)
        
        # 提取收益率序列和ML指标
        returns = summary.get('returns_series', pd.Series())
        ml_metrics = {
            'f1_macro': summary.get('f1_macro_mean', 0),
            'accuracy': summary.get('accuracy_mean', 0),
            'auc_macro': summary.get('auc_macro_mean', 0)
        }
        
        # 计算Financial-First综合得分（金融主导）
        summary['composite_score'] = objective.calculate_score(returns, ml_metrics)
        
        # 可选：记录组件得分（用于调试）
        if self.logger.isEnabledFor(logging.DEBUG):
            components = objective.get_component_scores(returns, ml_metrics)
            self.logger.debug(f"Financial-First组件得分:")
            self.logger.debug(f"  Sharpe组件: {components['sharpe_component']:.4f} (权重40%)")
            self.logger.debug(f"  MaxDD组件: {components['maxdd_component']:.4f} (权重20%)")
            self.logger.debug(f"  Calmar组件: {components['calmar_component']:.4f} (权重15%)")
            self.logger.debug(f"  F1组件: {components['f1_component']:.4f} (权重2%)")
            self.logger.debug(f"  金融贡献: {components['financial_contribution']:.4f} (98%)")
            self.logger.debug(f"  ML贡献: {components['ml_contribution']:.4f} (2%)")
        
        return summary

    def objective(self, trial: optuna.Trial, model_type: str) -> float:
        """
        Optuna目標函數 - 針對指定模型類型
        
        阶段C修复：动态CV参数调整（基于数据量和理论要求）
        """

        try:
            # 加載數據
            features_df, labels, returns_series = self.load_data()

            if len(features_df) == 0 or len(labels) == 0:
                return -999.0

            # ==============================
            # 阶段C：动态计算CV参数范围
            # 基于López de Prado理论和数据量
            # ==============================
            total_samples = len(features_df)
            
            # 假设label lag=17（15分钟数据的典型值）
            # 可以根据实际配置调整
            assumed_label_lag = 17
            
            # 理论最小值（以周期数计算）
            min_embargo_periods = max(int(assumed_label_lag * 1.5), 12)
            min_purge_periods = max(int(assumed_label_lag * 2.0), 24)
            
            # 转换为百分比
            min_embargo_pct = min_embargo_periods / total_samples
            min_purge_pct = min_purge_periods / total_samples
            
            # 动态范围（允许exploration，但不低于理论最小值）
            embargo_range = (
                max(0.01, min_embargo_pct),
                min(0.12, min_embargo_pct * 2.5)
            )
            purge_range = (
                max(0.005, min_purge_pct),
                min(0.10, min_purge_pct * 2.0)
            )
            
            # n_splits根据样本量调整
            if total_samples < 5000:
                n_splits_range = (3, 5)
            elif total_samples < 10000:
                n_splits_range = (4, 6)
            else:
                n_splits_range = (5, 7)
            
            model_info = self.suggest_model_params(trial, model_type)
            cv_results = self.purged_walk_forward_cv(
                features_df,
                labels,
                model_info,
                # 阶段C：使用动态范围
                n_splits=trial.suggest_int('n_cv_splits', *n_splits_range),
                embargo_pct=trial.suggest_float('embargo_pct', *embargo_range),
                purge_pct=trial.suggest_float('purge_pct', *purge_range),
                returns=returns_series
            )

            return cv_results.get('composite_score', -999.0)

        except Exception as e:
            self.logger.error(f"模型優化過程出錯 ({model_type}): {e}")
            return -999.0

    def optimize_single_model(self, model_type: str, n_trials: int = 50) -> Dict:
        """優化單個模型的超參數"""
        self.logger.info(f"開始優化 {model_type} 模型...")

        # 創建研究
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_optimization_layer3',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # 執行優化 - 使用lambda包裝objective
        study.optimize(lambda trial: self.objective(trial, model_type), n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"{model_type} 優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"{model_type} 最優參數: {best_params}")

        return {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials
        }

    def train_and_save_predictions(self, model_type: str, best_params: Dict) -> None:
        """使用最優參數訓練模型並保存預測結果"""
        self.logger.info(f"開始訓練 {model_type} 並生成預測...")

        try:
            features_df, labels, returns_series = self.load_data()
            
            if len(features_df) == 0 or len(labels) == 0:
                self.logger.error(f"{model_type}: 數據加載失敗")
                return

            # 使用最優參數重建模型信息
            fixed_trial = optuna.trial.FixedTrial(best_params)
            model_info = self.suggest_model_params(fixed_trial, model_type)

            # 使用Walk-Forward方式生成全量預測
            total_len = len(features_df)
            n_splits = 5
            split_size = total_len // (n_splits + 1)
            
            all_predictions = []
            all_probabilities = []
            all_indices = []

            for i in range(n_splits):
                train_end = (i + 1) * split_size
                test_start = train_end
                test_end = min(total_len, test_start + split_size)

                if test_end - test_start < 20:
                    continue

                train_idx = list(range(0, train_end))
                test_idx = list(range(test_start, test_end))

                X_train, X_test = features_df.iloc[train_idx], features_df.iloc[test_idx]
                y_train = labels.iloc[train_idx]

                preds, proba = self.fit_and_predict(model_info, X_train, y_train, X_test)

                all_predictions.extend(preds)
                all_probabilities.extend(proba)
                all_indices.extend(features_df.index[test_idx])

            # 保存預測結果
            predictions_df = pd.DataFrame({
                'prediction': all_predictions,
                'proba_class_0': [p[0] for p in all_probabilities],
                'proba_class_1': [p[1] for p in all_probabilities],
                'proba_class_2': [p[2] for p in all_probabilities]
            }, index=all_indices)

            # 根據模型類型確定文件名
            filename_map = {
                'lightgbm': 'lgb_predictions.parquet',
                'xgboost': 'xgb_predictions.parquet',
                'catboost': 'cat_predictions.parquet',
                'randomforest': 'rf_predictions.parquet',
                'extratrees': 'et_predictions.parquet'
            }
            
            output_file = self.results_path / filename_map[model_type]
            predictions_df.to_parquet(output_file)
            
            self.logger.info(f"✅ {model_type} 預測已保存: {output_file}")
            self.logger.info(f"   預測樣本數: {len(predictions_df)}")

        except Exception as e:
            self.logger.error(f"{model_type} 訓練和預測失敗: {e}")

    def optimize(self, n_trials: int = 50) -> Dict:
        """執行所有模型的超參數優化和預測生成"""
        self.logger.info("=" * 80)
        self.logger.info("開始多模型集成優化（第3層）...")
        self.logger.info("=" * 80)

        all_results = {}
        
        # 檢查可用的模型
        available_models = self.get_available_models()
        
        if not available_models:
            self.logger.error("沒有可用的模型！請檢查依賴安裝。")
            return {}

        self.logger.info(f"可用模型: {available_models}")

        # 為每個模型獨立優化
        for model_type in available_models:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"處理模型: {model_type.upper()}")
            self.logger.info(f"{'='*60}")
            
            # 步驟1: 優化超參數
            result = self.optimize_single_model(model_type, n_trials)
            all_results[model_type] = result
            
            # 步驟2: 訓練並保存預測
            self.train_and_save_predictions(model_type, result['best_params'])

            # 步驟3: 即時寫出累積的 model_params.json，避免中途停止無總結
            try:
                incremental_path = self.config_path / "model_params.json"
                with open(incremental_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                self.logger.info(f"📝 已更新: {incremental_path}（累積 {len(all_results)} 個模型結果）")
            except Exception as e:
                self.logger.warning(f"⚠️ 寫出 model_params.json 失敗: {e}")

        # 保存整合結果
        output_file = self.config_path / "model_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ 多模型優化完成!")
        self.logger.info(f"✅ 結果已保存至: {output_file}")
        self.logger.info(f"✅ 預測文件保存在: {self.results_path}")
        self.logger.info("=" * 80)

        # 返回統一格式的結果（保持向後兼容）
        best_model = max(all_results.items(), key=lambda x: x[1]['best_score'])
        return {
            'best_params': best_model[1]['best_params'],
            'best_score': best_model[1]['best_score'],
            'best_model_type': best_model[0],
            'all_models': all_results,
            'n_trials': n_trials
        }


def main():
    """主函數"""
    optimizer = ModelOptimizer(
        data_path='data',
        config_path='configs',
        results_path='optuna_system/results/BTCUSDT_15m'
    )
    result = optimizer.optimize(n_trials=50)
    print(f"\n最佳模型: {result.get('best_model_type', 'N/A')}")
    print(f"最佳得分: {result.get('best_score', 0):.4f}")


if __name__ == "__main__":
    main()
