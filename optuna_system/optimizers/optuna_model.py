# -*- coding: utf-8 -*-
"""
模型超參數優化器 (第3層) - 多模型集成版本
分別優化LightGBM、XGBoost、CatBoost並保存所有預測結果
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

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """使用與Layer2相同的數據加載方式"""
        try:
            # 讀取Layer2生成的特徵
            feature_file = self.config_path / f"selected_features_{self.timeframe}.parquet"
            if not feature_file.exists():
                feature_file = self.config_path / "selected_features.parquet"
            
            if not feature_file.exists():
                self.logger.error(f"特徵文件不存在: {feature_file}")
                return pd.DataFrame(), pd.Series(), None

            features_df = pd.read_parquet(feature_file)
            self.logger.info(f"成功加載特徵文件: {feature_file}")

            # 讀取Layer1生成的標籤
            label_file = self.config_path / f"labels_{self.timeframe}.parquet"
            if not label_file.exists():
                label_file = self.config_path / "labels.parquet"
            
            if not label_file.exists():
                self.logger.error(f"標籤文件不存在: {label_file}")
                return pd.DataFrame(), pd.Series(), None

            labels_df = pd.read_parquet(label_file)
            labels = labels_df['label'] if 'label' in labels_df.columns else labels_df.iloc[:, 0]
            self.logger.info(f"成功加載標籤文件: {label_file}")

            # 讀取Layer0生成的清洗數據（用於計算收益）
            cleaned_file = self.config_path / f"cleaned_ohlcv_{self.timeframe}.parquet"
            if not cleaned_file.exists():
                # 嘗試從processed目錄讀取
                processed_dir = Path("data/processed/cleaned") / f"{self.symbol}_{self.timeframe}"
                cleaned_file = processed_dir / "cleaned_ohlcv.parquet"
            
            returns_series = None
            if cleaned_file.exists():
                ohlcv_df = pd.read_parquet(cleaned_file)
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
        """為指定模型類型生成超參數"""

        if model_type == 'lightgbm':
            max_depth = trial.suggest_int('max_depth', 3, 10)
            num_leaves = trial.suggest_int('num_leaves', 16, min(2 ** max_depth - 1, 256))
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
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 120),
                    'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 5.0),
                    'class_weight': 'balanced',
                    'num_threads': 0,
                    'deterministic': True
                }
            }

        elif model_type == 'xgboost' and xgb is not None:
            max_depth = trial.suggest_int('max_depth', 3, 10)
            return {
                'type': 'xgboost',
                'params': {
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'tree_method': trial.suggest_categorical('tree_method', ['hist', 'approx']),
                    'max_depth': max_depth,
                    'eta': trial.suggest_float('eta', 0.01, 0.2),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'lambda': trial.suggest_float('lambda', 0.0, 5.0),
                    'alpha': trial.suggest_float('alpha', 0.0, 5.0),
                    'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'seed': 42
                }
            }

        elif model_type == 'catboost' and CatBoostClassifier is not None:
            depth = trial.suggest_int('depth', 4, 10)
            return {
                'type': 'catboost',
                'params': {
                    'loss_function': 'MultiClass',
                    'eval_metric': 'TotalF1',
                    'iterations': trial.suggest_int('iterations', 500, 1500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3),
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
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
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
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
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
        
        # Sharpe比率归一化到0-1区间
        def normalize_sharpe(sharpe, min_val=-2.0, max_val=5.0):
            """将Sharpe比率归一化到0-1区间"""
            if sharpe is None or (isinstance(sharpe, float) and sharpe != sharpe):  # None或NaN
                return 0.0
            return max(0.0, min(1.0, (sharpe - min_val) / (max_val - min_val)))
        
        sharpe_mean = summary.get('sharpe_mean', 0)
        normalized_sharpe = normalize_sharpe(sharpe_mean)
        
        # 修改后的composite_score（使用归一化的Sharpe，范围0-1）
        summary['composite_score'] = (
            summary['f1_weighted_mean'] * 0.40 +      # 提高F1权重
            summary['f1_macro_mean'] * 0.25 +         # 均衡性
            summary['accuracy_mean'] * 0.10 +         # 整体准确性
            summary['auc_macro_mean'] * 0.15 +        # 排序能力
            normalized_sharpe * 0.08 +                # 归一化Sharpe
            summary['win_rate_mean'] * 0.02           # 胜率
        )
        
        # 保存原始和归一化的Sharpe供参考
        summary['sharpe_raw'] = sharpe_mean
        summary['sharpe_normalized'] = normalized_sharpe
        
        return summary

    def objective(self, trial: optuna.Trial, model_type: str) -> float:
        """Optuna目標函數 - 針對指定模型類型"""

        try:
            # 加載數據
            features_df, labels, returns_series = self.load_data()

            if len(features_df) == 0 or len(labels) == 0:
                return -999.0

            model_info = self.suggest_model_params(trial, model_type)
            cv_results = self.purged_walk_forward_cv(
                features_df,
                labels,
                model_info,
                n_splits=trial.suggest_int('n_cv_splits', 3, 5),
                embargo_pct=trial.suggest_float('embargo_pct', 0.01, 0.05),
                purge_pct=trial.suggest_float('purge_pct', 0.005, 0.03),
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
        available_models = []
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
