# -*- coding: utf-8 -*-
"""
多模型融合權重優化器
優化LightGBM、XGBoost、CatBoost等模型的融合權重
"""
import optuna
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

import os
from pathlib import Path
workspace_root = Path(os.getcwd())  # Use current workspace


class EnsembleOptimizer:
    """多模型融合權重參數優化器"""

    def __init__(self, data_path: str, config_path: str = "configs/", model_path: str = None):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.model_path = Path(model_path) if model_path else None
        self.config_path.mkdir(exist_ok=True)

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

        # 支持的模型類型（完整5模型方案）
        self.supported_models = ['lgb', 'xgb', 'catboost', 'rf', 'et']
        self.weight_update_freq = 252  # 1 week for 15m

    def load_real_predictions(self):
        """Load real model predictions from Layer3"""
        lgb_file = workspace_root / 'optuna_system' / 'results' / self.version / 'lgb_predictions.parquet'
        xgb_file = workspace_root / 'optuna_system' / 'results' / self.version / 'xgb_predictions.parquet'
        cat_file = workspace_root / 'optuna_system' / 'results' / self.version / 'cat_predictions.parquet'
        rf_file = workspace_root / 'optuna_system' / 'results' / self.version / 'rf_predictions.parquet'
        et_file = workspace_root / 'optuna_system' / 'results' / self.version / 'et_predictions.parquet'
        
        predictions = {}
        for model, file in [('lgb', lgb_file), ('xgb', xgb_file), ('catboost', cat_file), ('rf', rf_file), ('et', et_file)]:
            if file.exists():
                predictions[model] = pd.read_parquet(file)
                self.logger.info(f'✅ Loaded {model} predictions: {file}')
            else:
                raise FileNotFoundError(f'❌ Missing {model} predictions file: {file}. Run Layer3 first.')
        
        # Load labels from Layer1
        labels_file = workspace_root / 'optuna_system' / 'configs' / 'labels.parquet'
        if labels_file.exists():
            self.labels = pd.read_parquet(labels_file)
            self.logger.info(f'✅ Loaded labels: {labels_file}')
        else:
            raise FileNotFoundError(f'❌ Missing labels file: {labels_file}. Run Layer1 first.')
        
        return predictions

    def load_true_labels(self) -> np.ndarray:
        """加載真實標籤"""
        try:
            # 嘗試多個可能的標籤文件
            label_files = [
                self.data_path / "true_labels.csv",
                self.data_path / "labels.csv",
                self.data_path / "targets.csv",
                self.data_path / "y_true.csv"
            ]

            for label_file in label_files:
                if label_file.exists():
                    df = pd.read_csv(label_file)
                    if 'label' in df.columns:
                        return df['label'].values
                    elif 'target' in df.columns:
                        return df['target'].values
                    else:
                        return df.iloc[:, 0].values

            # 如果沒有找到標籤文件，生成模擬標籤
            self.logger.warning("未找到標籤文件，生成模擬標籤")
            np.random.seed(42)
            return np.random.choice([0, 1, 2], size=1000, p=[0.3, 0.4, 0.3])

        except Exception as e:
            self.logger.error(f"載入真實標籤失敗: {e}")
            return np.array([])

    def calculate_ensemble_predictions(self, predictions: Dict[str, np.ndarray],
                                     weights: Dict[str, float]) -> np.ndarray:
        """計算融合預測結果"""

        # 確保權重歸一化
        total_weight = sum(weights.values())
        if total_weight == 0:
            return np.zeros(len(list(predictions.values())[0]))

        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # 計算加權平均
        ensemble_pred = np.zeros(len(list(predictions.values())[0]))

        for model_name, pred in predictions.items():
            if model_name in normalized_weights:
                ensemble_pred += normalized_weights[model_name] * pred

        return ensemble_pred

    def calculate_diversity_score(self, predictions: Dict[str, np.ndarray]) -> float:
        """計算模型多樣性得分"""
        if len(predictions) < 2:
            return 0.0

        model_names = list(predictions.keys())
        correlations = []

        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                pred1 = predictions[model_names[i]]
                pred2 = predictions[model_names[j]]

                # 計算相關係數
                corr = np.corrcoef(pred1, pred2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

        if not correlations:
            return 0.0

        # 多樣性 = 1 - 平均相關係數
        diversity = 1.0 - np.mean(correlations)
        return max(0.0, diversity)

    def evaluate_ensemble(self, predictions: Dict[str, np.ndarray],
                         weights: Dict[str, float],
                         true_labels: np.ndarray) -> Dict[str, float]:
        """評估融合模型性能"""

        try:
            ensemble_pred = self.calculate_ensemble_predictions(predictions, weights)

            # 轉換為分類預測（假設三分類）
            if len(np.unique(true_labels)) <= 2:
                # 二分類
                binary_pred = (ensemble_pred > 0.5).astype(int)
                accuracy = accuracy_score(true_labels, binary_pred)
                f1 = f1_score(true_labels, binary_pred, average='weighted')

                try:
                    auc = roc_auc_score(true_labels, ensemble_pred)
                except Exception:
                    auc = 0.5
            else:
                # 多分類
                # 假設輸出是概率，轉換為類別
                if len(ensemble_pred.shape) == 1:
                    # 單維輸出，轉換為分類
                    class_pred = np.zeros_like(ensemble_pred, dtype=int)
                    class_pred[ensemble_pred <= 0.33] = 0
                    class_pred[(ensemble_pred > 0.33) & (ensemble_pred <= 0.67)] = 1
                    class_pred[ensemble_pred > 0.67] = 2
                else:
                    class_pred = np.argmax(ensemble_pred, axis=1)

                accuracy = accuracy_score(true_labels, class_pred)
                f1 = f1_score(true_labels, class_pred, average='weighted')
                auc = 0.5  # 多分類情況下簡化處理

            # 計算其他指標
            diversity = self.calculate_diversity_score(predictions)

            # 權重平衡性（避免過度依賴單一模型）
            weight_values = list(weights.values())
            weight_entropy = -sum(w * np.log(w + 1e-8) for w in weight_values if w > 0)
            max_entropy = np.log(len(weight_values))
            weight_balance = weight_entropy / max_entropy if max_entropy > 0 else 0

            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'auc_score': auc,
                'diversity': diversity,
                'weight_balance': weight_balance,
                'num_active_models': sum(1 for w in weights.values() if w > 0.01)
            }

        except Exception as e:
            self.logger.error(f"融合模型評估失敗: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'auc_score': 0.0,
                'diversity': 0.0,
                'weight_balance': 0.0,
                'num_active_models': 0
            }

    def time_series_validation(self, predictions: Dict[str, np.ndarray],
                              weights: Dict[str, float],
                              true_labels: np.ndarray,
                              n_splits: int = 5) -> List[Dict[str, float]]:
        """時序交叉驗證"""

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []

        # 確保所有數組長度一致
        min_length = min(len(true_labels),
                        min(len(pred) for pred in predictions.values()))

        true_labels = true_labels[:min_length]
        trimmed_predictions = {k: v[:min_length] for k, v in predictions.items()}

        for train_idx, test_idx in tscv.split(true_labels):
            test_predictions = {k: v[test_idx] for k, v in trimmed_predictions.items()}
            test_labels = true_labels[test_idx]

            fold_result = self.evaluate_ensemble(test_predictions, weights, test_labels)
            results.append(fold_result)

        return results

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna優化目標函數"""

        # 載入數據
        predictions = self.load_real_predictions()
        true_labels = self.load_true_labels()

        if not predictions or len(true_labels) == 0:
            self.logger.warning("無法載入必要數據")
            return -999.0

        # 動態權重參數
        available_models = list(predictions.keys())
        weights = {}

        for model in available_models:
            weights[model] = trial.suggest_float(f'{model}_weight', 0.0, 1.0)

        # 融合參數
        dynamic_weighting = trial.suggest_categorical('dynamic_weighting', [True, False])
        min_diversity = trial.suggest_float('min_diversity', 0.1, 0.5)
        weight_update_freq = trial.suggest_int('weight_update_freq', 10, 100)

        try:
            # 確保所有數組長度一致
            min_length = min(len(true_labels),
                           min(len(pred) for pred in predictions.values()))

            true_labels = true_labels[:min_length]
            trimmed_predictions = {k: v[:min_length] for k, v in predictions.items()}

            # 時序交叉驗證
            cv_results = self.time_series_validation(
                trimmed_predictions, weights, true_labels, n_splits=3
            )

            if not cv_results:
                return -999.0

            # 計算平均性能
            avg_accuracy = np.mean([r['accuracy'] for r in cv_results])
            avg_f1 = np.mean([r['f1_score'] for r in cv_results])
            avg_diversity = np.mean([r['diversity'] for r in cv_results])
            avg_balance = np.mean([r['weight_balance'] for r in cv_results])

            # 約束條件
            if avg_diversity < min_diversity:
                return -999.0  # 多樣性不足

            if sum(weights.values()) == 0:
                return -999.0  # 權重全為0

            # 多目標優化得分
            score = (avg_accuracy * 0.3 +
                    avg_f1 * 0.4 +
                    avg_diversity * 0.2 +
                    avg_balance * 0.1)

            return score

        except Exception as e:
            self.logger.error(f"融合優化過程出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 100) -> Dict:
        """執行多模型融合權重優化"""
        self.logger.info("開始多模型融合權重參數優化...")

        # 創建研究
        study = optuna.create_study(
            direction='maximize',
            study_name='ensemble_optimization'
        )

        # 執行優化
        study.optimize(self.objective, n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"融合優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"最優參數: {best_params}")

        # 使用最優參數重新評估
        try:
            predictions = self.load_real_predictions()
            true_labels = self.load_true_labels()

            if predictions and len(true_labels) > 0:
                # 提取權重
                weights = {k: v for k, v in best_params.items() if k.endswith('_weight')}
                weights = {k.replace('_weight', ''): v for k, v in weights.items()}

                detailed_results = self.evaluate_ensemble(predictions, weights, true_labels)
            else:
                detailed_results = {}
        except Exception as e:
            self.logger.warning(f"無法獲取詳細結果: {e}")
            detailed_results = {}

        # 保存結果
        result = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials,
            'detailed_results': detailed_results,
            'optimization_history': [
                {'trial': i, 'score': trial.value}
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ]
        }

        # 保存到JSON文件
        output_file = self.config_path / "ensemble_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"結果已保存至: {output_file}")

        return result


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='多模型融合權重參數優化')
    parser.add_argument('--data_path', type=str, default='data/', help='數據路徑')
    parser.add_argument('--config_path', type=str, default='configs/', help='配置保存路徑')
    parser.add_argument('--model_path', type=str, help='模型路徑')
    parser.add_argument('--n_trials', type=int, default=100, help='優化試驗次數')

    args = parser.parse_args()

    # 創建優化器
    optimizer = EnsembleOptimizer(
        data_path=args.data_path,
        config_path=args.config_path,
        model_path=args.model_path
    )

    # 執行優化
    result = optimizer.optimize(n_trials=args.n_trials)

    print("多模型融合權重參數優化完成!")
    print(f"最優參數已保存至: {optimizer.config_path}/ensemble_params.json")


if __name__ == "__main__":
    main()