# -*- coding: utf-8 -*-
"""
多項式特徵生成優化器
自動生成和選擇最優的多項式特徵組合
"""
import optuna
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class PolynomialOptimizer:
    """多項式特徵生成參數優化器"""

    def __init__(self, data_path: str, config_path: str = "configs/"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

        # 緩存生成的特徵
        self.feature_cache = {}

    def generate_polynomial_features(self, X: np.ndarray, params: Dict) -> Tuple[np.ndarray, List[str]]:
        """生成多項式特徵"""
        try:
            # 多項式特徵生成器
            poly = PolynomialFeatures(
                degree=params['poly_degree'],
                interaction_only=params['interaction_only'],
                include_bias=params['include_bias'],
                order='C'
            )

            # 為了避免內存問題，限制輸入特徵數量
            max_features = params.get('max_input_features', 50)
            if X.shape[1] > max_features:
                # 使用方差篩選保留最重要的特徵
                variances = np.var(X, axis=0)
                top_indices = np.argsort(variances)[-max_features:]
                X_reduced = X[:, top_indices]
            else:
                X_reduced = X

            # 生成多項式特徵
            X_poly = poly.fit_transform(X_reduced)

            # 獲取特徵名稱
            if hasattr(poly, 'get_feature_names_out'):
                feature_names = poly.get_feature_names_out([f'f{i}' for i in range(X_reduced.shape[1])]).tolist()
            else:
                # 手動生成特徵名稱
                feature_names = [f'poly_feature_{i}' for i in range(X_poly.shape[1])]

            self.logger.info(f"生成了 {X_poly.shape[1]} 個多項式特徵")

            return X_poly, feature_names

        except Exception as e:
            self.logger.error(f"多項式特徵生成失敗: {e}")
            return X, [f'original_feature_{i}' for i in range(X.shape[1])]

    def select_polynomial_features(self, X_poly: np.ndarray, y: np.ndarray,
                                 feature_names: List[str], params: Dict) -> Tuple[np.ndarray, List[str]]:
        """選擇最優的多項式特徵"""
        try:
            if X_poly.shape[1] <= params['n_poly_features']:
                return X_poly, feature_names

            # 特徵選擇方法
            selection_method = params['poly_selection_method']
            n_features = min(params['n_poly_features'], X_poly.shape[1])

            if selection_method == 'f_regression':
                selector = SelectKBest(score_func=f_regression, k=n_features)
            elif selection_method == 'mutual_info':
                selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
            else:
                # 默認使用f_regression
                selector = SelectKBest(score_func=f_regression, k=n_features)

            # 處理無穷大和NaN值
            X_poly = np.nan_to_num(X_poly, nan=0.0, posinf=1e10, neginf=-1e10)

            # 執行特徵選擇
            X_selected = selector.fit_transform(X_poly, y)
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices]

            self.logger.info(f"從 {X_poly.shape[1]} 個特徵中選擇了 {X_selected.shape[1]} 個")

            return X_selected, selected_names

        except Exception as e:
            self.logger.error(f"多項式特徵選擇失敗: {e}")
            # 返回前n個特徵作為備選
            n = min(params['n_poly_features'], X_poly.shape[1])
            return X_poly[:, :n], feature_names[:n]

    def combine_original_and_polynomial(self, X_original: np.ndarray, X_poly: np.ndarray,
                                      original_names: List[str], poly_names: List[str],
                                      params: Dict) -> Tuple[np.ndarray, List[str]]:
        """將原始特徵和多項式特徵組合"""

        # 計算特徵比例
        poly_ratio = params['poly_feature_ratio']
        total_features = params.get('max_total_features', 200)

        n_poly_features = int(total_features * poly_ratio)
        n_original_features = total_features - n_poly_features

        # 限制原始特徵數量
        if X_original.shape[1] > n_original_features:
            X_original_selected = X_original[:, :n_original_features]
            original_names_selected = original_names[:n_original_features]
        else:
            X_original_selected = X_original
            original_names_selected = original_names

        # 限制多項式特徵數量
        if X_poly.shape[1] > n_poly_features:
            X_poly_selected = X_poly[:, :n_poly_features]
            poly_names_selected = poly_names[:n_poly_features]
        else:
            X_poly_selected = X_poly
            poly_names_selected = poly_names

        # 組合特徵
        X_combined = np.hstack([X_original_selected, X_poly_selected])
        combined_names = original_names_selected + [f"poly_{name}" for name in poly_names_selected]

        return X_combined, combined_names

    def evaluate_polynomial_features(self, X: np.ndarray, y: np.ndarray, params: Dict) -> Dict:
        """評估多項式特徵的有效性"""
        try:
            # 數據分割
            split_idx = int(len(X) * 0.7)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 原始特徵名稱
            original_names = [f'feature_{i}' for i in range(X.shape[1])]

            # 生成多項式特徵
            X_train_poly, poly_feature_names = self.generate_polynomial_features(X_train, params)

            # 選擇多項式特徵
            X_train_poly_selected, selected_poly_names = self.select_polynomial_features(
                X_train_poly, y_train, poly_feature_names, params
            )

            # 組合原始特徵和多項式特徵
            X_train_combined, combined_names = self.combine_original_and_polynomial(
                X_train, X_train_poly_selected,
                original_names, selected_poly_names, params
            )

            # 標準化特徵
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_combined)

            # 對驗證集應用相同的變換
            X_val_poly = PolynomialFeatures(
                degree=params['poly_degree'],
                interaction_only=params['interaction_only'],
                include_bias=params['include_bias']
            ).fit(X_train).transform(X_val)

            # 選擇相同的多項式特徵
            if X_val_poly.shape[1] > len(selected_poly_names):
                # 使用訓練時的選擇器
                selector = SelectKBest(score_func=f_regression, k=len(selected_poly_names))
                selector.fit(X_train_poly, y_train)
                X_val_poly_selected = selector.transform(X_val_poly)
            else:
                X_val_poly_selected = X_val_poly

            # 組合驗證集特徵
            X_val_combined, _ = self.combine_original_and_polynomial(
                X_val, X_val_poly_selected,
                original_names, selected_poly_names, params
            )

            X_val_scaled = scaler.transform(X_val_combined)

            # 訓練簡單的線性回歸模型進行評估
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # 預測
            y_train_pred = model.predict(X_train_scaled)
            y_val_pred = model.predict(X_val_scaled)

            # 計算評估指標
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)

            # 計算過擬合程度
            overfitting = train_r2 - val_r2

            # 特徵數量統計
            n_original = X.shape[1]
            n_polynomial = len(selected_poly_names)
            n_total = X_train_combined.shape[1]

            return {
                'val_r2': val_r2,
                'train_r2': train_r2,
                'val_mse': val_mse,
                'train_mse': train_mse,
                'overfitting': overfitting,
                'n_original_features': n_original,
                'n_polynomial_features': n_polynomial,
                'n_total_features': n_total,
                'feature_names': combined_names[:50]  # 只保存前50個特徵名
            }

        except Exception as e:
            self.logger.error(f"多項式特徵評估失敗: {e}")
            return {
                'val_r2': -999.0,
                'train_r2': 0.0,
                'val_mse': 999.0,
                'train_mse': 999.0,
                'overfitting': 999.0,
                'n_original_features': X.shape[1],
                'n_polynomial_features': 0,
                'n_total_features': X.shape[1]
            }

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna優化目標函數"""

        # 多項式特徵生成參數
        params = {
            'poly_degree': trial.suggest_int('poly_degree', 2, 4),
            'interaction_only': trial.suggest_categorical('interaction_only', [True, False]),
            'include_bias': trial.suggest_categorical('include_bias', [True, False]),

            # 特徵選擇參數
            'poly_selection_method': trial.suggest_categorical('poly_selection_method',
                                                             ['f_regression', 'mutual_info']),
            'n_poly_features': trial.suggest_int('n_poly_features', 10, 100),
            'poly_feature_ratio': trial.suggest_float('poly_feature_ratio', 0.1, 0.5),

            # 限制參數
            'max_input_features': trial.suggest_int('max_input_features', 20, 80),
            'max_total_features': trial.suggest_int('max_total_features', 50, 300),

            # 正則化參數
            'max_overfitting': trial.suggest_float('max_overfitting', 0.1, 0.3),
        }

        try:
            # 加載數據
            data_file = self.data_path / "feature_data.csv"
            if not data_file.exists():
                # 嘗試其他可能的數據文件
                alternative_files = [
                    self.data_path / "features.csv",
                    self.data_path / "X_data.csv",
                    self.data_path / "training_data.csv"
                ]

                data_file = None
                for alt_file in alternative_files:
                    if alt_file.exists():
                        data_file = alt_file
                        break

                if data_file is None:
                    self.logger.warning("無法找到特徵數據文件，生成模擬數據")
                    # 生成模擬數據用於測試
                    np.random.seed(42)
                    n_samples, n_features = 1000, 20
                    X = np.random.randn(n_samples, n_features)
                    y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1
                else:
                    df = pd.read_csv(data_file)

                    # 準備數據
                    feature_cols = [col for col in df.columns if col.startswith('feature_') or col.startswith('f_')]
                    if len(feature_cols) == 0:
                        # 如果沒有明確的特徵列，使用除最後一列外的所有數值列
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) > 1:
                            feature_cols = numeric_cols[:-1]  # 假設最後一列是目標變量
                        else:
                            self.logger.warning("無法識別特徵列")
                            return -999.0

                    X = df[feature_cols].values

                    # 目標變量
                    if 'target' in df.columns:
                        y = df['target'].values
                    elif 'label' in df.columns:
                        y = df['label'].values.astype(float)
                    elif 'y' in df.columns:
                        y = df['y'].values
                    else:
                        # 使用最後一個數值列作為目標
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        y = df[numeric_cols[-1]].values
            else:
                df = pd.read_csv(data_file)

                # 準備數據
                feature_cols = [col for col in df.columns if col.startswith('feature_')]
                if len(feature_cols) == 0:
                    self.logger.warning("未找到特徵列")
                    return -999.0

                X = df[feature_cols].values

                # 目標變量 (假設是連續值，用於回歸)
                if 'target' in df.columns:
                    y = df['target'].values
                elif 'label' in df.columns:
                    y = df['label'].values.astype(float)
                else:
                    self.logger.warning("未找到目標變量")
                    return -999.0

            # 評估多項式特徵
            results = self.evaluate_polynomial_features(X, y, params)

            # 過擬合約束
            if results['overfitting'] > params['max_overfitting']:
                return -999.0

            # 驗證集R²得分作為主要優化目標
            val_r2 = results['val_r2']

            # 對特徵數量進行輕微懲罰，鼓勵簡潔的特徵集
            feature_penalty = results['n_total_features'] / 1000.0

            # 最終得分
            score = val_r2 - feature_penalty

            return score

        except Exception as e:
            self.logger.error(f"優化過程中出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 100) -> Dict:
        """執行多項式特徵參數優化"""
        self.logger.info("開始多項式特徵生成參數優化...")

        # 創建研究
        study = optuna.create_study(
            direction='maximize',
            study_name='polynomial_optimization'
        )

        # 執行優化
        study.optimize(self.objective, n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"最優參數: {best_params}")

        # 使用最優參數重新評估，獲取詳細信息
        try:
            # 生成模擬數據用於最終評估
            np.random.seed(42)
            n_samples, n_features = 1000, 20
            X = np.random.randn(n_samples, n_features)
            y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1

            detailed_results = self.evaluate_polynomial_features(X, y, best_params)
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
        output_file = self.config_path / "polynomial_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"結果已保存至: {output_file}")

        return result


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='多項式特徵生成參數優化')
    parser.add_argument('--data_path', type=str, default='data/', help='數據路徑')
    parser.add_argument('--config_path', type=str, default='configs/', help='配置保存路徑')
    parser.add_argument('--n_trials', type=int, default=100, help='優化試驗次數')

    args = parser.parse_args()

    # 創建優化器
    optimizer = PolynomialOptimizer(
        data_path=args.data_path,
        config_path=args.config_path
    )

    # 執行優化
    result = optimizer.optimize(n_trials=args.n_trials)

    print("多項式特徵生成參數優化完成!")
    print(f"最優參數已保存至: {optimizer.config_path}/polynomial_params.json")


if __name__ == "__main__":
    main()