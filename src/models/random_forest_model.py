"""
隨機森林模型
基於scikit-learn的隨機森林分類器和回歸器
針對量化交易場景進行優化，支持特徵重要性分析
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings

# 機器學習庫
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class RandomForestModel(BaseModel):
    """隨機森林模型類"""

    def __init__(self, symbol: str, timeframe: str, version: str = "1.0.0", **kwargs):
        """
        初始化隨機森林模型

        Args:
            symbol: 交易品種
            timeframe: 時間框架
            version: 模型版本
            **kwargs: 隨機森林參數
        """
        super().__init__(symbol, timeframe, version, **kwargs)

        # 設置模型類型
        self.model_type = kwargs.get('model_type', 'classification')  # 'classification' or 'regression'

        # 隨機森林參數
        self.n_estimators = kwargs.get('n_estimators', 100)  # 樹的數量
        self.max_depth = kwargs.get('max_depth', None)  # 最大深度
        self.min_samples_split = kwargs.get('min_samples_split', 2)  # 分裂所需最小樣本數
        self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)  # 葉節點最小樣本數
        self.max_features = kwargs.get('max_features', 'sqrt')  # 最大特徵數
        self.bootstrap = kwargs.get('bootstrap', True)  # 是否使用bootstrap
        self.oob_score = kwargs.get('oob_score', False)  # 是否計算OOB分數
        self.random_state = kwargs.get('random_state', 42)  # 隨機種子

        # 高級參數
        self.class_weight = kwargs.get('class_weight', None)  # 類別權重
        self.criterion = kwargs.get('criterion', 'gini' if self.model_type == 'classification' else 'squared_error')
        self.max_leaf_nodes = kwargs.get('max_leaf_nodes', None)  # 最大葉節點數
        self.min_impurity_decrease = kwargs.get('min_impurity_decrease', 0.0)  # 最小不純度減少
        self.ccp_alpha = kwargs.get('ccp_alpha', 0.0)  # 複雜性修剪參數

        # 並行處理
        self.n_jobs = kwargs.get('n_jobs', -1)  # 並行作業數

        # 特徵重要性相關
        self.feature_importance_method = kwargs.get('feature_importance_method', 'gini')  # 'gini', 'permutation'
        self.permutation_scoring = kwargs.get('permutation_scoring', 'accuracy')  # 排列重要性評分方法

        # 超參數搜索
        self.enable_grid_search = kwargs.get('enable_grid_search', False)
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.search_type = kwargs.get('search_type', 'grid')  # 'grid' or 'random'

        # 存儲額外信息
        self.feature_importances_ = None
        self.oob_score_ = None
        self.label_encoder = None  # 用於多分類標籤編碼

        logger.info(f"初始化隨機森林模型: {self.model_type}, n_estimators={self.n_estimators}")

    def _create_model(self) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """創建隨機森林模型實例"""
        try:
            base_params = {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'oob_score': self.oob_score,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'criterion': self.criterion,
                'max_leaf_nodes': self.max_leaf_nodes,
                'min_impurity_decrease': self.min_impurity_decrease,
                'ccp_alpha': self.ccp_alpha
            }

            if self.model_type == 'classification':
                base_params['class_weight'] = self.class_weight
                model = RandomForestClassifier(**base_params)
            else:
                model = RandomForestRegressor(**base_params)

            logger.debug(f"隨機森林模型創建完成: {self.model_type}")
            return model

        except Exception as e:
            logger.error(f"創建隨機森林模型失敗: {e}")
            raise

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'RandomForestModel':
        """訓練隨機森林模型"""
        try:
            logger.info("開始訓練隨機森林模型")
            start_time = datetime.now()

            # 驗證數據
            if not self.validate_input_data(X, y):
                raise ValueError("輸入數據驗證失敗")

            # 保存特徵信息
            self.feature_names = X.columns.tolist()
            self.target_name = y.name if hasattr(y, 'name') else 'target'

            # 處理分類標籤
            y_processed = y.copy()
            if self.model_type == 'classification':
                # 檢查是否需要標籤編碼
                if y.dtype == 'object' or not np.issubdtype(y.dtype, np.integer):
                    self.label_encoder = LabelEncoder()
                    y_processed = self.label_encoder.fit_transform(y)
                    logger.debug(f"標籤編碼完成，類別數: {len(self.label_encoder.classes_)}")

            # 創建模型
            if self.enable_grid_search:
                self.model = self._perform_hyperparameter_search(X, y_processed)
            else:
                self.model = self._create_model()

            # 訓練模型
            logger.info(f"開始訓練，數據形狀: {X.shape}")
            self.model.fit(X, y_processed)

            # 更新模型狀態
            self.is_fitted = True
            training_time = (datetime.now() - start_time).total_seconds()

            # 保存特徵重要性
            self._calculate_feature_importance(X, y_processed)

            # 保存OOB分數（如果啟用）
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
                self.oob_score_ = self.model.oob_score_
                logger.info(f"OOB分數: {self.oob_score_:.4f}")

            # 更新元數據
            self.metadata.update({
                'trained_at': datetime.now(),
                'training_time': training_time,
                'data_shape': X.shape,
                'feature_count': len(self.feature_names),
                'n_estimators': self.model.n_estimators,
                'oob_score': self.oob_score_,
                'max_depth': self.model.max_depth,
                'n_classes': len(np.unique(y_processed)) if self.model_type == 'classification' else None
            })

            logger.info(f"隨機森林模型訓練完成，用時: {training_time:.2f}秒")

            # 評估模型
            self.evaluate(X, y, "training")
            if X_val is not None and y_val is not None:
                self.evaluate(X_val, y_val, "validation")

            return self

        except Exception as e:
            logger.error(f"隨機森林模型訓練失敗: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """預測"""
        try:
            if not self.is_fitted:
                raise ValueError("模型尚未訓練")

            # 驗證輸入特徵
            if not self.validate_input_data(X):
                raise ValueError("輸入數據驗證失敗")

            # 確保特徵順序一致
            X_ordered = X[self.feature_names]

            # 預測
            predictions = self.model.predict(X_ordered)

            # 如果有標籤編碼器，需要反編碼
            if self.model_type == 'classification' and self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(predictions)

            return predictions

        except Exception as e:
            logger.error(f"隨機森林預測失敗: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """預測概率（僅分類模型）"""
        try:
            if self.model_type != 'classification':
                raise ValueError("概率預測僅適用於分類模型")

            if not self.is_fitted:
                raise ValueError("模型尚未訓練")

            # 確保特徵順序一致
            X_ordered = X[self.feature_names]

            # 預測概率
            probabilities = self.model.predict_proba(X_ordered)

            return probabilities

        except Exception as e:
            logger.error(f"隨機森林概率預測失敗: {e}")
            raise

    def _calculate_feature_importance(self, X: pd.DataFrame, y: np.ndarray):
        """計算特徵重要性"""
        try:
            # 基於不純度的特徵重要性（默認）
            importances = self.model.feature_importances_
            self.feature_importances_ = dict(zip(self.feature_names, importances))

            # 排列重要性（可選）
            if self.feature_importance_method == 'permutation':
                logger.debug("計算排列特徵重要性")
                perm_importance = permutation_importance(
                    self.model, X, y,
                    scoring=self.permutation_scoring,
                    n_repeats=5,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )

                # 使用排列重要性替換默認重要性
                perm_importances = perm_importance.importances_mean
                self.feature_importances_ = dict(zip(self.feature_names, perm_importances))

            logger.debug("特徵重要性計算完成")

        except Exception as e:
            logger.error(f"計算特徵重要性失敗: {e}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """獲取特徵重要性"""
        return self.feature_importances_

    def _perform_hyperparameter_search(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Union[RandomForestClassifier, RandomForestRegressor]:
        """執行超參數搜索"""
        try:
            logger.info("開始超參數搜索")

            # 定義參數網格
            param_grid = self._get_parameter_grid()

            # 創建基礎模型
            base_model = self._create_model()

            # 選擇搜索策略
            if self.search_type == 'grid':
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=self.cv_folds,
                    scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error',
                    n_jobs=self.n_jobs,
                    verbose=1
                )
            else:  # random search
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=20,
                    cv=self.cv_folds,
                    scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error',
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=1
                )

            # 執行搜索
            search.fit(X, y)

            # 保存最佳參數
            self.best_params = search.best_params_
            logger.info(f"最佳參數: {self.best_params}")
            logger.info(f"最佳分數: {search.best_score_:.4f}")

            return search.best_estimator_

        except Exception as e:
            logger.error(f"超參數搜索失敗: {e}")
            return self._create_model()

    def _get_parameter_grid(self) -> Dict:
        """獲取參數搜索網格"""
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        if self.model_type == 'classification':
            param_grid['criterion'] = ['gini', 'entropy']
        else:
            param_grid['criterion'] = ['squared_error', 'absolute_error', 'poisson']

        return param_grid

    def get_tree_depths(self) -> List[int]:
        """獲取所有樹的深度"""
        if not self.is_fitted:
            return []

        depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        return depths

    def get_oob_score(self) -> Optional[float]:
        """獲取袋外分數"""
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
            return self.model.oob_score_
        return None

    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """繪製特徵重要性圖"""
        try:
            if self.feature_importances_ is None:
                logger.warning("特徵重要性未計算")
                return

            # 準備數據
            importance_df = pd.DataFrame([
                {'feature': feature, 'importance': importance}
                for feature, importance in self.feature_importances_.items()
            ]).sort_values('importance', ascending=False).head(top_n)

            # 繪製圖表
            plt.figure(figsize=figsize)
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importances - {self.symbol} {self.timeframe}')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"特徵重要性圖表已保存到: {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"繪製特徵重要性圖失敗: {e}")

    def plot_tree_depths(self, save_path: Optional[Path] = None):
        """繪製樹深度分布"""
        try:
            depths = self.get_tree_depths()
            if not depths:
                logger.warning("無樹深度數據")
                return

            plt.figure(figsize=(10, 6))
            plt.hist(depths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f'Distribution of Tree Depths - {self.symbol} {self.timeframe}')
            plt.xlabel('Tree Depth')
            plt.ylabel('Frequency')
            plt.axvline(np.mean(depths), color='red', linestyle='--', label=f'Mean: {np.mean(depths):.1f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"樹深度分布圖已保存到: {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"繪製樹深度分布圖失敗: {e}")

    def get_model_complexity(self) -> Dict[str, Any]:
        """獲取模型複雜度信息"""
        if not self.is_fitted:
            return {}

        complexity_info = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': len(self.feature_names),
            'n_nodes_total': sum(tree.tree_.node_count for tree in self.model.estimators_),
            'n_leaves_total': sum(tree.tree_.n_leaves for tree in self.model.estimators_),
            'avg_nodes_per_tree': np.mean([tree.tree_.node_count for tree in self.model.estimators_]),
            'avg_leaves_per_tree': np.mean([tree.tree_.n_leaves for tree in self.model.estimators_]),
            'avg_depth': np.mean(self.get_tree_depths()),
            'max_depth_actual': max(self.get_tree_depths()) if self.get_tree_depths() else 0
        }

        return complexity_info

    def analyze_overfitting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """分析過擬合情況"""
        try:
            if not self.is_fitted:
                raise ValueError("模型尚未訓練")

            # 訓練集性能
            train_pred = self.predict(X_train)
            if self.model_type == 'classification':
                from sklearn.metrics import accuracy_score
                train_score = accuracy_score(y_train, train_pred)
                val_pred = self.predict(X_val)
                val_score = accuracy_score(y_val, val_pred)
                metric_name = 'accuracy'
            else:
                from sklearn.metrics import r2_score
                train_score = r2_score(y_train, train_pred)
                val_pred = self.predict(X_val)
                val_score = r2_score(y_val, val_pred)
                metric_name = 'r2_score'

            # 過擬合分析
            overfitting_gap = train_score - val_score
            overfitting_ratio = overfitting_gap / train_score if train_score != 0 else 0

            analysis = {
                f'train_{metric_name}': train_score,
                f'val_{metric_name}': val_score,
                'overfitting_gap': overfitting_gap,
                'overfitting_ratio': overfitting_ratio,
                'is_overfitting': overfitting_gap > 0.05  # 5%閾值
            }

            logger.info(f"過擬合分析: 訓練={train_score:.4f}, 驗證={val_score:.4f}, 差距={overfitting_gap:.4f}")
            return analysis

        except Exception as e:
            logger.error(f"過擬合分析失敗: {e}")
            return {}

    def feature_selection_by_importance(self, threshold: float = 0.01) -> List[str]:
        """基於重要性進行特徵選擇"""
        if self.feature_importances_ is None:
            logger.warning("特徵重要性未計算")
            return self.feature_names

        selected_features = [
            feature for feature, importance in self.feature_importances_.items()
            if importance >= threshold
        ]

        logger.info(f"基於重要性閾值 {threshold} 選擇了 {len(selected_features)} 個特徵")
        return selected_features

    def get_decision_path(self, X: pd.DataFrame, tree_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """獲取決策路徑（用於解釋預測）"""
        try:
            if not self.is_fitted:
                raise ValueError("模型尚未訓練")

            if tree_index >= len(self.model.estimators_):
                raise ValueError(f"樹索引 {tree_index} 超出範圍")

            tree = self.model.estimators_[tree_index]

            # 確保特徵順序一致
            X_ordered = X[self.feature_names]

            # 獲取決策路徑
            leaf_ids = tree.apply(X_ordered)
            feature_names = self.feature_names

            return leaf_ids, feature_names

        except Exception as e:
            logger.error(f"獲取決策路徑失敗: {e}")
            return np.array([]), []
