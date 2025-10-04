#!/usr/bin/env python3
"""
特徵選擇器 - 簡化版本，專注於特徵選擇邏輯
原 Optuna 優化功能已遷移至 optuna_system/optimizers/optuna_feature.py

負責：
- 基於不同方法進行特徵選擇
- 支持多種選擇方法: LightGBM重要性、互信息、F檢驗、RFE等
- 相關性分析和重複特徵過濾
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from .config import get_config


class FeatureSelector:
    """特徵選擇器 - 專注於特徵選擇邏輯"""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = get_config("feature", timeframe)

    def select_features_by_importance(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int, method: str = 'lgb') -> List[str]:
        """基於重要性選擇特徵"""
        try:
            print(f"🔍 使用 {method} 方法選擇 {n_features} 個特徵...")
            
            if method == 'lgb':
                return self._select_by_lgb_importance(X, y, n_features)
            elif method == 'mutual_info':
                return self._select_by_mutual_info(X, y, n_features)
            elif method == 'f_test':
                return self._select_by_f_test(X, y, n_features)
            elif method == 'rfe':
                return self._select_by_rfe(X, y, n_features)
            else:
                print(f"⚠️ 未知方法 {method}，使用默認 LightGBM 重要性")
                return self._select_by_lgb_importance(X, y, n_features)
                
        except Exception as e:
            print(f"❌ 特徵選擇失敗: {e}")
            return X.columns[:min(n_features, len(X.columns))].tolist()

    def _select_by_lgb_importance(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """基於 LightGBM 重要性選擇特徵"""
        try:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
                num_threads=1
            )
            
            model.fit(X, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            print(f"✅ LightGBM 重要性選擇完成")
            
            return selected_features
            
        except Exception as e:
            print(f"⚠️ LightGBM 重要性選擇失敗: {e}")
            return X.columns[:n_features].tolist()
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """基於互信息選擇特徵"""
        try:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            print(f"✅ 互信息選擇完成")
            
            return selected_features
            
        except Exception as e:
            print(f"⚠️ 互信息選擇失敗: {e}")
            return X.columns[:n_features].tolist()

    def _select_by_f_test(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """基於 F 檢驗選擇特徵"""
        try:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
            
            f_scores, p_values = f_classif(X_scaled, y)
            
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'f_score': f_scores,
                'p_value': p_values
            }).sort_values('f_score', ascending=False)
            
            selected_features = importance_df.head(n_features)['feature'].tolist()
            print(f"✅ F檢驗選擇完成")
            
            return selected_features
            
        except Exception as e:
            print(f"⚠️ F檢驗選擇失敗: {e}")
            return X.columns[:n_features].tolist()
    
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> List[str]:
        """基於遞歸特徵消除選擇特徵"""
        try:
            estimator = lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
                num_threads=1
            )
            
            rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
            rfe.fit(X, y)
            
            selected_features = X.columns[rfe.support_].tolist()
            print(f"✅ RFE選擇完成，選中 {len(selected_features)} 個特徵")
            
            return selected_features
            
        except Exception as e:
            print(f"⚠️ RFE選擇失敗: {e}")
            return X.columns[:n_features].tolist()
    
    def remove_correlated_features(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """移除高相關性特徵"""
        try:
            print(f"🔍 移除相關性 > {threshold} 的特徵...")
            
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > threshold)]
            
            X_filtered = X.drop(columns=to_drop)
            
            print(f"✅ 移除 {len(to_drop)} 個高相關性特徵，剩餘 {len(X_filtered.columns)} 個特徵")
            
            return X_filtered
            
        except Exception as e:
            print(f"⚠️ 相關性過濾失敗: {e}")
            return X

    def two_stage_selection(self, X: pd.DataFrame, y: pd.Series, 
                          coarse_n: int = None, fine_n: int = None,
                          coarse_method: str = 'lgb', fine_method: str = 'mutual_info') -> List[str]:
        """兩階段特徵選擇"""
        try:
            if coarse_n is None:
                coarse_n = self.config.get('coarse_k_range', (60, 80))[1]
            if fine_n is None:
                fine_n = self.config.get('fine_k_range', (20, 30))[1]
            
            print(f"🎯 兩階段特徵選擇: 粗選 {coarse_n} -> 精選 {fine_n}")
            
            # 第一階段：粗選
            coarse_features = self.select_features_by_importance(X, y, coarse_n, coarse_method)
            X_coarse = X[coarse_features]
            
            # 第二階段：精選
            fine_features = self.select_features_by_importance(X_coarse, y, fine_n, fine_method)
            
            print(f"✅ 兩階段選擇完成：{len(X.columns)} -> {len(coarse_features)} -> {len(fine_features)}")
            
            return fine_features
            
        except Exception as e:
            print(f"❌ 兩階段選擇失敗: {e}")
            return X.columns[:fine_n if fine_n else 20].tolist()

    def get_available_methods(self) -> List[str]:
        """獲取可用的特徵選擇方法"""
        return ['lgb', 'mutual_info', 'f_test', 'rfe']


# 向後兼容性
FeatureOptimizer = FeatureSelector  # 別名，保持向後兼容
