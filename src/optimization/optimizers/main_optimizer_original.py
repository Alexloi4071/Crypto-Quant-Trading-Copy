#!/usr/bin/env python3
"""
主控制器 - 模組化Optuna優化系統

統一管理標籤優化、特徵選擇、模型超參數優化三個模組
實現端到端的自動化優化流程，支持方案B到方案C的平滑升級
增強報告功能與過度擬合檢測
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

# 添加項目根目錄到Python路徑
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

# 導入模組
try:
    from .config import OptimizationConfig, get_config
    from .label_optimizer import LabelOptimizer
    from .feature_selector import FeatureSelector
    from .model_optimizer import ModelOptimizer
except ImportError:
    # 處理相對導入問題
    from config import OptimizationConfig, get_config
    from label_optimizer import LabelOptimizer
    from feature_selector import FeatureSelector
    from model_optimizer import ModelOptimizer

# 🔧 移除旧系统导入 - 使用新的optimization模块


class ModularOptunaOptimizer:
    """模組化Optuna優化系統主控制器"""

    def __init__(self, symbol: str, timeframe: str, version: str = None, auto_version: bool = True, 
                 use_saved_params: bool = True):
        self.symbol = symbol
        self.timeframe = timeframe
        
        # 🔢 版本管理
        if version:
            self.version = version
        elif auto_version:
            self.version = self.get_next_version()
        else:
            self.version = "v1"
        
        # 初始化配置
        self.config = OptimizationConfig()
        
        # 🔧 載入已保存的Optuna最佳參數（如果存在）
        self.saved_params = {}
        if use_saved_params:
            self.saved_params = self._load_saved_optuna_params()
            if self.saved_params:
                print(f"🔧 載入已保存的Optuna參數: {len(self.saved_params)}個組件")
                for component, params in self.saved_params.items():
                    print(f"   - {component}: {len(params)}個參數")
        
        # 驗證配置
        if not self.config.validate_config(symbol, timeframe):
            print("⚠️ 配置驗證失敗，使用默認配置")
        
        # 創建目錄結構
        self.config.create_directories()
        
        # 🔧 移除旧系统初始化 - 优化模块已足够
        
        # 初始化優化模組（使用已保存的參數）
        label_params = self.saved_params.get('labels', {})
        feature_params = self.saved_params.get('features', {})
        model_params = self.saved_params.get('model', {})
        
        self.label_optimizer = LabelOptimizer(symbol, timeframe, custom_params=label_params)
        self.feature_selector = FeatureSelector(symbol, timeframe, custom_params=feature_params)
        self.model_optimizer = ModelOptimizer(symbol, timeframe, custom_params=model_params)
        
        # 結果存儲
        self.optimization_results = {}
        self.execution_log = []
        
        # 數據路徑
        self.setup_data_paths()
        
        print(f"🚀 初始化模組化Optuna優化器 - {symbol} {timeframe}")
        print(f"📊 當前版本: {self.version}")
        print(f"✅ 已載入新的optimization優化模組")
    
    def _load_saved_optuna_params(self) -> Dict[str, Dict[str, Any]]:
        """載入已保存的Optuna最佳參數"""
        saved_params = {}
        
        try:
            # 參數目錄結構: results/optimal_params/SYMBOL/TIMEFRAME/COMPONENT/
            base_dir = f"results/optimal_params/{self.symbol}/{self.timeframe}"
            
            for component in ["labels", "features", "model"]:
                component_dir = f"{base_dir}/{component}"
                latest_file = f"{component_dir}/{component}_latest.json"
                
                if os.path.exists(latest_file):
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        param_data = json.load(f)
                        saved_params[component] = param_data.get('best_params', {})
                        print(f"   ✅ 載入{component}參數: {len(saved_params[component])}個")
                else:
                    print(f"   ⚠️  {component}參數檔案不存在: {latest_file}")
            
            return saved_params
            
        except Exception as e:
            print(f"⚠️  載入已保存參數失敗: {e}")
            return {}
    
    def save_optuna_params(self, component: str, params: Dict[str, Any], score: float):
        """保存Optuna最佳參數到結構化目錄"""
        try:
            # 創建目錄結構
            component_dir = f"results/optimal_params/{self.symbol}/{self.timeframe}/{component}"
            os.makedirs(component_dir, exist_ok=True)
            
            # 創建參數記錄
            param_record = {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "component": component,
                "timestamp": datetime.now().isoformat(),
                "best_params": params,
                "best_score": score,
                "metadata": {"source": "main_optimizer", "version": self.version}
            }
            
            # 保存到latest檔案
            latest_file = f"{component_dir}/{component}_latest.json"
            with open(latest_file, 'w', encoding='utf-8') as f:
                json.dump(param_record, f, indent=2, ensure_ascii=False)
            
            print(f"💾 {component}參數已保存: {len(params)}個參數")
            return True
            
        except Exception as e:
            print(f"❌ 保存{component}參數失敗: {e}")
            return False
    
    def run_labels_optuna_optimization(self, n_trials: int = 200, 
                                      features_df: pd.DataFrame = None, ohlcv_df: pd.DataFrame = None) -> Dict[str, Any]:
        """執行Labels參數的Optuna優化"""
        print(f"🎯 開始Labels Optuna優化 - {self.symbol} {self.timeframe}")
        print(f"Trial數量: {n_trials}")
        
        # 🔧 如果沒有提供數據，自動載入
        if features_df is None or ohlcv_df is None:
            print("🔧 自動載入數據...")
            features_df, ohlcv_df = self.load_data()
            if features_df is None or ohlcv_df is None:
                print("❌ 數據載入失敗")
                return {}
        
        try:
            # 使用LabelOptimizer進行優化
            price_data = ohlcv_df['close']
            results = self.label_optimizer.optimize(features_df, price_data, n_trials=n_trials)
            
            if results and 'best_params' in results:
                # 保存最佳參數
                self.save_optuna_params('labels', results['best_params'], results.get('best_score', 0.0))
                
                print(f"✅ Labels優化完成")
                print(f"最佳參數: {results['best_params']}")
                print(f"最佳分數: {results.get('best_score', 0.0):.4f}")
                
                return results
            else:
                print("❌ Labels優化失敗")
                return {}
                
        except Exception as e:
            print(f"❌ Labels優化出錯: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_models_optuna_optimization(self, n_trials: int = 100,
                                     features_df: pd.DataFrame = None, labels: pd.Series = None,
                                     selected_features: List[str] = None) -> Dict[str, Any]:
        """執行Models參數的Optuna優化"""
        print(f"🤖 開始Models Optuna優化 - {self.symbol} {self.timeframe}")
        print(f"Trial數量: {n_trials}")
        
        # 🔧 如果沒有提供數據，自動載入
        if features_df is None or labels is None:
            print("🔧 自動載入數據...")
            features_df, ohlcv_df = self.load_data()
            if features_df is None:
                print("❌ 特徵數據載入失敗")
                return {}
            
            # 生成標籤
            try:
                labels = self.label_optimizer.generate_labels(
                    price_data=ohlcv_df['close'],
                    lag=5,
                    pos_threshold=0.75,
                    neg_threshold=0.25, 
                    label_type="technical_based",
                    threshold_method="quantile"
                )
            except Exception as e:
                print(f"❌ 標籤生成失敗: {e}")
                return {}
        
        # 如果沒有提供選定特徵，使用所有特徵
        if selected_features is None:
            selected_features = features_df.columns.tolist()
            
        print(f"特徵數量: {len(selected_features)}")
        
        try:
            # 使用ModelOptimizer進行優化（完整驗證模式）
            X = features_df[selected_features]
            results = self.model_optimizer.optimize(X, labels, n_trials=n_trials, fast_mode=False)
            
            if results and 'best_params' in results:
                # 保存最佳參數
                self.save_optuna_params('model', results['best_params'], results.get('best_score', 0.0))
                
                print(f"✅ Models優化完成")
                print(f"最佳參數: {results['best_params']}")
                print(f"最佳分數: {results.get('best_score', 0.0):.4f}")
                
                return results
            else:
                print("❌ Models優化失敗")
                return {}
                
        except Exception as e:
            print(f"❌ Models優化出錯: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def run_features_optuna_optimization(self, n_trials: int = 100, 
                                        features_df: pd.DataFrame = None, labels: pd.Series = None) -> Dict[str, Any]:
        """為Features組件運行Optuna優化，找出最佳特徵選擇參數"""
        print(f"\n🔍 開始Features組件Optuna優化 - {self.symbol} {self.timeframe}")
        
        # 🔧 如果沒有提供數據，自動載入
        if features_df is None or labels is None:
            print("🔧 自動載入數據...")
            features_df, ohlcv_df = self.load_data()
            if features_df is None:
                print("❌ 特徵數據載入失敗")
                return {}
            
            # 生成標籤
            try:
                labels = self.label_optimizer.generate_labels(
                    price_data=ohlcv_df['close'],
                    lag=5,
                    pos_threshold=0.75,
                    neg_threshold=0.25, 
                    label_type="technical_based",
                    threshold_method="quantile"
                )
            except Exception as e:
                print(f"❌ 標籤生成失敗: {e}")
                return {}
        
        print(f"🎯 目標: 從{len(features_df.columns)}個特徵中找出最佳選擇策略")
        print(f"🔄 試驗數量: {n_trials}")
        
        import optuna
        
        # 🔧 預處理數據，避免每個trial重複處理
        print("🔧 預處理數據以加速Optuna搜索...")
        min_length = min(len(features_df), len(labels))
        X_base = features_df.iloc[:min_length]
        y_base = labels.iloc[:min_length]
        
        # 使用採樣來加速搜索 (最新20%數據)
        sample_size = min(20000, len(X_base))  # 最多2萬樣本
        X_sample = X_base.tail(sample_size)
        y_sample = y_base.tail(sample_size)
        print(f"🎯 使用採樣數據: {X_sample.shape} (原始: {X_base.shape})")
        
        def features_objective(trial):
            """Features優化目標函數 (優化版)"""
            try:
                # 1. 特徵選擇方法
                selection_method = trial.suggest_categorical(
                    "feature_selection_method", 
                    ["lightgbm", "mutual_info", "combined"]  # 移除RFE，太慢
                )
                
                # 2. 特徵數量控制
                target_feature_count = trial.suggest_int("target_feature_count", 15, 35)
                
                # 3. 相關性控制
                correlation_threshold = trial.suggest_float("correlation_threshold", 0.80, 0.95)
                
                # 4. 特徵類型權重
                technical_weight = trial.suggest_float("technical_weight", 0.8, 1.5)
                volume_weight = trial.suggest_float("volume_weight", 0.8, 1.5)
                time_weight = trial.suggest_float("time_weight", 0.5, 1.2)
                
                # 使用預處理的採樣數據
                X = X_sample
                y = y_sample
                
                # 應用特徵類型權重
                feature_weights = self._calculate_feature_weights(
                    X.columns, technical_weight, volume_weight, time_weight
                )
                
                # 第一階段：粗選 (使用目標數量的2倍作為粗選)
                coarse_k = min(target_feature_count * 2, len(X.columns))
                
                if selection_method == "lightgbm":
                    selected_features = self._select_features_lightgbm(X, y, coarse_k, feature_weights)
                elif selection_method == "mutual_info":
                    selected_features = self._select_features_mutual_info(X, y, coarse_k)
                elif selection_method == "combined":
                    # 組合方法
                    lgb_features = self._select_features_lightgbm(X, y, coarse_k//2, feature_weights)
                    mi_features = self._select_features_mutual_info(X, y, coarse_k//2)
                    selected_features = list(set(lgb_features + mi_features))[:coarse_k]
                
                # 去相關處理
                if len(selected_features) > 1:
                    selected_features = self._remove_correlated_features(
                        X[selected_features], correlation_threshold
                    )
                
                # 第二階段：精選到目標數量
                if len(selected_features) > target_feature_count:
                    final_features = self._select_features_lightgbm(
                        X[selected_features], y, target_feature_count, feature_weights
                    )
                else:
                    final_features = selected_features
                
                # 評估特徵質量
                if len(final_features) < 5:
                    return 0.0  # 特徵太少
                
                # 計算綜合評分
                quality_score = self._evaluate_feature_quality(
                    X[final_features], y, final_features, 
                    target_feature_count, correlation_threshold
                )
                
                return quality_score
                
            except Exception as e:
                print(f"Trial失敗: {e}")
                return 0.0
        
        # 執行Optuna搜索
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(features_objective, n_trials=n_trials, show_progress_bar=True)
            
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"✅ Features Optuna優化完成")
            print(f"🎯 最佳參數: {best_params}")
            print(f"📊 最佳分數: {best_score:.4f}")
            
            # 保存Features最佳參數
            self.save_optuna_params("features", best_params, best_score)
            
            return {
                "best_params": best_params,
                "best_score": best_score,
                "study": study
            }
            
        except Exception as e:
            print(f"❌ Features Optuna優化失敗: {e}")
            return {}
    
    def _calculate_feature_weights(self, feature_names: List[str], 
                                 technical_weight: float, volume_weight: float, 
                                 time_weight: float) -> Dict[str, float]:
        """計算特徵權重"""
        weights = {}
        
        for feature in feature_names:
            feature_lower = feature.lower()
            if any(word in feature_lower for word in ['rsi', 'macd', 'atr', 'adx', 'bb', 'cci']):
                weights[feature] = technical_weight
            elif any(word in feature_lower for word in ['volume', 'vol', 'obv', 'mfi']):
                weights[feature] = volume_weight
            elif any(word in feature_lower for word in ['hour', 'day', 'week', 'month', 'sin', 'cos']):
                weights[feature] = time_weight
            else:
                weights[feature] = 1.0  # 默認權重
        
        return weights
    
    def _select_features_lightgbm(self, X: pd.DataFrame, y: pd.Series, 
                                k: int, weights: Dict[str, float] = None) -> List[str]:
        """使用LightGBM選擇特徵"""
        try:
            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=3, random_state=42, verbose=-1
            )
            model.fit(X, y)
            
            # 獲取特徵重要性
            importance = model.feature_importances_
            
            # 應用權重
            if weights:
                weighted_importance = []
                for i, feature in enumerate(X.columns):
                    weight = weights.get(feature, 1.0)
                    weighted_importance.append(importance[i] * weight)
                importance = weighted_importance
            
            # 選擇top-k特徵
            feature_importance = list(zip(X.columns, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return [feat[0] for feat in feature_importance[:k]]
            
        except Exception as e:
            print(f"LightGBM特徵選擇失敗: {e}")
            return list(X.columns[:k])
    
    def _select_features_mutual_info(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """使用互信息選擇特徵"""
        try:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X, y)
            
            selected_mask = selector.get_support()
            return X.columns[selected_mask].tolist()
            
        except Exception as e:
            print(f"互信息特徵選擇失敗: {e}")
            return list(X.columns[:k])
    
    def _select_features_rfe(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """使用RFE選擇特徵"""
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            selector.fit(X, y)
            
            selected_mask = selector.get_support()
            return X.columns[selected_mask].tolist()
            
        except Exception as e:
            print(f"RFE特徵選擇失敗: {e}")
            return list(X.columns[:k])
    
    def _remove_correlated_features(self, X: pd.DataFrame, threshold: float) -> List[str]:
        """移除高相關性特徵"""
        try:
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            )
            
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > threshold)]
            
            return [col for col in X.columns if col not in to_drop]
            
        except Exception as e:
            print(f"去相關處理失敗: {e}")
            return list(X.columns)
    
    def _safe_time_series_split(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3, 
                               lag_periods: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """安全的時間序列分割，防止標籤數據洩漏"""
        total_samples = len(X)
        
        # 排除最後lag_periods個樣本，因為它們的標籤使用了未來數據
        safe_samples = total_samples - lag_periods
        
        if safe_samples <= 100:  # 數據太少
            return []
        
        splits = []
        fold_size = safe_samples // (n_splits + 1)
        
        for i in range(n_splits):
            # 訓練集：從開始到當前fold結束
            train_end = (i + 1) * fold_size
            train_idx = np.arange(0, train_end)
            
            # 測試集：下一個fold，但要確保不超過safe_samples
            test_start = train_end
            test_end = min(train_end + fold_size, safe_samples)
            
            if test_end - test_start < 50:  # 測試集太小
                continue
                
            test_idx = np.arange(test_start, test_end)
            
            # 驗證時間順序
            if len(train_idx) > 0 and len(test_idx) > 0 and max(train_idx) < min(test_idx):
                splits.append((train_idx, test_idx))
        
        return splits

    def _evaluate_feature_quality(self, X: pd.DataFrame, y: pd.Series, 
                                features: List[str], target_count: int, 
                                corr_threshold: float) -> float:
        """評估特徵質量綜合分數"""
        try:
            # 1. 數量效率分數 (接近目標數量得分更高)
            count_efficiency = 1.0 - abs(len(features) - target_count) / target_count
            
            # 2. 多樣性分數 (特徵類型越多樣越好)
            feature_types = self._categorize_features_simple(features)
            diversity_score = len(feature_types) / 6  # 最多6種類型
            
            # 3. 相關性分數 (低相關性更好)
            if len(features) > 1:
                corr_matrix = X.corr().abs()
                avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                correlation_score = 1.0 - min(avg_correlation, 0.9) / 0.9
            else:
                correlation_score = 1.0
            
            # 4. 預測能力分數 (嚴格時間序列驗證，確保無數據洩漏)
            try:
                # 確保數據有時間索引，如果沒有則按順序處理
                if not isinstance(X.index, pd.DatetimeIndex):
                    # 假設數據已按時間排序，重置索引確保順序
                    X_sample = X.reset_index(drop=True)
                    y_sample = y.reset_index(drop=True)
                else:
                    X_sample = X.sort_index()
                    y_sample = y.sort_index()
                
                # 使用合理樣本大小進行評估
                sample_size = min(5000, len(X_sample))
                if len(X_sample) > sample_size:
                    # 使用最近的數據進行評估，但保持時間順序
                    X_sample = X_sample.tail(sample_size)
                    y_sample = y_sample.tail(sample_size)
                
                # 使用安全的時間序列分割，防止標籤數據洩漏
                safe_splits = self._safe_time_series_split(X_sample, y_sample, n_splits=3, lag_periods=5)
                scores = []
                
                if len(safe_splits) == 0:
                    print("⚠️ 警告: 數據不足，無法進行安全的時間序列分割")
                    prediction_score = 0.0
                else:
                    for fold, (train_idx, test_idx) in enumerate(safe_splits):
                        X_train = X_sample.iloc[train_idx]
                        X_test = X_sample.iloc[test_idx]
                        y_train = y_sample.iloc[train_idx]
                        y_test = y_sample.iloc[test_idx]
                        
                        # 檢查標籤分佈
                        if len(y_train.unique()) > 1 and len(y_test.unique()) > 1:
                            # 使用輕量級模型快速評估
                            model = lgb.LGBMClassifier(
                                n_estimators=15,  
                                max_depth=4,      
                                learning_rate=0.1,
                                num_leaves=15,
                                subsample=0.8,
                                feature_fraction=0.8,
                                random_state=42,
                                verbose=-1,
                                force_col_wise=True
                            )
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            scores.append(score)
                            
                            # 調試信息（可選）
                            if fold == 0:  # 只在第一次打印
                                print(f"   安全時間序列驗證: 訓練期[{train_idx[0]}:{train_idx[-1]}] -> 測試期[{test_idx[0]}:{test_idx[-1]}], F1: {score:.4f}")
                    
                    prediction_score = np.mean(scores) if scores else 0.0
            except:
                prediction_score = 0.0
            
            # 綜合評分
            total_score = (
                count_efficiency * 0.2 +    # 數量效率
                diversity_score * 0.3 +     # 多樣性
                correlation_score * 0.2 +   # 低相關性
                prediction_score * 0.3      # 預測能力
            )
            
            return max(0.0, min(1.0, total_score))
            
        except Exception as e:
            print(f"特徵質量評估失敗: {e}")
            return 0.0
    
    def _categorize_features_simple(self, features: List[str]) -> Dict[str, int]:
        """簡單特徵分類"""
        categories = {"technical": 0, "volume": 0, "time": 0, "price": 0, "statistical": 0, "other": 0}
        
        for feature in features:
            feature_lower = feature.lower()
            if any(kw in feature_lower for kw in ['rsi', 'macd', 'atr', 'adx', 'bb', 'cci']):
                categories["technical"] += 1
            elif any(kw in feature_lower for kw in ['volume', 'vol', 'obv', 'mfi']):
                categories["volume"] += 1
            elif any(kw in feature_lower for kw in ['hour', 'day', 'week', 'month', 'sin', 'cos']):
                categories["time"] += 1
            elif any(kw in feature_lower for kw in ['open', 'high', 'low', 'close', 'price']):
                categories["price"] += 1
            elif any(kw in feature_lower for kw in ['return', 'std', 'mean', 'skew', 'kurt']):
                categories["statistical"] += 1
            else:
                categories["other"] += 1
        
        return {k: v for k, v in categories.items() if v > 0}
    
    def get_next_version(self) -> str:
        """自動獲取下一個版本號"""
        base_dirs = [
            f"data/processed/features/{self.symbol}_{self.timeframe}",
            f"data/processed/labels/{self.symbol}_{self.timeframe}",
            f"results/modular_optimization/{self.symbol}_{self.timeframe}"
        ]
        
        max_version = 0
        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    if item.startswith('v') and item[1:].isdigit():
                        version_num = int(item[1:])
                        max_version = max(max_version, version_num)
        
        return f"v{max_version + 1}"
    
    def list_versions(self) -> List[str]:
        """列出所有可用版本"""
        base_dir = f"results/modular_optimization/{self.symbol}_{self.timeframe}"
        versions = []
        
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                if item.startswith('v') and item[1:].isdigit():
                    versions.append(item)
        
        # 按版本號排序
        versions.sort(key=lambda x: int(x[1:]))
        return versions
    
    def get_latest_version(self) -> str:
        """獲取最新版本號"""
        versions = self.list_versions()
        return versions[-1] if versions else "v1"

    def setup_data_paths(self):
        """設置版本化數據文件路徑 - 🔧 修復版本一致性問題"""
        # 🔢 版本化路徑結構 - 確保特徵和標籤使用相同版本
        self.data_paths = {
            # ✅ 修復：特徵也使用版本化路徑，確保版本一致性
            "features": f"data/processed/features/{self.symbol}_{self.timeframe}/{self.version}/{self.symbol}_{self.timeframe}_features.parquet",
            
            # 🔄 保留無版本特徵作為回退選項
            "features_fallback": f"data/processed/features/{self.symbol}_{self.timeframe}_features.parquet",
            
            # 版本化路徑
            "version_base": {
                "features": f"data/processed/features/{self.symbol}_{self.timeframe}/{self.version}",
                "labels": f"data/processed/labels/{self.symbol}_{self.timeframe}/{self.version}",
                "results": f"results/modular_optimization/{self.symbol}_{self.timeframe}/{self.version}",
                "models": f"results/models/{self.symbol}_{self.timeframe}/{self.version}",
                "logs": f"logs/optimization/{self.symbol}_{self.timeframe}/{self.version}"
            },
            
            # 版本化文件路徑 - 確保三階段使用一致版本
            "labels": f"data/processed/labels/{self.symbol}_{self.timeframe}/{self.version}/{self.symbol}_{self.timeframe}_labels.parquet",
            "selected_features": f"data/processed/features/{self.symbol}_{self.timeframe}/{self.version}/{self.symbol}_{self.timeframe}_selected_features.parquet",
            
            # 原始數據路徑（不版本化）
            "raw_ohlcv": f"data/raw/{self.symbol}/{self.symbol}_{self.timeframe}.parquet",
            
            # 版本化結果目錄
            "results_dir": f"results/modular_optimization/{self.symbol}_{self.timeframe}/{self.version}",
            "models_dir": f"results/models/{self.symbol}_{self.timeframe}/{self.version}",
            "logs_dir": f"logs/optimization/{self.symbol}_{self.timeframe}/{self.version}"
        }
        
        # 創建版本化目錄
        for key, dir_path in self.data_paths["version_base"].items():
            os.makedirs(dir_path, exist_ok=True)
        
        # 創建基礎目錄
        base_dirs = [
            "data/processed/labels",
            "data/processed/features"
        ]
        for dir_path in base_dirs:
            os.makedirs(dir_path, exist_ok=True)

    def log_execution(self, stage: str, status: str, details: str = "", 
                     duration: float = 0.0):
        """記錄執行日誌"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "status": status,
            "details": details,
            "duration_seconds": duration,
            "symbol": self.symbol,
            "timeframe": self.timeframe
        }
        
        self.execution_log.append(log_entry)
        
        # 實時保存日誌
        log_file = os.path.join(self.data_paths["logs_dir"], "execution.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, indent=2, ensure_ascii=False, default=str)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """載入或生成特徵數據 - 🔧 版本化特徵支持"""
        print("📊 載入數據...")
        
        try:
            # 🔧 優先載入版本化特徵，保證版本一致性
            features_df = None
            
            # 方案1：載入版本化特徵（優先）
            if os.path.exists(self.data_paths["features"]):
                features_df = pd.read_parquet(self.data_paths["features"])
                print(f" ✅ 載入版本化特徵數據 {self.version}: {features_df.shape}")
            
            # 方案2：回退到無版本特徵（如果版本化不存在）
            elif os.path.exists(self.data_paths["features_fallback"]):
                features_df = pd.read_parquet(self.data_paths["features_fallback"])
                print(f" 🔄 載入回退特徵數據: {features_df.shape}")
                print(f" ⚠️  警告：使用無版本特徵，可能導致版本不一致問題")
                
                # 自動生成版本化特徵副本
                print(f" 🔧 正在為{self.version}創建版本化特徵副本...")
                versioned_features_path = self.data_paths["features"]
                os.makedirs(os.path.dirname(versioned_features_path), exist_ok=True)
                features_df.to_parquet(versioned_features_path)
                print(f" 💾 版本化特徵已保存: {versioned_features_path}")
            
            # 方案3：重新生成特徵
            else:
                print(" 🔧 特徵文件不存在，需要先生成特徵")
                features_df = self.generate_features_from_raw()
                if features_df is None:
                    return None, None
            
            # 載入原始OHLCV數據 
            if os.path.exists(self.data_paths["raw_ohlcv"]):
                ohlcv_df = pd.read_parquet(self.data_paths["raw_ohlcv"])
                print(f" ✅ 載入OHLCV數據: {ohlcv_df.shape}")
            else:
                raise FileNotFoundError(f"原始OHLCV數據不存在: {self.data_paths['raw_ohlcv']}")
            
            return features_df, ohlcv_df
            
        except Exception as e:
            error_msg = f"數據載入失敗: {e}"
            print(f" ❌ {error_msg}")
            self.log_execution("data_loading", "failed", error_msg)
            return None, None
    
    def generate_features_from_raw(self) -> pd.DataFrame:
        """🔧 生成版本化特徵系統 - 確保版本一致性"""
        try:
            print(" 🔧 檢查特徵文件...")
            
            # 🆕 策略：生成版本化特徵，確保與標籤版本一致
            print(f" 🚀 為{self.version}生成版本化特徵系統...")
            
            # 載入原始OHLCV數據
            raw_ohlcv_path = self.data_paths["raw_ohlcv"]
            if not os.path.exists(raw_ohlcv_path):
                print(f" ❌ 原始OHLCV數據不存在: {raw_ohlcv_path}")
                return None
                
            ohlcv_df = pd.read_parquet(raw_ohlcv_path)
            print(f" ✅ 載入OHLCV數據: {ohlcv_df.shape}")
            
            # 使用新的特徵工程系統
            from src.features.feature_generator import FeatureEngineering
            fe = FeatureEngineering(self.timeframe)
            features_df = fe.generate_all_features(ohlcv_df)
            
            print(f" 🆕 生成{self.version}特徵: {features_df.shape} ({len(features_df.columns)}個特徵)")
            
            # 保存新特徵到版本化路徑
            features_save_path = self.data_paths["features"]
            os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
            features_df.to_parquet(features_save_path)
            print(f" 💾 版本化特徵已保存: {features_save_path}")
            
            # 🔧 同時保存到無版本路徑作為回退
            fallback_path = self.data_paths["features_fallback"]
            if not os.path.exists(fallback_path):
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                features_df.to_parquet(fallback_path)
                print(f" 🔄 回退特徵已保存: {fallback_path}")
            
            return features_df
            
        except Exception as e:
            print(f" ❌ 版本化特徵生成失敗: {e}")
            return None

    def run_complete_optimization(self) -> Dict[str, Any]:
        """運行完整的三階段優化流程"""
        print("🚀 開始模組化Optuna三階段優化")
        print(f"交易對: {self.symbol}")
        print(f"時間框架: {self.timeframe}")
        print("="*80)
        
        total_start_time = datetime.now()
        
        try:
            # 載入數據
            features_df, ohlcv_df = self.load_data()
            if features_df is None or ohlcv_df is None:
                raise ValueError("數據載入失敗")
            
            # 第一階段：標籤優化
            label_results = self.run_stage1_label_optimization(features_df, ohlcv_df)
            if not label_results:
                raise ValueError("標籤優化失敗")

            # 📊 完整保存第一階段結果
            self.optimization_results["label_optimization"] = self._extract_label_results(label_results)
            labels = label_results["labels"]
            
            # 第二階段：特徵選擇
            feature_results = self.run_stage2_feature_selection(features_df, labels)
            if not feature_results:
                raise ValueError("特徵選擇失敗")

            # 📊 完整保存第二階段結果
            self.optimization_results["feature_selection"] = self._extract_feature_results(feature_results)
            selected_features = feature_results["best_features"]
            
            # 第三階段：模型優化
            model_results = self.run_stage3_model_optimization(features_df, labels, selected_features)
            if not model_results:
                raise ValueError("模型優化失敗")

            # 📊 完整保存第三階段結果
            self.optimization_results["model_optimization"] = self._extract_model_results(model_results)
            
            # 🔍 生成過度擬合分析
            overfitting_analysis = self._analyze_overfitting()
            self.optimization_results["overfitting_analysis"] = overfitting_analysis
            
            # 🆕 自動化一致性復核（最後階段）
            consistency_report = self._run_automated_consistency_check()
            self.optimization_results["consistency_check"] = consistency_report
            
            # 保存結果
            results_file = self.save_optimization_results()

            # 生成最終報告
            final_report = self.generate_final_report()
            print(final_report)

            total_duration = (datetime.now() - total_start_time).total_seconds()
            self.log_execution("complete_optimization", "success", 
                             f"完整優化成功，耗時 {total_duration:.1f} 秒", total_duration)
            
            return self.optimization_results
            
        except Exception as e:
            total_duration = (datetime.now() - total_start_time).total_seconds()
            error_msg = f"完整優化失敗: {e}"
            print(f"❌ {error_msg}")
            self.log_execution("complete_optimization", "failed", error_msg, total_duration)
            return {}

    def run_complete_training_with_optimized_params(self, use_full_data: bool = True) -> Dict[str, Any]:
        """🎯 使用優化後的參數進行完整模型訓練 - 正確的ML流程"""
        print("🚀 開始完整模型訓練（使用優化參數）")
        print(f"交易對: {self.symbol}")
        print(f"時間框架: {self.timeframe}")
        print(f"版本: {self.version}")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 🔍 第一步：載入已優化的參數
            print("\n📋 第一步：載入已優化的參數...")
            
            # 載入優化結果
            results_file = os.path.join(self.data_paths["results_dir"], "modular_optuna_results.json")
            if not os.path.exists(results_file):
                raise FileNotFoundError(f"優化結果不存在: {results_file}")
            
            with open(results_file, 'r', encoding='utf-8') as f:
                optimization_results = json.load(f)
            
            # 提取已優化的參數
            label_params = optimization_results["results"]["label_optimization"]["best_params"]
            feature_selection = optimization_results["results"]["feature_selection"]["best_features"]
            model_params = optimization_results["results"]["model_optimization"]["best_params"]
            
            print(f" ✅ 載入標籤參數: {label_params}")
            print(f" ✅ 載入特徵選擇: {len(feature_selection)}個特徵")
            print(f" ✅ 載入模型參數: {len(model_params)}個參數")
            
            # 🔍 第二步：載入版本化的特徵和標籤
            print("\n📊 第二步：載入版本化數據...")
            
            # 載入版本化特徵（v55特徵）
            if os.path.exists(self.data_paths["features"]):
                features_df = pd.read_parquet(self.data_paths["features"])
                print(f" ✅ 載入{self.version}特徵: {features_df.shape}")
            else:
                raise FileNotFoundError(f"版本化特徵不存在: {self.data_paths['features']}")
            
            # 載入版本化標籤（v55標籤）
            if os.path.exists(self.data_paths["labels"]):
                labels_df = pd.read_parquet(self.data_paths["labels"])
                labels = labels_df['label']  # 假設標籤列名為'label'
                print(f" ✅ 載入{self.version}標籤: {labels.shape}")
            else:
                raise FileNotFoundError(f"版本化標籤不存在: {self.data_paths['labels']}")
            
            # 🔍 第三步：使用優化後的特徵子集
            print("\n🎯 第三步：應用優化特徵選擇...")
            
            # 確保特徵存在
            missing_features = [f for f in feature_selection if f not in features_df.columns]
            if missing_features:
                print(f" ⚠️ 警告：缺少{len(missing_features)}個特徵: {missing_features[:5]}...")
                feature_selection = [f for f in feature_selection if f in features_df.columns]
            
            selected_features_df = features_df[feature_selection]
            print(f" ✅ 應用特徵選擇: {selected_features_df.shape}")
            
            # 🔍 第四步：數據對齊和清理
            print("\n🔧 第四步：數據對齊和清理...")
            
            min_length = min(len(selected_features_df), len(labels))
            X_final = selected_features_df.iloc[:min_length].copy()
            y_final = labels.iloc[:min_length].copy()
            
            # 移除NaN值
            valid_idx = ~(X_final.isnull().any(axis=1) | y_final.isnull())
            X_final = X_final[valid_idx]
            y_final = y_final[valid_idx]
            
            print(f" ✅ 最終訓練數據: {X_final.shape}, 標籤: {y_final.shape}")
            print(f" 📊 標籤分佈: {y_final.value_counts().to_dict()}")
            
            # 🔍 第五步：時序交叉驗證（確保模型穩健性）
            print("\n⏰ 第五步：時序交叉驗證...")
            
            # 配置LightGBM參數
            lgb_params = model_params.copy()
            
            # 根據標籤類別數調整目標函數
            n_classes = len(y_final.unique())
            if n_classes <= 2:
                lgb_params["objective"] = "binary"
                lgb_params["metric"] = "binary_logloss"
                lgb_params["num_class"] = None
            else:
                lgb_params["objective"] = "multiclass" 
                lgb_params["metric"] = "multi_logloss"
                lgb_params["num_class"] = n_classes
            
            # 添加固定參數
            lgb_params.update({
                "verbose": -1,
                "random_state": 42,
                "num_threads": 1,
                "force_col_wise": True
            })
            
            # 時序交叉驗證 - 確保模型在不同時期表現穩定
            cv_scores = self._run_time_series_cv(X_final, y_final, lgb_params, n_splits=5)
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            print(f" 📊 時序CV F1分數: {cv_scores}")
            print(f" 📊 平均CV F1: {mean_cv_score:.4f} (±{std_cv_score:.4f})")
            
            # 🔍 第六步：基於CV結果決定是否訓練最終模型
            print("\n🤖 第六步：訓練完整生產模型...")
            
            # 設定CV表現閾值（可根據具體情況調整）
            cv_threshold = 0.35  # F1分數閾值
            if mean_cv_score >= cv_threshold:
                print(f" ✅ CV表現達標 ({mean_cv_score:.4f} >= {cv_threshold})")
                
                # 創建並訓練最終模型
                final_model = lgb.LGBMClassifier(**lgb_params)
                
                if use_full_data:
                    # 使用全部數據訓練（生產模式）
                    print(" 🚀 使用全部數據訓練生產模型...")
                    final_model.fit(X_final, y_final)
                    training_mode = "全數據訓練"
                else:
                    # 保留測試集進行驗證
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
                    )
                    
                    final_model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        callbacks=[lgb.early_stopping(30)]
                    )
                    training_mode = "留出驗證集訓練"
                    
                    # 計算測試集性能
                    y_pred = final_model.predict(X_test)
                    if n_classes > 2:
                        test_f1 = f1_score(y_test, y_pred, average='weighted')
                    else:
                        test_f1 = f1_score(y_test, y_pred, average='binary')
                    print(f" 📊 測試集F1分數: {test_f1:.4f}")
            
            else:
                print(f" ❌ CV表現未達標 ({mean_cv_score:.4f} < {cv_threshold})")
                print(" ⚠️  建議重新調整參數或增加數據質量")
                return {
                    "status": "cv_failed",
                    "cv_scores": cv_scores,
                    "mean_cv_score": mean_cv_score,
                    "threshold": cv_threshold,
                    "message": "時序交叉驗證表現未達標，未生成最終模型"
                }
            
            # 🔍 第六步：保存完整模型
            print("\n💾 第六步：保存完整生產模型...")
            
            model_save_dir = os.path.join(self.data_paths["models_dir"], "production")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # 保存LightGBM模型
            model_file = os.path.join(model_save_dir, f"{self.symbol}_{self.timeframe}_{self.version}_production.txt")
            final_model.booster_.save_model(model_file)
            
            # 保存模型配置
            model_config = {
                "version": self.version,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "training_mode": training_mode,
                "training_data_shape": X_final.shape,
                "feature_count": len(feature_selection),
                "feature_names": feature_selection,
                "label_params": label_params,
                "model_params": lgb_params,
                "trained_at": datetime.now().isoformat(),
                "training_duration": (datetime.now() - start_time).total_seconds()
            }
            
            config_file = os.path.join(model_save_dir, f"{self.symbol}_{self.timeframe}_{self.version}_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存特徵重要性
            feature_importance = dict(zip(feature_selection, final_model.feature_importance()))
            importance_file = os.path.join(model_save_dir, f"{self.symbol}_{self.timeframe}_{self.version}_importance.json")
            with open(importance_file, 'w', encoding='utf-8') as f:
                json.dump(feature_importance, f, indent=2, ensure_ascii=False)
            
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # 🎯 生成訓練報告
            training_results = {
                "status": "success",
                "version": self.version,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "training_mode": training_mode,
                "training_duration": training_duration,
                "cross_validation": {
                    "cv_scores": cv_scores,
                    "mean_cv_score": mean_cv_score,
                    "std_cv_score": std_cv_score,
                    "cv_threshold": cv_threshold,
                    "cv_passed": mean_cv_score >= cv_threshold
                },
                "data_summary": {
                    "feature_count": len(feature_selection),
                    "sample_count": len(X_final),
                    "label_distribution": y_final.value_counts().to_dict()
                },
                "model_files": {
                    "model": model_file,
                    "config": config_file,
                    "importance": importance_file
                },
                "parameters_used": {
                    "label_params": label_params,
                    "selected_features": feature_selection,
                    "model_params": lgb_params
                }
            }
            
            if not use_full_data:
                training_results["test_performance"] = {
                    "f1_score": test_f1,
                    "test_samples": len(X_test)
                }
            
            print(f"\n✅ 完整訓練完成！")
            print(f"⏱️  訓練時間: {training_duration:.2f}秒")
            print(f"📁 模型已保存: {model_file}")
            print(f"📋 配置已保存: {config_file}")
            print("="*80)
            
            return training_results
            
        except Exception as e:
            error_msg = f"完整訓練失敗: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {}

    def _run_time_series_cv(self, X: pd.DataFrame, y: pd.Series, lgb_params: dict, n_splits: int = 5) -> List[float]:
        """執行時序交叉驗證 - 基於文檔設計"""
        from sklearn.model_selection import TimeSeriesSplit
        
        print(f" ⏰ 執行{n_splits}折時序交叉驗證...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"   🔄 第{fold}折驗證...")
            
            # 分割數據
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 檢查標籤分佈
            if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                print(f"     ⚠️ 第{fold}折標籤類別不足，跳過...")
                continue
            
            try:
                # 訓練模型
                model = lgb.LGBMClassifier(**lgb_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                )
                
                # 預測和評估
                y_val_pred = model.predict(X_val)
                
                # 計算F1分數
                n_classes = len(y.unique())
                if n_classes > 2:
                    fold_score = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
                else:
                    fold_score = f1_score(y_val, y_val_pred, average='binary', zero_division=0)
                
                cv_scores.append(fold_score)
                print(f"     ✅ 第{fold}折 F1分數: {fold_score:.4f}")
                
            except Exception as e:
                print(f"     ❌ 第{fold}折訓練失敗: {e}")
                continue
        
        if not cv_scores:
            print("   ❌ 所有折次都失敗，使用默認分數0.0")
            cv_scores = [0.0]
        
        return cv_scores

    def _evaluate_sharpe_ratio(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              returns: np.ndarray) -> float:
        """根據預測概率和真實收益計算年化Sharpe Ratio - 基於文檔設計"""
        try:
            # 基於預測概率生成交易信號
            signals = (y_pred_proba >= 0.5).astype(int) * 2 - 1  # 轉換為 -1, 1 信號
            
            # 計算策略收益（避免前瞻偏差）
            strategy_returns = signals[:-1] * returns[1:]
            
            if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
                return 0.0
            
            # 計算年化Sharpe比率（假設15分鐘數據）
            mean_return = np.mean(strategy_returns)
            std_return = np.std(strategy_returns)
            
            # 年化：252天 * 24小時 * 4個15分鐘
            annualization_factor = np.sqrt(252 * 24 * 4)
            sharpe_ratio = (mean_return / std_return) * annualization_factor
            
            return sharpe_ratio
            
        except Exception as e:
            print(f"   ⚠️ Sharpe比率計算失敗: {e}")
            return 0.0

    def run_model_optimization_only(self) -> Dict[str, Any]:
        """🎯 僅使用已有的v55特徵和標籤進行模型參數優化"""
        print("🚀 開始模型參數優化（使用已有v55數據）")
        print(f"交易對: {self.symbol}")
        print(f"時間框架: {self.timeframe}")
        print(f"版本: {self.version}")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 🔍 第一步：載入已有的v55特徵和標籤
            print("\n📊 第一步：載入已有的v55數據...")
            
            # 載入v55選中特徵
            if os.path.exists(self.data_paths["selected_features"]):
                features_df = pd.read_parquet(self.data_paths["selected_features"])
                print(f" ✅ 載入v55選中特徵: {features_df.shape}")
            else:
                raise FileNotFoundError(f"v55選中特徵不存在: {self.data_paths['selected_features']}")
            
            # 載入v55標籤
            if os.path.exists(self.data_paths["labels"]):
                labels_df = pd.read_parquet(self.data_paths["labels"])
                labels = labels_df['label']
                print(f" ✅ 載入v55標籤: {labels.shape}")
            else:
                raise FileNotFoundError(f"v55標籤不存在: {self.data_paths['labels']}")
            
            # 🔍 第二步：數據對齊和清理
            print("\n🔧 第二步：數據對齊和清理...")
            
            min_length = min(len(features_df), len(labels))
            features_aligned = features_df.iloc[:min_length].copy()
            labels_aligned = labels.iloc[:min_length].copy()
            
            # 移除NaN值
            valid_idx = ~(features_aligned.isnull().any(axis=1) | labels_aligned.isnull())
            X_final = features_aligned[valid_idx]
            y_final = labels_aligned[valid_idx]
            
            print(f" ✅ 最終數據: {X_final.shape}, 標籤: {y_final.shape}")
            print(f" 📊 標籤分佈: {y_final.value_counts().to_dict()}")
            
            # 🔍 第三步：模型參數優化
            print("\n🤖 第三步：執行模型參數優化...")
            
            model_results = self.model_optimizer.optimize(X_final, y_final, n_trials=50, fast_mode=False)
            
            if not model_results:
                raise ValueError("模型優化失敗")
            
            # 🔍 第四步：保存優化結果
            print("\n💾 第四步：保存模型參數...")
            
            # 保存最佳參數
            self.save_optuna_params('model', model_results['best_params'], model_results.get('best_score', 0.0))
            
            # 保存模型參數到版本化目錄
            model_params_path = os.path.join(self.data_paths["models_dir"], f"{self.symbol}_{self.timeframe}_model_params.json")
            os.makedirs(os.path.dirname(model_params_path), exist_ok=True)
            
            model_info = {
                "best_params": model_results.get('best_params', {}),
                "best_score": model_results.get('best_score', 0.0),
                "cv_results": model_results.get('cv_results', {}),
                "version": self.version,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "optimized_at": datetime.now().isoformat(),
                "optimization_duration": (datetime.now() - start_time).total_seconds()
            }
            
            with open(model_params_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\n✅ 模型參數優化完成！")
            print(f"⏱️  優化時間: {duration:.2f}秒")
            print(f"📊 最佳F1分數: {model_results.get('best_score', 0.0):.4f}")
            print(f"📁 參數已保存: {model_params_path}")
            print(f"🎯 最佳參數: {model_results.get('best_params', {})}")
            print("="*80)
            
            return {
                "status": "success",
                "version": self.version,
                "best_params": model_results.get('best_params', {}),
                "best_score": model_results.get('best_score', 0.0),
                "optimization_duration": duration,
                "data_info": {
                    "features_shape": X_final.shape,
                    "labels_shape": y_final.shape,
                    "label_distribution": y_final.value_counts().to_dict()
                },
                "files": {
                    "model_params": model_params_path,
                    "used_features": self.data_paths["selected_features"],
                    "used_labels": self.data_paths["labels"]
                }
            }
            
        except Exception as e:
            error_msg = f"模型參數優化失敗: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": error_msg}
    
    def run_stage1_label_optimization(self, features_df: pd.DataFrame, 
                                    ohlcv_df: pd.DataFrame) -> Dict[str, Any]:
        """執行第一階段：標籤優化"""
        print("\n" + "="*60)
        print("🎯 第一階段：標籤優化")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # 獲取價格數據
            price_data = ohlcv_df['close']
            
            # 執行標籤優化
            label_results = self.label_optimizer.optimize(features_df, price_data, n_trials=200)
            
            if label_results:
                duration = (datetime.now() - start_time).total_seconds()
                success_msg = f"標籤優化成功，最佳F1: {label_results.get('best_score', 0):.6f}"
                print(f"✅ {success_msg}")
                self.log_execution("label_optimization", "success", success_msg, duration)
                
                # 💾 保存標籤到版本化 Parquet 文件
                try:
                    labels = label_results["labels"]  # pandas Series
                    labels_path = self.data_paths["labels"]
                    
                    # 🔢 確保版本化目錄存在
                    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
                    
                    # 將 Series 轉換為 DataFrame 並保存
                    labels_df = labels.to_frame(name="label")
                    labels_df.to_parquet(labels_path)
                    print(f"💾 標籤已保存到: {labels_path}")
                    
                    # 同時保存標籤參數到同目錄
                    params_path = labels_path.replace('_labels.parquet', '_label_params.json')
                    with open(params_path, 'w', encoding='utf-8') as f:
                        json.dump(label_results.get('best_params', {}), f, indent=2, ensure_ascii=False)
                    print(f"💾 標籤參數已保存到: {params_path}")
                    
                except Exception as e:
                    print(f"⚠️ 標籤保存失敗: {e}")
                
                # 生成並打印報告
                report = self.label_optimizer.generate_report()
                print(report)
                
                return label_results
            else:
                raise ValueError("標籤優化返回空結果")
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"標籤優化失敗: {e}"
            print(f"❌ {error_msg}")
            self.log_execution("label_optimization", "failed", error_msg, duration)
            return {}
    
    def run_stage2_feature_selection(self, features_df: pd.DataFrame, 
                                   labels: pd.Series) -> Dict[str, Any]:
        """執行第二階段：兩階段特徵選擇 (基於學術文獻的4階段架構)"""
        print("\n" + "="*80)  
        print("🎯 第二階段：兩階段特徵選擇 (4階段架構)")
        print("="*80)
        
        # 🔧 關鍵修復：立即保存原始特徵信息，防止後續修改
        original_feature_count = len(features_df.columns)
        original_feature_names = list(features_df.columns)
        
        print(f"📊 時間框架: {self.timeframe} | 原始特徵數: {original_feature_count}")
        print(f"🔧 原始特徵數已安全保存: {original_feature_count}")
        
        total_start_time = datetime.now()
        
        try:
            # 數據對齊和預處理
            min_length = min(len(features_df), len(labels))
            features_aligned = features_df.iloc[:min_length]
            labels_aligned = labels.iloc[:min_length]
            
            print(f"📈 對齊後數據: 特徵{len(features_aligned.columns)}個，樣本{len(features_aligned)}個")
            print(f"📋 標籤分佈: {labels_aligned.value_counts().to_dict()}")
            
            # 🔍 第2A階段：粗選特徵候選池
            print(f"\n{'-'*60}")
            print("🔍 第2A階段：粗選特徵候選池 (快速篩選)")
            print(f"{'-'*60}")
            
            coarse_results = self._run_stage2a_coarse_selection(features_aligned, labels_aligned)
            if not coarse_results or not coarse_results.get("candidate_features"):
                raise ValueError("粗選階段失敗或未找到候選特徵")
                
            candidate_features = coarse_results["candidate_features"]
            print(f"✅ 第2A階段完成: 從{len(features_aligned.columns)}個特徵中粗選出{len(candidate_features)}個候選特徵")
            print(f"📊 候選特徵: {candidate_features[:10]}..." if len(candidate_features) > 10 else f"📊 候選特徵: {candidate_features}")
            
            # 🎯 第2B階段：精選最終特徵  
            print(f"\n{'-'*60}")
            print("🎯 第2B階段：精選最終特徵 (精細優化)")
            print(f"{'-'*60}")
            
            candidate_features_df = features_aligned[candidate_features]
            fine_results = self._run_stage2b_fine_selection(candidate_features_df, labels_aligned)
            if not fine_results or not fine_results.get("final_features"):
                raise ValueError("精選階段失敗或未找到最終特徵")
                
            final_features = fine_results["final_features"]
            print(f"✅ 第2B階段完成: 從{len(candidate_features)}個候選中精選出{len(final_features)}個最終特徵")
            print(f"🏆 最終特徵: {final_features}")
            
            # 📊 合併兩階段結果
            feature_results = {
                "stage2a_results": coarse_results,
                "stage2b_results": fine_results,
                "best_features": final_features,
                "best_score": fine_results.get("best_score", 0.0),
                "selection_pipeline": {
                    "original_count": original_feature_count,  # 🔧 關鍵修復：使用保存的原始數據
                    "cleaned_count": len(features_aligned.columns),  # 清洗後數量
                    "candidate_count": len(candidate_features), 
                    "final_count": len(final_features),
                    "coarse_ratio": len(candidate_features) / original_feature_count if original_feature_count > 0 else 0,
                    "fine_ratio": len(final_features) / len(candidate_features) if len(candidate_features) > 0 else 0,
                    "overall_ratio": len(final_features) / original_feature_count if original_feature_count > 0 else 0,
                    "pipeline": f"{original_feature_count} → {len(candidate_features)} → {len(final_features)}"
                }
            }
            
            if feature_results:
                total_duration = (datetime.now() - total_start_time).total_seconds()
                best_features = feature_results.get('best_features', [])
                pipeline_info = feature_results.get('selection_pipeline', {})
                
                success_msg = f"兩階段特徵選擇成功: {pipeline_info.get('pipeline', 'N/A')}"
                print(f"\n🎉 {success_msg}")
                print(f"⏰ 總耗時: {total_duration:.1f}秒")
                print(f"📊 最終效果: {len(best_features)}個高質量特徵")
                
                self.log_execution("feature_selection", "success", success_msg, total_duration)
                
                # 💾 保存選中的特徵到版本化 Parquet 文件
                try:
                    if best_features:
                        # 獲取選中的特徵數據
                        selected_features_df = features_aligned[best_features]
                        selected_features_path = self.data_paths["selected_features"]
                        
                        # 🔢 確保版本化目錄存在
                        os.makedirs(os.path.dirname(selected_features_path), exist_ok=True)
                        
                        # 保存選中的特徵數據
                        selected_features_df.to_parquet(selected_features_path)
                        print(f"💾 選中特徵數據已保存到: {selected_features_path}")
                        
                        # 保存特徵選擇參數和結果 - 🔧 修復特徵計數邏輯bug  
                        feature_params_path = selected_features_path.replace('_selected_features.parquet', '_feature_selection_params.json')
                        feature_selection_info = {
                            # 兩階段選擇結果
                            "stage2a_results": feature_results.get('stage2a_results', {}),
                            "stage2b_results": feature_results.get('stage2b_results', {}),
                            "best_features": best_features,
                            "best_score": feature_results.get('best_score', 0.0),
                            
                            # 🔧 完全修復：使用保存的原始特徵數
                            "original_feature_count": original_feature_count,  # ✅ 使用安全保存的原始數據
                            "original_feature_names": original_feature_names,  # ✅ 原始特徵名稱列表
                            "cleaned_feature_count": len(features_aligned.columns),  # 清洗後數量  
                            "selected_feature_count": len(best_features),
                            
                            # 完整的選擇管道信息
                            "selection_pipeline": feature_results.get('selection_pipeline', {}),
                            "selection_ratio": len(best_features) / original_feature_count if original_feature_count > 0 else 0,  # ✅ 使用安全計算
                            "method": "two_stage_selection_4stage_architecture"
                        }
                        
                        with open(feature_params_path, 'w', encoding='utf-8') as f:
                            json.dump(feature_selection_info, f, indent=2, ensure_ascii=False)
                        print(f"💾 特徵選擇信息已保存到: {feature_params_path}")
                        
                except Exception as e:
                    print(f"⚠️ 特徵保存失敗: {e}")
                
                # 生成並打印報告
                report = self.feature_selector.generate_report(features_df)
                print(report)
                
                return feature_results
            else:
                raise ValueError("特徵選擇返回空結果")
                
        except Exception as e:
            total_duration = (datetime.now() - total_start_time).total_seconds()
            error_msg = f"兩階段特徵選擇失敗: {e}"
            print(f"❌ {error_msg}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            self.log_execution("feature_selection", "failed", error_msg, total_duration)
            return {}
    
    def _run_stage2a_coarse_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """第2A階段：粗選特徵候選池 (快速篩選) - 基於HFT文獻"""
        try:
            print(f"🔍 開始粗選階段...")
            start_time = datetime.now()
            
            # 獲取時間框架配置
            feature_config = self.config.get_feature_config(self.timeframe)
            coarse_config = feature_config.get("two_stage_selection", {}).get("coarse_selection", {})
            
            # 粗選參數
            coarse_k_range = feature_config.get("coarse_k_range", (30, 50))
            target_k = min(coarse_k_range[1], len(X.columns))  # 不超過現有特徵數
            
            print(f"📊 粗選目標: 選出約{target_k}個候選特徵")
            
            # 🚀 使用LightGBM快速重要性選擇 (主要方法)
            candidate_features_lgb = self.feature_selector.select_features_lightgbm(
                X, y, k=target_k
            )
            
            if len(candidate_features_lgb) == 0:
                print("⚠️ LightGBM未選出任何特徵，使用前N個特徵作為後備")
                candidate_features_lgb = list(X.columns[:target_k])
            
            print(f"✅ LightGBM粗選: {len(candidate_features_lgb)}個特徵")
            
            # 🔄 互信息輔助驗證 (輔助方法，基於文獻推薦)
            if len(X.columns) > 50:  # 只在特徵較多時使用輔助驗證
                try:
                    candidate_features_mi = self.feature_selector.select_features_mutual_info(
                        X, y, k=target_k
                    )
                    
                    # 取交集和並集的平衡
                    intersection = set(candidate_features_lgb) & set(candidate_features_mi)
                    union = set(candidate_features_lgb) | set(candidate_features_mi)
                    
                    # 優先保留交集，然後按重要性補充
                    final_candidates = list(intersection)
                    remaining_slots = target_k - len(final_candidates)
                    
                    if remaining_slots > 0:
                        additional = [f for f in candidate_features_lgb if f not in final_candidates]
                        final_candidates.extend(additional[:remaining_slots])
                    
                    candidate_features = final_candidates[:target_k]
                    print(f"✅ 互信息驗證: 交集{len(intersection)}個, 最終{len(candidate_features)}個")
                    
                except Exception as e:
                    print(f"⚠️ 互信息輔助驗證失敗: {e}, 使用LightGBM結果")
                    candidate_features = candidate_features_lgb[:target_k]
            else:
                candidate_features = candidate_features_lgb[:target_k]
            
            # 📊 粗選結果統計
            duration = (datetime.now() - start_time).total_seconds()
            selection_ratio = len(candidate_features) / len(X.columns)
            
            print(f"🎯 粗選完成: {len(X.columns)} → {len(candidate_features)} 特徵")
            print(f"📈 選擇比例: {selection_ratio:.2%}, 耗時: {duration:.1f}秒")
            
            return {
                "candidate_features": candidate_features,
                "method": "lightgbm_primary_mutual_info_auxiliary",
                "selection_stats": {
                    "original_count": len(X.columns),
                    "selected_count": len(candidate_features),
                    "selection_ratio": selection_ratio,
                    "duration_seconds": duration
                },
                "stage": "coarse_selection"
            }
            
        except Exception as e:
            print(f"❌ 粗選階段失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            return {}
    
    def _run_stage2b_fine_selection(self, candidate_X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """第2B階段：精選最終特徵 (精細優化) - 基於量化交易最佳實踐"""
        try:
            print(f"🎯 開始精選階段...")
            start_time = datetime.now()
            
            # 獲取時間框架配置
            feature_config = self.config.get_feature_config(self.timeframe)
            fine_config = feature_config.get("two_stage_selection", {}).get("fine_selection", {})
            
            # 精選參數
            fine_k_range = feature_config.get("fine_k_range", (15, 25))
            fine_trials = feature_config.get("fine_n_trials", 150)
            
            print(f"📊 精選目標: {fine_k_range[0]}-{fine_k_range[1]}個最終特徵")
            print(f"🔄 優化試驗: {fine_trials}次")
            
            # 🎯 使用完整的Optuna優化進行精選
            # 臨時修改特徵選擇器配置為精選模式
            original_config = self.feature_selector.config.copy()
            
            # 設置精選模式配置
            self.feature_selector.config.update({
                "k_range": fine_k_range,
                "n_trials": fine_trials,
                "correlation_threshold": fine_config.get("correlation_threshold", 0.85),
                "feature_importance_threshold": 0.001  # 更嚴格的閾值
            })
            
            # 執行精細優化
            print("🔄 執行Optuna精細優化...")
            fine_results = self.feature_selector.optimize(candidate_X, y)
            
            # 恢復原始配置
            self.feature_selector.config = original_config
            
            if not fine_results or not fine_results.get("best_features"):
                # 後備方案：如果優化失敗，使用簡單的重要性排序
                print("⚠️ Optuna優化失敗，使用後備方案")
                backup_features = self.feature_selector.select_features_lightgbm(
                    candidate_X, y, k=fine_k_range[1]
                )
                fine_results = {
                    "best_features": backup_features,
                    "best_score": 0.5,  # 默認分數
                    "method": "backup_lightgbm"
                }
            
            final_features = fine_results["best_features"]
            best_score = fine_results.get("best_score", 0.0)
            
            # 📊 最終去相關處理（基於文獻建議）
            if len(final_features) > 1:
                final_features = self.feature_selector.remove_correlated_features(
                    candidate_X, final_features
                )
            
            # 📊 精選結果統計
            duration = (datetime.now() - start_time).total_seconds()
            selection_ratio = len(final_features) / len(candidate_X.columns)
            
            print(f"🎯 精選完成: {len(candidate_X.columns)} → {len(final_features)} 特徵")
            print(f"📈 最佳分數: {best_score:.6f}, 選擇比例: {selection_ratio:.2%}")
            print(f"⏰ 耗時: {duration:.1f}秒")
            
            # 🔍 特徵質量分析
            feature_quality = self._analyze_feature_quality(candidate_X, final_features, y)
            
            return {
                "final_features": final_features,
                "best_score": best_score,
                "method": "optuna_combined_optimization",
                "selection_stats": {
                    "candidate_count": len(candidate_X.columns),
                    "selected_count": len(final_features),
                    "selection_ratio": selection_ratio,
                    "duration_seconds": duration
                },
                "feature_quality": feature_quality,
                "stage": "fine_selection"
            }
            
        except Exception as e:
            print(f"❌ 精選階段失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            return {}
    
    def _analyze_feature_quality(self, X: pd.DataFrame, features: List[str], y: pd.Series) -> Dict[str, Any]:
        """分析最終特徵質量 - 基於金融ML文獻"""
        try:
            if not features:
                return {}
            
            X_selected = X[features]
            
            # 特徵間相關性分析
            feature_corr = X_selected.corr().abs()
            avg_correlation = feature_corr.values[np.triu_indices_from(feature_corr.values, k=1)].mean()
            max_correlation = feature_corr.values[np.triu_indices_from(feature_corr.values, k=1)].max()
            
            # 特徵穩定性分析（方差）
            feature_stability = X_selected.var().mean()
            
            # 特徵多樣性分析
            feature_categories = self._categorize_features(features)
            
            quality_metrics = {
                "avg_inter_correlation": avg_correlation,
                "max_inter_correlation": max_correlation,
                "feature_stability": feature_stability,
                "feature_diversity": len(feature_categories),
                "feature_categories": feature_categories,
                "quality_score": self._calculate_quality_score(avg_correlation, max_correlation, len(feature_categories))
            }
            
            print(f"📊 特徵質量分析:")
            print(f"   平均相關性: {avg_correlation:.3f}")
            print(f"   最大相關性: {max_correlation:.3f}")
            print(f"   特徵多樣性: {len(feature_categories)}類")
            print(f"   質量評分: {quality_metrics['quality_score']:.3f}")
            
            return quality_metrics
            
        except Exception as e:
            print(f"⚠️ 特徵質量分析失敗: {e}")
            return {}
    
    def _categorize_features(self, features: List[str]) -> Dict[str, int]:
        """特徵分類統計 - 基於量化交易特徵分類"""
        categories = {
            "price": 0, "volume": 0, "technical": 0, "momentum": 0,
            "volatility": 0, "time": 0, "statistical": 0, "other": 0
        }
        
        for feature in features:
            feature_lower = feature.lower()
            if any(kw in feature_lower for kw in ["price", "open", "high", "low", "close", "ma", "ema"]):
                categories["price"] += 1
            elif any(kw in feature_lower for kw in ["volume", "vol"]):
                categories["volume"] += 1
            elif any(kw in feature_lower for kw in ["rsi", "macd", "adx", "atr", "bb", "kdj"]):
                categories["technical"] += 1
            elif any(kw in feature_lower for kw in ["roc", "momentum", "mom"]):
                categories["momentum"] += 1
            elif any(kw in feature_lower for kw in ["std", "var", "volatility", "vol"]):
                categories["volatility"] += 1
            elif any(kw in feature_lower for kw in ["hour", "day", "week", "month", "time"]):
                categories["time"] += 1
            elif any(kw in feature_lower for kw in ["mean", "median", "skew", "kurt"]):
                categories["statistical"] += 1
            else:
                categories["other"] += 1
        
        return {k: v for k, v in categories.items() if v > 0}
    
    def _calculate_quality_score(self, avg_corr: float, max_corr: float, diversity: int) -> float:
        """計算特徵質量評分 (0-1之間，越高越好)"""
        # 相關性懲罰 (相關性越低越好)
        corr_penalty = 1.0 - min(avg_corr, 0.8) / 0.8
        
        # 多樣性獎勵 (分類越多越好，最多8類)
        diversity_reward = min(diversity, 8) / 8
        
        # 最大相關性懲罰 (避免極度相關的特徵對)
        max_corr_penalty = 1.0 - min(max_corr, 0.95) / 0.95
        
        # 綜合評分
        quality_score = (corr_penalty * 0.4 + diversity_reward * 0.3 + max_corr_penalty * 0.3)
        return max(0.0, min(1.0, quality_score))
    
    def run_stage3_model_optimization(self, features_df: pd.DataFrame, 
                                    labels: pd.Series,
                                    selected_features: List[str]) -> Dict[str, Any]:
        """執行第三階段：模型超參數優化"""
        print("\n" + "="*60)
        print("🎯 第三階段：模型超參數優化")
        print("="*60)
        
        start_time = datetime.now()
        
        try:
            # 準備數據
            X_selected = features_df[selected_features]
            min_length = min(len(X_selected), len(labels))
            X_aligned = X_selected.iloc[:min_length]
            y_aligned = labels.iloc[:min_length]
            
            # 執行模型優化
            model_results = self.model_optimizer.optimize(X_aligned, y_aligned)
            
            if model_results:
                duration = (datetime.now() - start_time).total_seconds()
                success_msg = f"模型優化成功，最佳F1: {model_results.get('best_score', 0):.6f}"
                print(f"✅ {success_msg}")
                self.log_execution("model_optimization", "success", success_msg, duration)
                
                # 💾 保存模型參數到版本化目錄
                try:
                    model_params_path = os.path.join(self.data_paths["models_dir"], f"{self.symbol}_{self.timeframe}_model_params.json")
                    
                    # 🔢 確保版本化模型目錄存在
                    os.makedirs(os.path.dirname(model_params_path), exist_ok=True)
                    
                    model_info = {
                        "best_params": model_results.get('best_params', {}),
                        "best_score": model_results.get('best_score', 0.0),
                        "cv_results": model_results.get('cv_results', {}),
                        "version": self.version,
                        "symbol": self.symbol,
                        "timeframe": self.timeframe
                    }
                    
                    with open(model_params_path, 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
                    print(f"💾 模型參數已保存到: {model_params_path}")
                    
                except Exception as e:
                    print(f"⚠️ 模型參數保存失敗: {e}")
                
                # 生成並打印報告
                report = self.model_optimizer.generate_report()
                print(report)
                
                return model_results
            else:
                raise ValueError("模型優化返回空結果")

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"模型優化失敗: {e}"
            print(f"❌ {error_msg}")
            self.log_execution("model_optimization", "failed", error_msg, duration)
            return {}

    def save_optimization_results(self) -> str:
        """保存優化結果到版本化目錄"""
        try:
            # 準備完整結果
            complete_results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "version": self.version,  # 🔢 添加版本信息
                    "optimization_strategy": "Modular Optuna Optimization",
                    "system_version": "2.1"
                },
                "results": self.optimization_results.copy(),
                "execution_log": self.execution_log,
                "summary": self.generate_optimization_summary()
            }
            
            # 清理不可序列化的對象
            self.clean_results_for_json(complete_results["results"])
            
            # 🔢 確保版本化結果目錄存在
            os.makedirs(self.data_paths["results_dir"], exist_ok=True)
            
            # 保存JSON結果到版本化目錄
            results_file = os.path.join(self.data_paths["results_dir"], "modular_optuna_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 優化結果已保存到: {results_file}")
            
            # 📋 生成版本摘要文件
            version_summary_file = os.path.join(self.data_paths["results_dir"], "version_summary.json")
            version_summary = {
                "version": self.version,
                "created_at": datetime.now().isoformat(),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "performance": {
                    "final_f1_score": complete_results["results"].get("model_optimization", {}).get("best_score", 0),
                    "selected_features_count": len(complete_results["results"].get("feature_selection", {}).get("best_features", [])),
                    "overfitting_risk": complete_results["results"].get("overfitting_analysis", {}).get("overall_risk_level", "UNKNOWN")
                },
                "file_paths": {
                    "labels": self.data_paths["labels"],
                    "selected_features": self.data_paths["selected_features"], 
                    "results": results_file,
                    "models": self.data_paths["models_dir"]
                }
            }
            
            with open(version_summary_file, 'w', encoding='utf-8') as f:
                json.dump(version_summary, f, indent=2, ensure_ascii=False)
            print(f"📋 版本摘要已保存到: {version_summary_file}")
            
            return results_file
            
        except Exception as e:
            print(f"❌ 結果保存失敗: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _extract_label_results(self, label_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取並格式化標籤優化結果"""
        try:
            extracted = {
                "best_params": label_results.get("best_params", {}),
                "best_score": label_results.get("best_score", 0.0),
                "labels": self._safe_serialize_series(label_results.get("labels")),
                "n_pareto_solutions": label_results.get("n_pareto_solutions", 0),
                "cv_details": {
                    "pareto_solutions_count": label_results.get("n_pareto_solutions", 0),
                    "optimization_direction": ["maximize", "maximize"],  # F1 + Stability
                    "sampler_type": "NSGA-II"
                }
            }
            
            # 如果有Study對象，提取試驗歷史
            if "study" in label_results:
                study = label_results["study"]
                if hasattr(study, 'trials'):
                    extracted["trial_history"] = [
                        {
                            "number": trial.number,
                            "values": trial.values if trial.values else [0.0, 0.0],
                            "params": trial.params,
                            "state": trial.state.name if hasattr(trial.state, 'name') else 'COMPLETE'
                        }
                        for trial in study.trials[-20:]  # 保存最後20次trial
                    ]
            
            return extracted
            
        except Exception as e:
            print(f"⚠️ 標籤結果提取失敗: {e}")
            return label_results

    def _extract_feature_results(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取並格式化特徵選擇結果"""
        try:
            # 🔧 修复：确保兼容性，使用正确的键名
            original_count = feature_results.get("original_feature_count", 0)
            selected_count = len(feature_results.get("best_features", []))
            
            extracted = {
                "best_params": feature_results.get("best_params", {}),
                "best_features": feature_results.get("best_features", []),
                "best_score": feature_results.get("best_score", 0.0),
                "feature_importance": self._safe_serialize_dataframe(
                    feature_results.get("feature_importance")
                ),
                # 🔧 使用优先读取的键名 selection_pipeline
                "selection_pipeline": {
                    "original_count": original_count,
                    "final_count": selected_count,
                    "overall_ratio": selected_count / max(original_count, 1),
                    "pipeline": feature_results.get("selection_pipeline", {}).get("pipeline", "two_stage_selection"),
                    "stage2a_results": feature_results.get("stage2a_results", {}),
                    "stage2b_results": feature_results.get("stage2b_results", {})
                },
                # 🔧 保持向后兼容
                "selection_details": {
                    "original_feature_count": original_count,
                    "selected_feature_count": selected_count,
                    "selection_ratio": selected_count / max(original_count, 1),
                    "selection_method": feature_results.get("best_params", {}).get("selector_method", "unknown")
                }
            }
            
            # 提取試驗歷史
            if "study" in feature_results:
                study = feature_results["study"]
                if hasattr(study, 'trials'):
                    extracted["trial_history"] = [
                        {
                            "number": trial.number,
                            "value": trial.value if trial.value else 0.0,
                            "params": trial.params,
                            "state": trial.state.name if hasattr(trial.state, 'name') else 'COMPLETE'
                        }
                        for trial in study.trials[-20:]  # 保存最後20次trial
                    ]
            
            return extracted
            
        except Exception as e:
            print(f"⚠️ 特徵結果提取失敗: {e}")
            return feature_results

    def _extract_model_results(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取並格式化模型優化結果"""
        try:
            extracted = {
                "best_params": model_results.get("best_params", {}),
                "best_score": model_results.get("best_score", 0.0),
                "cv_results": model_results.get("cv_results", {}),
                "model_details": {
                    "model_type": "LightGBM",
                    "optimization_metric": "F1-Score",
                    "cv_folds": 5,
                    "cv_method": "TimeSeriesSplit"
                }
            }
            
            # 🔍 提取CV詳細信息用於過度擬合檢測
            cv_results = model_results.get("cv_results", {})
            if cv_results:
                fold_scores = cv_results.get("fold_scores", [])
                extracted["cv_detailed"] = {
                    "fold_scores": fold_scores,
                    "cv_mean": np.mean(fold_scores) if fold_scores else 0.0,
                    "cv_std": np.std(fold_scores) if fold_scores else 0.0,
                    "cv_min": np.min(fold_scores) if fold_scores else 0.0,
                    "cv_max": np.max(fold_scores) if fold_scores else 0.0,
                    "fold_score_range": np.max(fold_scores) - np.min(fold_scores) if fold_scores else 0.0
                }
            
            # 提取試驗歷史和收斂信息
            if "study" in model_results:
                study = model_results["study"]
                if hasattr(study, 'trials'):
                    trial_values = [t.value for t in study.trials if t.value is not None]
                    extracted["convergence_analysis"] = {
                        "total_trials": len(study.trials),
                        "successful_trials": len(trial_values),
                        "best_trial_number": study.best_trial.number if hasattr(study, 'best_trial') else -1,
                        "score_progression": trial_values[-10:] if len(trial_values) >= 10 else trial_values,
                        "early_convergence": len(trial_values) < len(study.trials) * 0.5  # 如果超過一半被剪枝
                    }
                    
                    extracted["trial_history"] = [
                        {
                            "number": trial.number,
                            "value": trial.value if trial.value else 0.0,
                            "params": trial.params,
                            "state": trial.state.name if hasattr(trial.state, 'name') else 'COMPLETE'
                        }
                        for trial in study.trials[-20:]  # 保存最後20次trial
                    ]
            
            return extracted
            
        except Exception as e:
            print(f"⚠️ 模型結果提取失敗: {e}")
            return model_results

    def _analyze_overfitting(self) -> Dict[str, Any]:
        """增強過度擬合風險分析 - 🔧 修復檢測失效問題，基於量化交易風控標準"""
        overfitting_analysis = {
            "overall_risk_level": "LOW",
            "risk_factors": [],
            "recommendations": [],
            "detailed_metrics": {}
        }
        
        try:
            print("🔍 執行增強過度擬合分析...")
            
            # 1. 檢查標籤優化階段
            if "label_optimization" in self.optimization_results:
                label_data = self.optimization_results["label_optimization"]
                
                # 📊 標籤分數異常檢查 (基於業界標準修正)
                label_score = label_data.get("best_score", 0.0)
                if label_score > 0.95:  # 🔧 修正：F1分數>95%才視為高度懷疑過度擬合
                    overfitting_analysis["risk_factors"].append(
                        f"標籤優化F1分數極高({label_score:.3f} > 0.95)，高度懷疑過度擬合"
                    )
                    overfitting_analysis["overall_risk_level"] = "HIGH"
                elif label_score > 0.85:  # F1 85-95%為極優秀表現，需要驗證但不直接判定過擬合
                    overfitting_analysis["risk_factors"].append(
                        f"標籤優化F1分數極優秀({label_score:.3f} > 0.85)，建議進一步驗證"
                    )
                    overfitting_analysis["overall_risk_level"] = max(
                        overfitting_analysis["overall_risk_level"], "MEDIUM"
                    )
                
                # Pareto解數量檢查
                n_pareto = label_data.get("n_pareto_solutions", 0)
                if n_pareto <= 1:
                    overfitting_analysis["risk_factors"].append(
                        "標籤優化僅產生1個Pareto解，可能過度擬合特定時間段"
                    )
                    overfitting_analysis["overall_risk_level"] = max(
                        overfitting_analysis["overall_risk_level"], "MEDIUM"
                    )
            
                overfitting_analysis["detailed_metrics"]["label_score"] = label_score
                overfitting_analysis["detailed_metrics"]["n_pareto_solutions"] = n_pareto
            
            # 2. 檢查特徵選擇階段 - 🔧 增強檢測邏輯
            if "feature_selection" in self.optimization_results:
                feature_data = self.optimization_results["feature_selection"]
                
                # 處理兩種數據結構 (兼容舊版和新版)
                if "selection_pipeline" in feature_data:  # 新版4階段架構
                    pipeline = feature_data["selection_pipeline"]
                    original_count = pipeline.get("original_count", 0)
                    selected_count = pipeline.get("final_count", 0)
                    selection_ratio = pipeline.get("overall_ratio", 0.0)
                else:  # 舊版結構
                    selection_details = feature_data.get("selection_details", {})
                    original_count = selection_details.get("original_feature_count", 0)
                    selected_count = selection_details.get("selected_feature_count", 0)
                    selection_ratio = selection_details.get("selection_ratio", 0.0)
                
                # 🔧 修復：檢查數據完整性
                if original_count == 0:
                    overfitting_analysis["risk_factors"].append(
                        "特徵計數異常(原始特徵數=0)，可能存在數據流問題"
                    )
                    overfitting_analysis["overall_risk_level"] = "HIGH"
                elif selected_count == 0:
                    overfitting_analysis["risk_factors"].append(
                        "特徵選擇失敗(選中特徵數=0)，系統存在嚴重問題"
                    )
                    overfitting_analysis["overall_risk_level"] = "HIGH"
                else:
                    # 特徵數量檢查 (基於量化交易風控標準)
                    if selected_count == 1:
                        overfitting_analysis["risk_factors"].append(
                            f"極度特徵壓縮(僅{selected_count}個特徵)，嚴重過度擬合風險"
                        )
                        overfitting_analysis["overall_risk_level"] = "CRITICAL"  # 新增危急等級
                    elif selected_count < 5:
                        overfitting_analysis["risk_factors"].append(
                            f"特徵數過少({selected_count}個)，缺乏足夠信息多樣性"
                        )
                        overfitting_analysis["overall_risk_level"] = "HIGH"
                    elif selection_ratio < 0.03:  # 選中比例<3%
                        overfitting_analysis["risk_factors"].append(
                            f"特徵選擇過度壓縮(僅保留{selection_ratio:.1%})，資訊損失嚴重"
                        )
                        overfitting_analysis["overall_risk_level"] = "HIGH"
                    elif selected_count < 10:
                        overfitting_analysis["risk_factors"].append(
                            f"特徵數偏少({selected_count}個)，建議增加到15-25個"
                        )
                        overfitting_analysis["overall_risk_level"] = "MEDIUM"
                
                # 記錄詳細指標 - 🔧 增強調試信息
                overfitting_analysis["detailed_metrics"].update({
                    "original_feature_count": original_count,
                    "selected_feature_count": selected_count,
                    "selection_ratio": selection_ratio,
                    "feature_count_debug": {
                        "has_pipeline": "selection_pipeline" in feature_data,
                        "has_legacy": "selection_details" in feature_data,
                        "data_source": "pipeline" if "selection_pipeline" in feature_data else "legacy"
                    }
                })
                
                # 🆕 特徵重要性穩定性分析檢查
                feature_stability = feature_data.get("feature_importance_stability", {})
                if feature_stability:
                    stability_report = feature_stability.get("stability_report", {})
                    if stability_report:
                        stability_summary = stability_report.get("summary", {})
                        stability_score = stability_summary.get("overall_stability_score", 0.0)
                        stability_grade = stability_summary.get("overall_stability_grade", "N/A")
                        meets_threshold = stability_summary.get("meets_minimum_threshold", False)
                        
                        # 穩定性風險檢查
                        if stability_score < 0.3:
                            overfitting_analysis["risk_factors"].append(
                                f"特徵重要性極不穩定(穩定性:{stability_score:.2f} < 0.3)，嚴重過擬合風險"
                            )
                            overfitting_analysis["overall_risk_level"] = "CRITICAL"
                        elif stability_score < 0.5:
                            overfitting_analysis["risk_factors"].append(
                                f"特徵重要性不穩定(穩定性:{stability_score:.2f} < 0.5)，模型可能過擬合"
                            )
                            overfitting_analysis["overall_risk_level"] = max(
                                overfitting_analysis["overall_risk_level"], "HIGH"
                            )
                        elif stability_score < 0.6 and selected_count > 20:
                            overfitting_analysis["risk_factors"].append(
                                f"特徵較多且穩定性一般(穩定性:{stability_score:.2f}, 特徵數:{selected_count})，建議減少特徵"
                            )
                            overfitting_analysis["overall_risk_level"] = max(
                                overfitting_analysis["overall_risk_level"], "MEDIUM"
                            )
                        
                        # 穩定性建議
                        recommendations = stability_report.get("recommendations", [])
                        if recommendations:
                            overfitting_analysis["recommendations"].extend([
                                f"特徵穩定性建議: {rec}" for rec in recommendations
                            ])
                        
                        # 記錄穩定性指標
                        overfitting_analysis["detailed_metrics"]["feature_stability"] = {
                            "overall_score": stability_score,
                            "grade": stability_grade,
                            "meets_threshold": meets_threshold,
                            "num_windows_analyzed": stability_summary.get("num_windows_analyzed", 0)
                        }
                    else:
                        overfitting_analysis["risk_factors"].append(
                            "特徵重要性穩定性分析未完成，無法評估時間一致性"
                        )
                        overfitting_analysis["overall_risk_level"] = max(
                            overfitting_analysis["overall_risk_level"], "MEDIUM"
                        )
            
            # 3. 檢查模型優化階段 - 🔧 增強性能檢測
            if "model_optimization" in self.optimization_results:
                model_data = self.optimization_results["model_optimization"]
                model_score = model_data.get("best_score", 0.0)
                
                # 📊 模型分數異常檢查 (基於業界標準修正)
                if model_score > 0.95:  # F1>95%高度懷疑過度擬合
                    overfitting_analysis["risk_factors"].append(
                        f"模型F1分數極高({model_score:.3f} > 0.95)，高度懷疑過度擬合"
                    )
                    overfitting_analysis["overall_risk_level"] = "CRITICAL"  # 新增危急等級
                elif model_score > 0.85:  # F1 85-95%為極優秀表現，需要多維度驗證
                    overfitting_analysis["risk_factors"].append(
                        f"模型F1分數極優秀({model_score:.3f} > 0.85)，建議多維度驗證"
                    )
                    overfitting_analysis["overall_risk_level"] = max(
                        overfitting_analysis["overall_risk_level"], "MEDIUM"
                    )
                elif model_score > 0.75 and self.timeframe in ["5m", "15m"]:  # 短時框更嚴格但合理調整
                    overfitting_analysis["risk_factors"].append(
                        f"短時框F1分數優秀({model_score:.3f} > 0.75)，建議額外驗證"
                    )
                    overfitting_analysis["overall_risk_level"] = max(
                        overfitting_analysis["overall_risk_level"], "LOW"
                    )
                
                # CV詳細分析
                cv_detailed = model_data.get("cv_detailed", {})
                if cv_detailed:
                    cv_std = cv_detailed.get("cv_std", 0.0)
                    cv_max = cv_detailed.get("cv_max", 0.0)
                    cv_min = cv_detailed.get("cv_min", 0.0)
                    fold_score_range = cv_detailed.get("fold_score_range", 0.0)
                    
                    # CV標準差檢查 (基於業界標準修正)
                    if cv_std > 0.05:  # 🔧 修正：標準差>5%為高風險 (業界推薦)
                        overfitting_analysis["risk_factors"].append(
                            f"CV折間差異過大(std={cv_std:.4f} > 0.05)，模型不穩定，可能過度擬合"
                        )
                        overfitting_analysis["overall_risk_level"] = "HIGH"
                    elif cv_std > 0.03:  # 標準差>3%為中等風險
                        overfitting_analysis["risk_factors"].append(
                            f"CV折間差異較大(std={cv_std:.4f} > 0.03)，建議增加樣本或調整模型"
                        )
                        overfitting_analysis["overall_risk_level"] = max(
                            overfitting_analysis["overall_risk_level"], "MEDIUM"
                        )
                    
                    # 🔧 新增：訓練/驗證差距檢查 (業界推薦多維度檢測)
                    if cv_max > 0 and cv_min > 0:
                        performance_gap = (cv_max - cv_min) / cv_min
                        if performance_gap > 0.15:  # 差距>15%為風險
                            overfitting_analysis["risk_factors"].append(
                                f"不同時期表現差距過大({performance_gap:.1%})，模型泛化能力不足"
                            )
                            overfitting_analysis["overall_risk_level"] = max(
                                overfitting_analysis["overall_risk_level"], "MEDIUM"
                            )
                    
                    # CV範圍檢查
                    if fold_score_range > 0.15:  # 最高最低差>15%
                        overfitting_analysis["risk_factors"].append(
                            f"CV折間分數範圍過大({fold_score_range:.3f})，不同數據段表現差異極大"
                        )
                        overfitting_analysis["overall_risk_level"] = "HIGH"
                    
                    overfitting_analysis["detailed_metrics"].update({
                        "cv_std": cv_std,
                        "cv_range": fold_score_range,
                        "cv_max": cv_max,
                        "cv_min": cv_min
                    })
                
                # 模型複雜度檢查
                best_params = model_data.get("best_params", {})
                n_estimators = best_params.get("n_estimators", 0)
                max_depth = best_params.get("max_depth", 0)
                num_leaves = best_params.get("num_leaves", 0)
                
                # 檢查模型是否過於複雜
                if n_estimators > 500 and num_leaves > 60:
                    overfitting_analysis["risk_factors"].append(
                        f"模型過度複雜(trees={n_estimators}, leaves={num_leaves})，容易過擬合"
                    )
                    overfitting_analysis["overall_risk_level"] = max(
                        overfitting_analysis["overall_risk_level"], "MEDIUM"
                    )
                
                overfitting_analysis["detailed_metrics"].update({
                    "model_score": model_score,
                    "model_complexity": {"n_estimators": n_estimators, "max_depth": max_depth, "num_leaves": num_leaves}
                })
                
                # 收斂檢查
                convergence = model_data.get("convergence_analysis", {})
                early_convergence = convergence.get("early_convergence", False)
                if early_convergence:
                    overfitting_analysis["risk_factors"].append(
                        "模型優化提前收斂，可能陷入局部最優"
                    )
            
                # 🆕 CV vs WFA 一致性檢查
                trial_results = model_data.get("trial_results", {})
                if trial_results:
                    best_trial_num = model_data.get("best_trial_number", None)
                    if best_trial_num and str(best_trial_num) in trial_results:
                        best_trial_data = trial_results[str(best_trial_num)]
                        cv_vs_oos_analysis = best_trial_data.get("cv_vs_oos_analysis", {})
                        
                        if cv_vs_oos_analysis:
                            oos_delta_pct = cv_vs_oos_analysis.get("delta_pct", 0.0)
                            consistency_status = cv_vs_oos_analysis.get("consistency_status", "未知")
                            cv_mean = cv_vs_oos_analysis.get("cv_mean", 0.0)
                            oos_mean = cv_vs_oos_analysis.get("oos_mean", 0.0)
                            
                            # CV vs WFA 一致性風險檢查
                            if oos_delta_pct > 0.20:  # 差異>20%為嚴重不一致
                                overfitting_analysis["risk_factors"].append(
                                    f"CV與樣本外驗證嚴重不一致(差異:{oos_delta_pct:.1%} > 20%)，高度懷疑過擬合"
                                )
                                overfitting_analysis["overall_risk_level"] = "CRITICAL"
                            elif oos_delta_pct > 0.15:  # 差異>15%為中等不一致
                                overfitting_analysis["risk_factors"].append(
                                    f"CV與樣本外驗證不一致(差異:{oos_delta_pct:.1%} > 15%)，可能過擬合"
                                )
                                overfitting_analysis["overall_risk_level"] = max(
                                    overfitting_analysis["overall_risk_level"], "HIGH"
                                )
                            elif oos_delta_pct > 0.10:  # 差異>10%為輕微不一致
                                overfitting_analysis["risk_factors"].append(
                                    f"CV與樣本外驗證輕微不一致(差異:{oos_delta_pct:.1%} > 10%)，需要關注"
                                )
                                overfitting_analysis["overall_risk_level"] = max(
                                    overfitting_analysis["overall_risk_level"], "MEDIUM"
                                )
                            
                            # 記錄CV vs WFA指標
                            overfitting_analysis["detailed_metrics"]["cv_vs_oos_consistency"] = {
                                "delta_pct": oos_delta_pct,
                                "consistency_status": consistency_status,
                                "cv_mean": cv_mean,
                                "oos_mean": oos_mean,
                                "is_consistent": oos_delta_pct <= 0.10
                            }
                            
                            print(f"📊 CV vs WFA一致性: CV={cv_mean:.4f}, WFA={oos_mean:.4f}, 差異={oos_delta_pct:.1%}")
                        else:
                            overfitting_analysis["risk_factors"].append(
                                "缺少CV與樣本外驗證對比，無法評估泛化一致性"
                            )
                            overfitting_analysis["overall_risk_level"] = max(
                                overfitting_analysis["overall_risk_level"], "MEDIUM"
                            )
            
            # 4. 檢查標籤平衡性 - 🔧 新增關鍵指標
            try:
                # 嘗試載入最新的標籤數據進行平衡性檢查
                labels_path = f"data/processed/labels/{self.symbol}_{self.timeframe}/{self.version}/{self.symbol}_{self.timeframe}_labels.parquet"
                if os.path.exists(labels_path):
                    labels_df = pd.read_parquet(labels_path)
                    label_counts = labels_df["label"].value_counts()
                    total_samples = len(labels_df)
                    
                    # 計算標籤分佈
                    label_distribution = {}
                    for label, count in label_counts.items():
                        label_distribution[f"label_{int(label)}"] = {
                            "count": int(count),
                            "percentage": float(count / total_samples * 100)
                        }
                    
                    # 檢查嚴重不平衡 - 🚨 關鍵風險檢測
                    max_percentage = max([dist["percentage"] for dist in label_distribution.values()])
                    if max_percentage > 90:  # 超過90%為一個類別
                        overfitting_analysis["risk_factors"].append(
                            f"標籤極度不平衡：最大類別佔{max_percentage:.1f}%，嚴重影響模型學習"
                        )
                        overfitting_analysis["overall_risk_level"] = "CRITICAL"
                    elif max_percentage > 80:  # 超過80%為一個類別
                        overfitting_analysis["risk_factors"].append(
                            f"標籤嚴重不平衡：最大類別佔{max_percentage:.1f}%，模型偏向性強"
                        )
                        overfitting_analysis["overall_risk_level"] = "HIGH"
                    elif max_percentage > 70:  # 超過70%為一個類別
                        overfitting_analysis["risk_factors"].append(
                            f"標籤不平衡：最大類別佔{max_percentage:.1f}%，建議重新調整閾值"
                        )
                        overfitting_analysis["overall_risk_level"] = max(
                            overfitting_analysis["overall_risk_level"], "MEDIUM"
                        )
                    
                    # 檢查三分類標籤分佈的合理性
                    if len(label_distribution) == 3:
                        percentages = [dist["percentage"] for dist in label_distribution.values()]
                        min_percentage = min(percentages)
                        if min_percentage < 5:  # 某類別少於5%
                            overfitting_analysis["risk_factors"].append(
                                f"標籤類別過少：最小類別僅{min_percentage:.1f}%，學習樣本不足"
                            )
                            overfitting_analysis["overall_risk_level"] = max(
                                overfitting_analysis["overall_risk_level"], "MEDIUM"
                            )
                    
                    # 記錄詳細標籤分佈信息
                    overfitting_analysis["detailed_metrics"]["label_distribution"] = label_distribution
                    overfitting_analysis["detailed_metrics"]["label_balance_ratio"] = max_percentage / min([dist["percentage"] for dist in label_distribution.values()])
                    
                    print(f"📊 標籤分佈檢查完成：{len(label_distribution)}類，最大佔比{max_percentage:.1f}%")
                else:
                    print("⚠️ 標籤文件不存在，跳過平衡性檢查")
            except Exception as e:
                print(f"⚠️ 標籤平衡性檢查失敗: {e}")
            
            # 5. 生成分級建議 - 🔧 基於風險等級
            risk_level = overfitting_analysis["overall_risk_level"]
            
            if risk_level == "CRITICAL":
                overfitting_analysis["recommendations"].extend([
                    "🚨 緊急：立即停止使用當前模型，存在嚴重過度擬合",
                    "📊 重新採樣數據：考慮從1m數據重新生成更長時間窗口的數據",
                    "🔧 大幅降低模型複雜度：減少特徵數和樹的數量",
                    "📈 使用獨立的長期留存集(6個月以上)進行驗證",
                    "⚙️ 重新設計特徵工程，避免未來信息洩露",
                    "🎯 調整標籤閾值：重新設計threshold_range以平衡標籤分佈",
                    "⚖️ 實施重採樣策略：考慮SMOTE或undersampling平衡類別"
                ])
            elif risk_level == "HIGH":
                overfitting_analysis["recommendations"].extend([
                    "⚠️ 高風險：需要多維度驗證確認模型可靠性",
                    "📊 CV標準差檢測：檢查模型在不同時期的穩定性",
                    "🔄 樣本外回測：使用未來6個月數據進行獨立驗證",
                    "🔧 增加正則化約束(L1/L2)和早停機制",
                    "📈 特徵穩定性分析：檢查特徵重要性是否一致",
                    "⚖️ 檢查標籤分佈：調整threshold參數以改善類別平衡"
                ])
            elif risk_level == "MEDIUM":
                overfitting_analysis["recommendations"].extend([
                    "📊 建議在不同市場環境下測試模型穩健性",
                    "🔄 考慮多時框聯合驗證和交叉驗證",
                    "⚙️ 監控模型在實際交易中的退化情況",
                    "📈 定期重新訓練和參數調整",
                    "⚖️ 監控標籤分佈變化：確保訓練和測試期間的一致性"
                ])
            else:
                overfitting_analysis["recommendations"].append(
                    "✅ 當前優化結果顯示較低的過度擬合風險，可以謹慎進入回測階段"
                )
            
            # 5. 記錄分析摘要
            risk_count = len(overfitting_analysis["risk_factors"])
            overfitting_analysis["analysis_summary"] = {
                "total_risk_factors": risk_count,
                "analysis_date": datetime.now().isoformat(),
                "timeframe_analyzed": self.timeframe,
                "recommendation_priority": "HIGH" if risk_level in ["CRITICAL", "HIGH"] else "MEDIUM"
            }
            
            print(f"🔍 過度擬合分析完成: {risk_level}風險, {risk_count}個風險因子")
            
        except Exception as e:
            overfitting_analysis["error"] = f"過度擬合分析失敗: {e}"
            import traceback
            print(f"⚠️ 過度擬合分析錯誤: {traceback.format_exc()}")
        
        return overfitting_analysis

    def _safe_serialize_series(self, series):
        """安全序列化pandas Series"""
        if series is None:
            return None
        try:
            if hasattr(series, 'to_dict'):
                return {str(k): v for k, v in series.to_dict().items()}
            return series
        except:
            return None

    def _safe_serialize_dataframe(self, df):
        """安全序列化pandas DataFrame"""
        if df is None:
            return None
        try:
            if hasattr(df, 'to_dict'):
                return df.to_dict('records')
            return df
        except:
            return None

    def clean_results_for_json(self, results: Dict[str, Any]):
        """清理結果中不可JSON序列化的對象"""
        for stage_key, stage_results in results.items():
            if isinstance(stage_results, dict):
                # 移除Optuna Study對象
                if 'study' in stage_results:
                    del stage_results['study']
                
                # 移除模型對象
                if 'best_model' in stage_results:
                    del stage_results['best_model']
                
                # 處理pandas對象和Timestamp
                for key, value in list(stage_results.items()):
                    if hasattr(value, 'to_dict'):
                        if hasattr(value, 'index') and len(value.index) > 0:
                            # 是Series或DataFrame，且有時間索引
                            try:
                                stage_results[key] = {
                                    str(k): v for k, v in value.to_dict().items()
                                }
                            except:
                                stage_results[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime)):
                        stage_results[key] = value.isoformat()
                    elif isinstance(value, dict):
                        # 遞歸處理嵌套字典
                        self._clean_nested_dict(value)

    def _clean_nested_dict(self, d: dict):
        """遞歸清理嵌套字典中的不可序列化對象"""
        for key, value in list(d.items()):
            if isinstance(value, (pd.Timestamp, datetime)):
                d[key] = value.isoformat()
            elif isinstance(value, dict):
                self._clean_nested_dict(value)
            elif hasattr(value, 'to_dict'):
                try:
                    d[key] = {str(k): v for k, v in value.to_dict().items()}
                except:
                    d[key] = None
    
    def generate_optimization_summary(self) -> Dict[str, Any]:
        """生成優化摘要"""
        summary = {
            "total_stages": 3,
            "completed_stages": 0,
            "failed_stages": 0,
            "total_duration": 0.0,
            "final_performance": {},
            "resource_usage": {}
        }
        
        # 統計執行情況
        for log_entry in self.execution_log:
            if log_entry["status"] == "success":
                summary["completed_stages"] += 1
            elif log_entry["status"] == "failed":
                summary["failed_stages"] += 1
            
            summary["total_duration"] += log_entry.get("duration_seconds", 0)
        
        # 提取最終性能
        if "model_optimization" in self.optimization_results:
            model_results = self.optimization_results["model_optimization"]
            summary["final_performance"] = {
                "best_f1_score": model_results.get("best_score", 0),
                "selected_features_count": len(model_results.get("cv_results", {}).get("fold_scores", [])),
                "model_complexity": {
                    "n_estimators": model_results.get("best_params", {}).get("n_estimators", 0),
                    "max_depth": model_results.get("best_params", {}).get("max_depth", 0),
                    "num_leaves": model_results.get("best_params", {}).get("num_leaves", 0)
                }
            }
        
        return summary
    
    def _run_automated_consistency_check(self) -> Dict[str, Any]:
        """
        🆕 自動化CV vs WFA一致性復核 - 嵌入main_optimizer最後階段
        
        自動讀取CV結果與WFA結果，並輸出統一JSON報告
        """
        print("\n" + "="*60)
        print("🔍 自動化一致性復核")
        print("="*60)
        
        try:
            consistency_report = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "version": self.version,
                "status": "pending",
                "cv_results": {},
                "wfa_results": {},
                "consistency_analysis": {},
                "recommendations": []
            }
            
            # 1. 提取CV結果
            model_data = self.optimization_results.get("model_optimization", {})
            cv_results = model_data.get("cv_results", {})
            
            if cv_results:
                cv_fold_scores = cv_results.get("fold_scores", [])
                if cv_fold_scores:
                    consistency_report["cv_results"] = {
                        "mean": float(np.mean(cv_fold_scores)),
                        "std": float(np.std(cv_fold_scores)),
                        "folds": len(cv_fold_scores),
                        "scores": [float(s) for s in cv_fold_scores],
                        "min": float(np.min(cv_fold_scores)),
                        "max": float(np.max(cv_fold_scores))
                    }
                    print(f"📊 CV結果: {consistency_report['cv_results']['mean']:.4f} ± {consistency_report['cv_results']['std']:.4f}")
            
            # 2. 提取WFA結果（從trial_results中查找）
            trial_results = model_data.get("trial_results", {})
            best_trial_num = model_data.get("best_trial_number", model_data.get("cv_results", {}).get("best_trial_number"))
            
            wfa_data = None
            if best_trial_num and str(best_trial_num) in trial_results:
                best_trial_data = trial_results[str(best_trial_num)]
                wfa_summary = best_trial_data.get("oos_summary", {})
                
                if wfa_summary:
                    consistency_report["wfa_results"] = {
                        "mean": float(wfa_summary.get("mean", 0.0)),
                        "std": float(wfa_summary.get("std", 0.0)),
                        "folds": int(wfa_summary.get("folds", 0)),
                        "scores": [float(s) for s in wfa_summary.get("scores", [])],
                        "min": float(np.min(wfa_summary.get("scores", [0]))),
                        "max": float(np.max(wfa_summary.get("scores", [0])))
                    }
                    wfa_data = consistency_report["wfa_results"]
                    print(f"📊 WFA結果: {wfa_data['mean']:.4f} ± {wfa_data['std']:.4f}")
            
            # 3. 計算一致性指標
            cv_data = consistency_report["cv_results"]
            if cv_data and wfa_data and cv_data.get("mean", 0) > 0:
                cv_mean = cv_data["mean"]
                wfa_mean = wfa_data["mean"]
                delta_pct = abs(wfa_mean - cv_mean) / cv_mean * 100
                
                consistency_report["consistency_analysis"] = {
                    "delta_pct": float(delta_pct),
                    "delta_abs": float(abs(wfa_mean - cv_mean)),
                    "is_consistent": delta_pct <= 10.0,
                    "consistency_level": self._get_consistency_level(delta_pct),
                    "cv_vs_wfa_ratio": float(wfa_mean / cv_mean) if cv_mean > 0 else 0.0
                }
                
                print(f"📊 一致性分析: 差異 {delta_pct:.2f}% ({consistency_report['consistency_analysis']['consistency_level']})")
                
                # 風險評估與建議
                consistency_report["recommendations"] = self._generate_consistency_recommendations(
                    delta_pct, cv_data, wfa_data
                )
                
                consistency_report["status"] = "completed"
            else:
                print("⚠️ 缺少CV或WFA結果，無法完成一致性分析")
                consistency_report["status"] = "incomplete"
                consistency_report["recommendations"] = [
                    "缺少WFA結果，建議重新運行模型優化並啟用WFA",
                    "確保enhanced_regularization.compare_cv_vs_oos=True"
                ]
            
            # 4. 計算標籤分佈和Macro F1（如果可能）
            try:
                labels_path = f"data/processed/labels/{self.symbol}_{self.timeframe}/{self.version}/{self.symbol}_{self.timeframe}_labels.parquet"
                if os.path.exists(labels_path):
                    labels_df = pd.read_parquet(labels_path)
                    label_counts = labels_df["label"].value_counts().sort_index()
                    total_samples = len(labels_df)
                    
                    label_distribution = {}
                    for label, count in label_counts.items():
                        label_distribution[f"label_{int(label)}"] = {
                            "count": int(count),
                            "percentage": float(count / total_samples * 100)
                        }
                    
                    consistency_report["label_distribution"] = label_distribution
                    
                    # 計算平衡分數
                    percentages = [dist["percentage"] for dist in label_distribution.values()]
                    balance_score = min(percentages) / max(percentages) if max(percentages) > 0 else 0
                    consistency_report["label_balance_score"] = float(balance_score)
                    
                    print(f"📊 標籤分佈: {len(label_distribution)} 類，平衡分數: {balance_score:.3f}")
                    
                    # 標籤平衡建議
                    if balance_score < 0.4:
                        consistency_report["recommendations"].append(
                            f"標籤嚴重不平衡（平衡分數={balance_score:.3f} < 0.4），建議重新優化標籤參數"
                        )
                        
            except Exception as e:
                print(f"⚠️ 標籤分佈分析失敗: {e}")
            
            # 5. 保存一致性報告到版本目錄
            try:
                consistency_file = f"results/models/{self.symbol}_{self.timeframe}/{self.version}/consistency_check_report.json"
                os.makedirs(os.path.dirname(consistency_file), exist_ok=True)
                
                with open(consistency_file, 'w', encoding='utf-8') as f:
                    json.dump(consistency_report, f, indent=2, ensure_ascii=False)
                
                print(f"💾 一致性報告已保存到: {consistency_file}")
                
            except Exception as e:
                print(f"⚠️ 一致性報告保存失敗: {e}")
            
            return consistency_report
            
        except Exception as e:
            print(f"❌ 自動化一致性復核失敗: {e}")
            import traceback
            print(f"詳細錯誤: {traceback.format_exc()}")
            
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "version": self.version,
                "status": "failed",
                "error": str(e),
                "recommendations": ["一致性復核失敗，請檢查模型優化結果的完整性"]
            }
    
    def _get_consistency_level(self, delta_pct: float) -> str:
        """獲取一致性水平描述"""
        if delta_pct <= 5.0:
            return "優秀"
        elif delta_pct <= 10.0:
            return "良好"
        elif delta_pct <= 15.0:
            return "一般"
        elif delta_pct <= 20.0:
            return "較差"
        else:
            return "不一致"
    
    def _generate_consistency_recommendations(self, delta_pct: float, 
                                            cv_data: Dict, wfa_data: Dict) -> List[str]:
        """生成一致性建議"""
        recommendations = []
        
        if delta_pct <= 5.0:
            recommendations.append("CV與WFA高度一致，模型泛化能力優秀")
            recommendations.append("可以進行實盤部署前的最終驗證")
        elif delta_pct <= 10.0:
            recommendations.append("CV與WFA基本一致，模型泛化能力良好")
            recommendations.append("建議進行更多樣本外測試以確認穩定性")
        elif delta_pct <= 15.0:
            recommendations.append("CV與WFA存在輕微不一致，需要謹慎")
            recommendations.append("建議降低模型複雜度或增加正則化")
            recommendations.append("考慮重新進行特徵選擇以提升穩健性")
        elif delta_pct <= 20.0:
            recommendations.append("CV與WFA不一致程度較高，存在過擬合風險")
            recommendations.append("強烈建議降低模型複雜度（減少樹深度、葉子數）")
            recommendations.append("增加正則化參數（reg_alpha, reg_lambda）")
            recommendations.append("減少特徵數量或重新進行特徵選擇")
        else:
            recommendations.append("CV與WFA嚴重不一致，模型嚴重過擬合")
            recommendations.append("不建議部署該模型")
            recommendations.append("需要重新設計特徵工程和模型架構")
            recommendations.append("考慮使用更簡單的模型或更強的正則化")
        
        # 基於標準差的額外建議
        cv_std = cv_data.get("std", 0.0)
        wfa_std = wfa_data.get("std", 0.0)
        
        if cv_std > 0.05:
            recommendations.append(f"CV標準差較大({cv_std:.3f})，模型在不同時期表現不穩定")
        
        if wfa_std > 0.05:
            recommendations.append(f"WFA標準差較大({wfa_std:.3f})，模型在時間序列上不穩定")
        
        return recommendations
    
    def generate_final_report(self) -> str:
        """生成最終綜合報告"""
        summary = self.generate_optimization_summary()
        overfitting = self.optimization_results.get("overfitting_analysis", {})
        
        report = f"""
🎉 {'='*78} 🎉
🏆 {self.symbol} {self.timeframe} 模組化優化系統總結報告
🎉 {'='*78} 🎉

📊 **執行摘要**
├─ 完成階段: {summary['completed_stages']}/3
├─ 失敗階段: {summary['failed_stages']}
├─ 總執行時間: {summary['total_duration']:.1f} 秒
└─ 優化策略: 三階段模組化優化

📈 **性能指標**"""
        
        if "final_performance" in summary and summary["final_performance"]:
            perf = summary["final_performance"]
            report += f"""
├─ 最終F1分數: {perf.get('best_f1_score', 0):.6f}
├─ 選中特徵數: {perf.get('selected_features_count', 0)}
└─ 模型複雜度: {perf.get('model_complexity', {}).get('n_estimators', 0)} 樹"""
        else:
            report += "\n└─ 優化未完成或失敗"
        
        # 🔍 過度擬合風險評估
        if overfitting:
            risk_level = overfitting.get("overall_risk_level", "UNKNOWN")
            risk_factors = overfitting.get("risk_factors", [])
            
            report += f"""
            
🔍 **過度擬合風險評估**
├─ 風險等級: {risk_level}
├─ 識別風險因子: {len(risk_factors)} 項"""
            
            for i, factor in enumerate(risk_factors[:3], 1):  # 只顯示前3個
                report += f"\n│  {i}. {factor}"
            
            if len(risk_factors) > 3:
                report += f"\n│  ... 及其他 {len(risk_factors) - 3} 項風險"

        report += f"""

💡 **下一步建議**
├─ 如需進一步提升: 考慮方案C擴展(高階特徵+多模型集成)
├─ 如需部署: 使用最佳參數重新訓練完整模型
└─ 如需回測: 結合Backtrader進行策略驗證

🎉 {'='*78} 🎉
"""

        return report

    def load_processed_data_for_training(self, version: str = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        載入已處理的特徵和標籤數據，用於模型訓練
        
        Args:
            version: 指定版本號，如不指定則使用當前版本
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict]: 選中的特徵數據、標籤、參數信息
        """
        # 🔢 確定使用的版本
        use_version = version or self.version
        
        # 構建版本化路徑
        version_paths = self._get_version_paths(use_version)
        
        try:
            print(f"📊 載入已處理的訓練數據 (版本: {use_version})...")
            
            # 載入選中的特徵
            if os.path.exists(version_paths["selected_features"]):
                features_df = pd.read_parquet(version_paths["selected_features"])
                print(f" ✅ 載入選中特徵: {features_df.shape}")
            else:
                print(f" ⚠️ 選中特徵文件不存在: {version_paths['selected_features']}")
                return None, None, {}
            
            # 載入標籤
            if os.path.exists(version_paths["labels"]):
                labels_df = pd.read_parquet(version_paths["labels"])
                labels = labels_df["label"]
                print(f" ✅ 載入標籤: {len(labels)} 個樣本")
            else:
                print(f" ⚠️ 標籤文件不存在: {version_paths['labels']}")
                return None, None, {}
            
            # 載入參數信息
            optimization_info = {"version": use_version}
            
            # 載入標籤參數
            label_params_path = version_paths["labels"].replace('_labels.parquet', '_label_params.json')
            if os.path.exists(label_params_path):
                with open(label_params_path, 'r', encoding='utf-8') as f:
                    optimization_info["label_params"] = json.load(f)
                print(f" ✅ 載入標籤參數")
            
            # 載入特徵選擇參數
            feature_params_path = version_paths["selected_features"].replace('_selected_features.parquet', '_feature_selection_params.json')
            if os.path.exists(feature_params_path):
                with open(feature_params_path, 'r', encoding='utf-8') as f:
                    optimization_info["feature_selection_params"] = json.load(f)
                print(f" ✅ 載入特徵選擇參數")
            
            # 數據對齊
            common_idx = features_df.index.intersection(labels.index)
            features_aligned = features_df.loc[common_idx]
            labels_aligned = labels.loc[common_idx]
            
            print(f" 🔗 對齊後數據: 特徵{features_aligned.shape}, 標籤{len(labels_aligned)}個")
            print(f" 📊 標籤分佈: {np.bincount(labels_aligned)}")
            
            return features_aligned, labels_aligned, optimization_info
            
        except Exception as e:
            print(f"❌ 訓練數據載入失敗: {e}")
            return None, None, {}
    
    def _get_version_paths(self, version: str) -> Dict[str, str]:
        """獲取指定版本的文件路徑"""
        return {
            "labels": f"data/processed/labels/{self.symbol}_{self.timeframe}/{version}/{self.symbol}_{self.timeframe}_labels.parquet",
            "selected_features": f"data/processed/features/{self.symbol}_{self.timeframe}/{version}/{self.symbol}_{self.timeframe}_selected_features.parquet",
            "results_dir": f"results/modular_optimization/{self.symbol}_{self.timeframe}/{version}",
            "models_dir": f"results/models/{self.symbol}_{self.timeframe}/{version}",
            "logs_dir": f"logs/optimization/{self.symbol}_{self.timeframe}/{version}"
        }
    
    def get_training_data_info(self, version: str = None) -> Dict[str, Any]:
        """
        獲取訓練數據的詳細信息
        
        Args:
            version: 指定版本號，如不指定則使用當前版本
            
        Returns:
            Dict: 包含數據路徑、文件狀態等信息
        """
        use_version = version or self.version
        version_paths = self._get_version_paths(use_version)
        
        info = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_version": self.version,
            "requested_version": use_version,
            "available_versions": self.list_versions(),
            "latest_version": self.get_latest_version() if self.list_versions() else "v1",
            "data_paths": version_paths,
            "file_status": {}
        }
        
        # 檢查版本化文件是否存在
        key_files = ["labels", "selected_features"]
        for key in key_files:
            file_path = version_paths[key]
            info["file_status"][key] = {
                "path": file_path,
                "exists": os.path.exists(file_path),
                "size_mb": round(os.path.getsize(file_path) / 1024 / 1024, 2) if os.path.exists(file_path) else 0
            }
        
        return info
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """比較兩個版本的結果"""
        comparison = {
            "version1": version1,
            "version2": version2,
            "comparison_results": {}
        }
        
        try:
            # 載入兩個版本的結果
            results1_path = f"results/modular_optimization/{self.symbol}_{self.timeframe}/{version1}/modular_optuna_results.json"
            results2_path = f"results/modular_optimization/{self.symbol}_{self.timeframe}/{version2}/modular_optuna_results.json"
            
            if os.path.exists(results1_path) and os.path.exists(results2_path):
                with open(results1_path, 'r', encoding='utf-8') as f:
                    results1 = json.load(f)
                with open(results2_path, 'r', encoding='utf-8') as f:
                    results2 = json.load(f)
                
                # 比較關鍵指標
                comparison["comparison_results"] = {
                    "model_performance": {
                        "v1_score": results1.get("results", {}).get("model_optimization", {}).get("best_score", 0),
                        "v2_score": results2.get("results", {}).get("model_optimization", {}).get("best_score", 0)
                    },
                    "feature_selection": {
                        "v1_features": len(results1.get("results", {}).get("feature_selection", {}).get("best_features", [])),
                        "v2_features": len(results2.get("results", {}).get("feature_selection", {}).get("best_features", []))
                    },
                    "overfitting_risk": {
                        "v1_risk": results1.get("results", {}).get("overfitting_analysis", {}).get("overall_risk_level", "UNKNOWN"),
                        "v2_risk": results2.get("results", {}).get("overfitting_analysis", {}).get("overall_risk_level", "UNKNOWN")
                    }
                }
                
                # 推薦最佳版本
                v1_score = comparison["comparison_results"]["model_performance"]["v1_score"]
                v2_score = comparison["comparison_results"]["model_performance"]["v2_score"]
                comparison["recommendation"] = version1 if v1_score > v2_score else version2
                
            else:
                comparison["error"] = "其中一個或兩個版本的結果文件不存在"
                
        except Exception as e:
            comparison["error"] = f"版本比較失敗: {e}"
        
        return comparison
    
    def delete_version(self, version: str) -> bool:
        """刪除指定版本的所有數據"""
        try:
            import shutil
            
            # 要刪除的目錄列表
            dirs_to_delete = [
                f"data/processed/features/{self.symbol}_{self.timeframe}/{version}",
                f"data/processed/labels/{self.symbol}_{self.timeframe}/{version}",
                f"results/modular_optimization/{self.symbol}_{self.timeframe}/{version}",
                f"results/models/{self.symbol}_{self.timeframe}/{version}",
                f"logs/optimization/{self.symbol}_{self.timeframe}/{version}"
            ]
            
            deleted_count = 0
            for dir_path in dirs_to_delete:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                    print(f"✅ 已刪除: {dir_path}")
            
            print(f"🗑️ 版本 {version} 已完全刪除 ({deleted_count} 個目錄)")
            return True
            
        except Exception as e:
            print(f"❌ 刪除版本 {version} 失敗: {e}")
            return False
    
    def run_stage3_only_optimization(self, n_trials: int = 100) -> Dict[str, Any]:
        """只運行第三階段模型優化（需要已有的標籤和特徵數據）"""
        print("🤖 開始僅第三階段模型優化")
        print(f"交易對: {self.symbol}")
        print(f"時間框架: {self.timeframe}")
        print("="*80)
        
        total_start_time = datetime.now()
        
        try:
            # 🔍 檢查是否有已保存的標籤和特徵數據
            print("🔍 載入已保存的標籤和特徵數據...")
            
            # 載入標籤數據
            labels_path = self.data_paths["labels"]
            if not os.path.exists(labels_path):
                raise ValueError(f"標籤數據不存在: {labels_path}")
            
            labels_df = pd.read_parquet(labels_path)
            if 'labels' not in labels_df.columns:
                raise ValueError("標籤數據格式錯誤，缺少 'labels' 列")
            
            labels = labels_df['labels']
            print(f"✅ 載入標籤數據: {len(labels)} 個樣本")
            
            # 載入特徵數據
            selected_features_path = os.path.join(
                self.data_paths["features_dir"], 
                f"{self.symbol}_{self.timeframe}_selected_features.parquet"
            )
            
            if not os.path.exists(selected_features_path):
                raise ValueError(f"選定特徵數據不存在: {selected_features_path}")
            
            selected_features_df = pd.read_parquet(selected_features_path)
            print(f"✅ 載入選定特徵數據: {selected_features_df.shape}")
            
            # 對齊數據長度
            min_length = min(len(selected_features_df), len(labels))
            X_aligned = selected_features_df.iloc[:min_length]
            y_aligned = labels.iloc[:min_length]
            
            print(f"📊 對齊後數據: 特徵 {X_aligned.shape}, 標籤 {len(y_aligned)}")
            
            # 執行第三階段：模型優化
            print("\n" + "="*60)
            print("🎯 第三階段：模型超參數優化")
            print("="*60)
            
            model_results = self.model_optimizer.optimize(X_aligned, y_aligned, n_trials=n_trials)
            
            if not model_results:
                raise ValueError("模型優化失敗")
            
            # 保存結果
            self.optimization_results["model_optimization"] = self._extract_model_results(model_results)
            
            # 保存最佳參數到JSON
            best_params = model_results.get('best_params', {})
            if best_params:
                self.save_optuna_params('model', best_params, model_results.get('best_score', 0.0))
            
            # 保存模型參數到版本化目錄
            try:
                model_params_path = os.path.join(
                    self.data_paths["models_dir"], 
                    f"{self.symbol}_{self.timeframe}_model_params.json"
                )
                
                os.makedirs(os.path.dirname(model_params_path), exist_ok=True)
                
                model_info = {
                    "best_params": best_params,
                    "best_score": model_results.get('best_score', 0.0),
                    "final_metrics": model_results.get('final_metrics', {}),
                    "n_trials": model_results.get('n_trials', 0),
                    "optimization_time": model_results.get('optimization_time', 0),
                    "version": self.version,
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(model_params_path, 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, indent=2, ensure_ascii=False, default=str)
                print(f"💾 模型參數已保存到: {model_params_path}")
                
            except Exception as e:
                print(f"⚠️ 模型參數保存失敗: {e}")
            
            # 生成報告
            final_report = self._generate_stage3_only_report(model_results)
            print(final_report)
            
            total_duration = (datetime.now() - total_start_time).total_seconds()
            self.log_execution("stage3_only_optimization", "success", 
                             f"第三階段優化成功，耗時 {total_duration:.1f} 秒", total_duration)
            
            return self.optimization_results
            
        except Exception as e:
            total_duration = (datetime.now() - total_start_time).total_seconds()
            error_msg = f"第三階段優化失敗: {e}"
            print(f"❌ {error_msg}")
            self.log_execution("stage3_only_optimization", "failed", error_msg, total_duration)
            return {}
    
    def _generate_stage3_only_report(self, model_results: Dict[str, Any]) -> str:
        """生成僅第三階段的優化報告"""
        report = []
        report.append("\n🎉 第三階段模型優化完成報告")
        report.append("=" * 60)
        report.append(f"📊 交易對: {self.symbol}")
        report.append(f"📊 時間框架: {self.timeframe}")
        report.append(f"📊 版本: {self.version}")
        report.append("")
        
        # 模型優化結果
        if model_results:
            report.append("🤖 模型優化結果:")
            report.append(f"├─ 最佳F1分數: {model_results.get('best_score', 0):.6f}")
            report.append(f"├─ 優化試驗數: {model_results.get('n_trials', 0)}")
            report.append(f"└─ 優化耗時: {model_results.get('optimization_time', 0):.1f}秒")
            report.append("")
            
            # 最佳參數
            best_params = model_results.get('best_params', {})
            if best_params:
                report.append("🏆 最佳模型參數:")
                for param, value in best_params.items():
                    if isinstance(value, float):
                        report.append(f"├─ {param}: {value:.4f}")
                    else:
                        report.append(f"├─ {param}: {value}")
                report.append("")
            
            # 最終測試指標
            final_metrics = model_results.get('final_metrics', {})
            if final_metrics:
                report.append("📈 最終測試指標:")
                for metric, value in final_metrics.items():
                    report.append(f"├─ {metric}: {value:.4f}")
                report.append("")
        
        report.append("✅ 第三階段模型優化成功完成！")
        report.append("💡 您現在可以使用優化後的模型進行預測或回測。")
        
        return "\n".join(report)
    
    @classmethod
    def create_version_manager(cls, symbol: str, timeframe: str):
        """創建版本管理器實例（不進行優化，僅用於版本管理）"""
        return cls(symbol, timeframe, auto_version=False)


# 主函數示例
if __name__ == "__main__":
    # 版本化優化示例
    
    # 方式1: 自動版本（推薦）
    print("🔢 自動版本優化...")
    optimizer = ModularOptunaOptimizer("BTCUSDT", "1h")  # 自動使用下一個版本
    results = optimizer.run_complete_optimization()
    
    # 方式2: 指定版本
    # optimizer = ModularOptunaOptimizer("BTCUSDT", "1h", version="v1")
    
    # 方式3: 版本管理示例
    # vm = ModularOptunaOptimizer.create_version_manager("BTCUSDT", "1h")
    # print(f"可用版本: {vm.list_versions()}")
    # print(f"最新版本: {vm.get_latest_version()}")
    
    # 載入特定版本的數據
    # features, labels, params = vm.load_processed_data_for_training(version="v1")
