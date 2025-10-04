#!/usr/bin/env python3
"""
模型訓練器 - 簡化版本，專注於模型訓練邏輯
原 Optuna 優化功能已遷移至 optuna_system/optimizers/optuna_model.py

負責：
- LightGBM 模型訓練
- 模型驗證和評估
- 模型保存和加載
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import lightgbm as lgb
from typing import Dict, List, Tuple, Any
import warnings
import pickle
import json
from pathlib import Path
warnings.filterwarnings('ignore')

from .config import get_config


class ModelTrainer:
    """模型訓練器 - 專注於模型訓練邏輯"""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = get_config("model", timeframe)
        self.base_config = get_config("base")
        
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_params: Dict[str, Any] = None) -> lgb.LGBMClassifier:
        """
        訓練 LightGBM 模型
        
        Args:
            X: 特徵DataFrame
            y: 標籤Series
            model_params: 模型參數字典
        
        Returns:
            訓練好的模型
        """
        try:
            print(f"🚀 開始訓練模型 - {self.symbol} {self.timeframe}")
            
            # 使用提供的參數或默認配置
            if model_params is None:
                model_params = self._get_default_params()
            
            print(f"📋 模型參數: {model_params}")
            
            # 創建並訓練模型
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X, y)
            
            print(f"✅ 模型訓練完成")
            
            return model
                
        except Exception as e:
            print(f"❌ 模型訓練失敗: {e}")
            raise e

    def train_with_validation(self, X: pd.DataFrame, y: pd.Series,
                            model_params: Dict[str, Any] = None,
                            validation_split: float = 0.2) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """
        使用驗證集訓練模型
        
        Args:
            X: 特徵DataFrame
            y: 標籤Series
            model_params: 模型參數字典
            validation_split: 驗證集比例
        
        Returns:
            (模型, 驗證指標字典)
        """
        try:
            print(f"🚀 開始帶驗證的模型訓練 - {self.symbol} {self.timeframe}")
            
            # 時間序列分割
            split_point = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
            
            print(f"📊 數據分割: 訓練 {len(X_train)}, 驗證 {len(X_val)}")
            
            # 使用提供的參數或默認配置
            if model_params is None:
                model_params = self._get_default_params()
                
                # 訓練模型
            model = lgb.LGBMClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # 驗證模型
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            
            print(f"📈 驗證結果:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")
            
            return model, metrics
                
        except Exception as e:
            print(f"❌ 帶驗證的模型訓練失敗: {e}")
            raise e

    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series,
                           model_params: Dict[str, Any] = None,
                           cv_folds: int = None) -> Dict[str, List[float]]:
        """
        交叉驗證模型
        
        Args:
            X: 特徵DataFrame
            y: 標籤Series
            model_params: 模型參數字典
            cv_folds: 交叉驗證折數
        
        Returns:
            各項指標的分數列表字典
        """
        try:
            if cv_folds is None:
                cv_folds = self.base_config.get("cv_folds", 5)
            
            print(f"🔄 交叉驗證 ({cv_folds} 折) - {self.symbol} {self.timeframe}")
            
            # 使用提供的參數或默認配置
            if model_params is None:
                model_params = self._get_default_params()
            
            # 時間序列交叉驗證
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            all_metrics = {'f1': [], 'precision': [], 'recall': [], 'accuracy': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                print(f"📂 第 {fold} 折...")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 檢查類別數量
                if len(y_train.unique()) < 2 or len(y_val.unique()) < 2:
                    print(f"   ⚠️ 跳過：類別不足")
                    continue
                
                # 訓練模型
                model = lgb.LGBMClassifier(**model_params)
                model.fit(X_train, y_train)
                
                # 預測和評估
                y_pred = model.predict(X_val)
                metrics = self._calculate_metrics(y_val, y_pred)
                
                # 記錄指標
                for metric, value in metrics.items():
                    all_metrics[metric].append(value)
                
                print(f"   F1: {metrics['f1']:.4f}")
            
            # 計算平均值
            avg_metrics = {metric: np.mean(scores) for metric, scores in all_metrics.items()}
            
            print(f"📊 交叉驗證結果:")
            for metric, value in avg_metrics.items():
                print(f"   平均 {metric}: {value:.4f} ± {np.std(all_metrics[metric]):.4f}")
            
            return all_metrics
        
        except Exception as e:
            print(f"❌ 交叉驗證失敗: {e}")
            return {}

    def save_model(self, model: lgb.LGBMClassifier, save_dir: str,
                  model_params: Dict[str, Any] = None,
                  feature_names: List[str] = None,
                  metrics: Dict[str, float] = None) -> str:
        """
        保存模型及相關信息
        
        Args:
            model: 訓練好的模型
            save_dir: 保存目錄
            model_params: 模型參數
            feature_names: 特徵名稱列表
            metrics: 模型指標
        
        Returns:
            保存路徑
        """
        try:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型
            model_file = save_path / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # 保存配置信息
            config_info = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'model_params': model_params or {},
                'feature_names': feature_names or [],
                'metrics': metrics or {},
                'feature_count': len(feature_names) if feature_names else 0
            }
            
            config_file = save_path / "model_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            
            # 保存特徵重要性
            if hasattr(model, 'feature_importances_') and feature_names:
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                importance_file = save_path / "feature_importance.json"
                with open(importance_file, 'w', encoding='utf-8') as f:
                    json.dump(importance_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 模型已保存至: {save_path}")
            
            return str(save_path)
        
        except Exception as e:
            print(f"❌ 模型保存失敗: {e}")
            raise e

    def load_model(self, model_path: str) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
        """
        加載模型及配置信息
        
        Args:
            model_path: 模型目錄路徑
        
        Returns:
            (模型, 配置信息)
        """
        try:
            model_dir = Path(model_path)
            
            # 加載模型
            model_file = model_dir / "model.pkl"
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # 加載配置信息
            config_file = model_dir / "model_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_info = json.load(f)
            else:
                config_info = {}
            
            print(f"✅ 模型已加載: {model_path}")
            
            return model, config_info
            
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")
            raise e

    def _get_default_params(self) -> Dict[str, Any]:
        """獲取默認模型參數"""
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.base_config.get("random_state", 42),
            'verbose': -1,
            'num_threads': 1
        }

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """計算評估指標"""
        try:
            metrics = {}
            
            # 根據類別數量選擇平均方法
            if len(y_true.unique()) > 2:
                avg_method = 'weighted'
            else:
                avg_method = 'binary'
            
            metrics['f1'] = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
            metrics['precision'] = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            return metrics
            
        except Exception as e:
            print(f"⚠️ 指標計算失敗: {e}")
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}

    def generate_report(self, model: lgb.LGBMClassifier, 
                       metrics: Dict[str, float],
                       feature_names: List[str] = None) -> str:
        """生成模型報告"""
        try:
            report = f"""
🤖 模型訓練報告 - {self.symbol} {self.timeframe}
{'='*50}
📊 模型信息:
├─ 模型類型: LightGBM
├─ 特徵數量: {len(feature_names) if feature_names else '未知'}
└─ 訓練完成

📈 性能指標:"""

            for metric, value in metrics.items():
                report += f"\n├─ {metric.upper()}: {value:.4f}"

            if hasattr(model, 'feature_importances_') and feature_names:
                # 獲取前5重要特徵
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                report += f"\n\n🏆 前5重要特徵:"
                for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                    report += f"\n├─ {i}. {row['feature']:<20} {row['importance']:.4f}"

            return report
                
        except Exception as e:
            return f"模型報告生成失敗: {e}"


# 向後兼容性
ModelOptimizer = ModelTrainer  # 別名，保持向後兼容