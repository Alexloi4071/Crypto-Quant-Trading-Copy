#!/usr/bin/env python3
"""
主優化器 - 簡化版本，專注於核心 ML 流程管理
原 Optuna 優化功能已遷移至 optuna_system/

負責：
- 統一管理標籤生成、特徵選擇、模型訓練流程
- 數據預處理和後處理
- 結果保存和報告生成
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加項目根目錄到Python路徑
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

try:
    from .config import get_config
    from .label_optimizer import LabelGenerator
    from .feature_selector import FeatureSelector
    from .model_optimizer import ModelTrainer
except ImportError:
    from config import get_config
    from label_optimizer import LabelGenerator
    from feature_selector import FeatureSelector
    from model_optimizer import ModelTrainer


class MLPipelineManager:
    """ML 流程管理器 - 簡化版本，專注於核心流程"""

    def __init__(self, symbol: str, timeframe: str, version: str = None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.version = version or f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🚀 初始化 ML 流程管理器 - {symbol} {timeframe}")
        print(f"📊 版本: {self.version}")
        
        # 初始化組件
        self.label_generator = LabelGenerator(symbol, timeframe)
        self.feature_selector = FeatureSelector(symbol, timeframe)
        self.model_trainer = ModelTrainer(symbol, timeframe)
        
        # 設置數據路徑
        self.setup_data_paths()
        
        # 結果存儲
        self.results = {}

    def setup_data_paths(self):
        """設置數據路徑"""
        self.base_dir = Path("data")
        self.raw_dir = self.base_dir / "raw" / self.symbol
        self.processed_dir = self.base_dir / "processed"
        self.features_dir = self.processed_dir / "features" / f"{self.symbol}_{self.timeframe}" / self.version
        self.labels_dir = self.processed_dir / "labels" / f"{self.symbol}_{self.timeframe}" / self.version
        self.models_dir = Path("results") / "models" / f"{self.symbol}_{self.timeframe}_{self.version}"
        
        # 創建目錄
        for dir_path in [self.features_dir, self.labels_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_ohlcv_data(self) -> pd.DataFrame:
        """加載 OHLCV 數據"""
        try:
            ohlcv_file = self.raw_dir / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
            
            if not ohlcv_file.exists():
                raise FileNotFoundError(f"OHLCV 文件不存在: {ohlcv_file}")
            
            print(f"📊 加載 OHLCV 數據: {ohlcv_file}")
            data = pd.read_parquet(ohlcv_file)
            
            # 確保索引是時間戳
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            print(f"✅ OHLCV 數據加載完成: {len(data)} 條記錄")
            
            return data
            
        except Exception as e:
            print(f"❌ OHLCV 數據加載失敗: {e}")
            raise e

    def generate_features_from_raw(self, ohlcv_data: pd.DataFrame = None) -> pd.DataFrame:
        """從原始數據生成技術指標特徵"""
        try:
            if ohlcv_data is None:
                ohlcv_data = self.load_ohlcv_data()
            
            print(f"🔧 生成技術指標特徵...")
            
            # 基本價格特徵
            features = pd.DataFrame(index=ohlcv_data.index)
            
            # 價格變化率
            features['returns'] = ohlcv_data['close'].pct_change()
            features['high_low_ratio'] = ohlcv_data['high'] / ohlcv_data['low']
            features['open_close_ratio'] = ohlcv_data['open'] / ohlcv_data['close']
            
            # 移動平均線
            for window in [5, 10, 20, 50]:
                features[f'sma_{window}'] = ohlcv_data['close'].rolling(window).mean()
                features[f'ema_{window}'] = ohlcv_data['close'].ewm(span=window).mean()
                features[f'price_sma_{window}_ratio'] = ohlcv_data['close'] / features[f'sma_{window}']
            
            # 布林帶
            for window in [20]:
                sma = ohlcv_data['close'].rolling(window).mean()
                std = ohlcv_data['close'].rolling(window).std()
                features[f'bb_upper_{window}'] = sma + (2 * std)
                features[f'bb_lower_{window}'] = sma - (2 * std)
                features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / sma
                features[f'bb_position_{window}'] = (ohlcv_data['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])
            
            # RSI
            for window in [14, 21]:
                delta = ohlcv_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # 成交量特徵
            features['volume_sma_ratio'] = ohlcv_data['volume'] / ohlcv_data['volume'].rolling(20).mean()
            features['price_volume'] = ohlcv_data['close'] * ohlcv_data['volume']
            
            # 波動率
            for window in [10, 20]:
                features[f'volatility_{window}'] = ohlcv_data['close'].pct_change().rolling(window).std()
            
            # 移除無效值
            features = features.fillna(method='bfill').fillna(method='ffill')
            features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            print(f"✅ 技術指標生成完成: {features.shape[1]} 個特徵")
            
            # 保存特徵
            features_file = self.features_dir / f"{self.symbol}_{self.timeframe}_features_full.parquet"
            features.to_parquet(features_file)
            print(f"💾 特徵已保存: {features_file}")
            
            return features
            
        except Exception as e:
            print(f"❌ 特徵生成失敗: {e}")
            raise e

    def run_complete_pipeline(self, 
                            label_params: Dict[str, Any] = None,
                            feature_params: Dict[str, Any] = None,
                            model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        運行完整的 ML 流程
        
        Args:
            label_params: 標籤生成參數
            feature_params: 特徵選擇參數
            model_params: 模型訓練參數
        
        Returns:
            完整結果字典
        """
        try:
            print(f"🚀 開始完整 ML 流程 - {self.symbol} {self.timeframe}")
            
            # 1. 加載數據
            ohlcv_data = self.load_ohlcv_data()
            
            # 2. 生成特徵
            print(f"\n📍 步驟 1: 生成特徵")
            features = self.generate_features_from_raw(ohlcv_data)
            
            # 3. 生成標籤
            print(f"\n📍 步驟 2: 生成標籤")
            default_label_params = {
                'lag': 3,
                'profit_threshold': 0.005,
                'loss_threshold': -0.005,
                'label_type': 'ternary',
                'threshold_method': 'fixed'
            }
            
            if label_params:
                default_label_params.update(label_params)
            
            labels = self.label_generator.generate_labels(
                price_data=ohlcv_data['close'],
                **default_label_params
            )
            
            # 保存標籤
            labels_file = self.labels_dir / f"{self.symbol}_{self.timeframe}_labels.parquet"
            labels.to_frame('label').to_parquet(labels_file)
            
            # 4. 數據對齊
            print(f"\n📍 步驟 3: 數據對齊")
            common_index = features.index.intersection(labels.index)
            X = features.loc[common_index]
            y = labels.loc[common_index]
            
            print(f"📊 對齊後數據: {len(X)} 條記錄, {X.shape[1]} 個特徵")
            
            # 5. 特徵選擇
            print(f"\n📍 步驟 4: 特徵選擇")
            default_feature_params = {
                'n_features': 20,
                'method': 'lgb'
            }
            
            if feature_params:
                default_feature_params.update(feature_params)
            
            # 相關性過濾
            X_filtered = self.feature_selector.remove_correlated_features(X, threshold=0.95)
            
            # 特徵選擇
            selected_features = self.feature_selector.select_features_by_importance(
                X_filtered, y, **default_feature_params
            )
            
            X_selected = X_filtered[selected_features]
            
            # 保存選中的特徵
            selected_features_file = self.features_dir / f"{self.symbol}_{self.timeframe}_selected_features.parquet"
            X_selected.to_parquet(selected_features_file)
            
            # 保存特徵選擇參數
            feature_selection_params = {
                'selected_features': selected_features,
                'n_features': len(selected_features),
                'selection_method': default_feature_params['method'],
                'correlation_threshold': 0.95
            }
            
            params_file = self.features_dir / f"{self.symbol}_{self.timeframe}_feature_selection_params.json"
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(feature_selection_params, f, indent=2, ensure_ascii=False)
            
            # 6. 模型訓練
            print(f"\n📍 步驟 5: 模型訓練")
            
            model, metrics = self.model_trainer.train_with_validation(
                X_selected, y, model_params=model_params
            )
            
            # 7. 保存模型
            print(f"\n📍 步驟 6: 保存結果")
            
            model_path = self.model_trainer.save_model(
                model=model,
                save_dir=str(self.models_dir),
                model_params=model_params,
                feature_names=selected_features,
                metrics=metrics
            )
            
            # 8. 生成報告
            label_report = self.label_generator.generate_report(y)
            feature_report = self.feature_selector.get_feature_importance_report(X_selected, y, selected_features)
            model_report = self.model_trainer.generate_report(model, metrics, selected_features)
            
            # 9. 整合結果
            results = {
                'version': self.version,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'data_info': {
                    'total_samples': len(X),
                    'feature_count': len(selected_features),
                    'selected_features': selected_features
                },
                'label_params': default_label_params,
                'feature_params': default_feature_params,
                'model_params': model_params or self.model_trainer._get_default_params(),
                'metrics': metrics,
                'model_path': model_path,
                'reports': {
                    'labels': label_report,
                    'features': feature_report,
                    'model': model_report
                }
            }
            
            # 保存完整結果
            results_file = self.models_dir / "pipeline_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✅ 完整 ML 流程完成!")
            print(f"📊 最終 F1 分數: {metrics.get('f1', 0):.4f}")
            print(f"💾 結果已保存: {self.models_dir}")
            
            self.results = results
            return results
            
        except Exception as e:
            print(f"❌ ML 流程失敗: {e}")
            raise e

    def generate_summary_report(self) -> str:
        """生成流程總結報告"""
        if not self.results:
            return "尚未運行完整流程"
        
        results = self.results
        
        report = f"""
🎯 ML 流程總結報告
{'='*60}
📊 基本信息:
├─ 交易對: {results['symbol']}
├─ 時間框架: {results['timeframe']}
├─ 版本: {results['version']}
└─ 運行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 數據統計:
├─ 總樣本數: {results['data_info']['total_samples']:,}
├─ 特徵數量: {results['data_info']['feature_count']}
└─ 標籤類型: {results['label_params']['label_type']}

🎯 性能指標:"""

        for metric, value in results['metrics'].items():
            report += f"\n├─ {metric.upper()}: {value:.4f}"

        report += f"""

💾 輸出文件:
├─ 模型: {results['model_path']}
├─ 特徵: {self.features_dir}
└─ 標籤: {self.labels_dir}

🔧 參數設置:
├─ 標籤參數: {results['label_params']}
├─ 特徵參數: {results['feature_params']}
└─ 模型參數數量: {len(results['model_params'])}個
"""

        return report


# 向後兼容性
ModularOptunaOptimizer = MLPipelineManager  # 別名，保持向後兼容
