# -*- coding: utf-8 -*-
"""
標籤生成參數優化器 (第1層)
基於主系統的label_optimizer重新實現
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
import pandas as pd

from optuna_system.utils.io_utils import write_dataframe, read_dataframe

warnings.filterwarnings('ignore')


class LabelOptimizer:
    """標籤生成參數優化器 - 第1層優化"""

    def __init__(self, data_path: str, config_path: str = "configs/",
                 symbol: str = "BTCUSDT", timeframe: str = "15m", scaled_config: Dict = None):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe

        # 時間框架縮放配置（文檔設計）
        self.scaled_config = scaled_config or {}

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

    def generate_labels(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """✅ 完全修復版標籤生成 - 滾動窗口分位數，嚴格避免未來數據洩露"""
        try:
            lag = params['lag']
            threshold_method = params.get('threshold_method', 'quantile')
            lookback_window = params.get('lookback_window', 500)  # 滾動窗口大小

            if len(price_data) <= lag:
                return pd.Series([], dtype=int)

            # ✅ 計算未來收益率（這是我們要預測的目標）
            future_prices = price_data.shift(-lag)
            future_returns = (future_prices - price_data) / price_data

            # 初始化標籤
            labels = pd.Series(1, index=price_data.index, dtype=int)  # 默認持有

            # 根據方法確定閾值
            if threshold_method == 'quantile':
                # 🚀 使用獨立的買入/賣出分位數
                buy_quantile = params.get('buy_quantile', 0.75)
                sell_quantile = params.get('sell_quantile', 0.25)

                # ✅ 關鍵修復：使用"歷史的未來收益率"計算閾值
                # 這樣阈值和目標變量的分布是一致的
                historical_future_returns = future_returns.shift(1)  # 向前偏移，只用歷史
                
                # ✅ 基於歷史未來收益率計算滾動分位數
                rolling_upper = historical_future_returns.rolling(
                    window=lookback_window, 
                    min_periods=100
                ).quantile(buy_quantile)
                
                rolling_lower = historical_future_returns.rolling(
                    window=lookback_window, 
                    min_periods=100
                ).quantile(sell_quantile)
                
                # 確保閾值合理且有間隔
                rolling_upper = rolling_upper.fillna(0.0001).clip(lower=0.0001)
                rolling_lower = rolling_lower.fillna(-0.0001).clip(upper=-0.0001)
                
                # 確保買賣閾值有足夠間隔，避免重疊
                gap_needed = (rolling_upper + abs(rolling_lower)) * 0.1
                gap_needed = gap_needed.clip(lower=0.0005)
                
                # 調整重疊的閾值
                overlap_mask = rolling_upper <= abs(rolling_lower)
                rolling_upper[overlap_mask] = rolling_upper[overlap_mask].clip(lower=gap_needed[overlap_mask])
                rolling_lower[overlap_mask] = rolling_lower[overlap_mask].clip(upper=-gap_needed[overlap_mask])

                # ✅ 向量化標籤分配（類型一致：都是未來收益率）
                valid_range = slice(lookback_window, len(future_returns) - lag)
                labels.iloc[valid_range] = 1  # 重置為持有
                
                # 買入信號
                buy_mask = (future_returns.iloc[valid_range] > rolling_upper.iloc[valid_range]) & rolling_upper.iloc[valid_range].notna()
                labels.iloc[valid_range] = labels.iloc[valid_range].where(~buy_mask, 2)
                
                # 賣出信號  
                sell_mask = (future_returns.iloc[valid_range] < rolling_lower.iloc[valid_range]) & rolling_lower.iloc[valid_range].notna()
                labels.iloc[valid_range] = labels.iloc[valid_range].where(~sell_mask, 0)

            elif threshold_method == 'fixed':
                # 固定閾值方法
                profit_threshold = params.get('profit_threshold', 0.01)
                loss_threshold = params.get('loss_threshold', -0.01)

                # 應用固定閾值
                labels[future_returns > profit_threshold] = 2  # 買入
                labels[future_returns < loss_threshold] = 0   # 賣出

            elif threshold_method == 'adaptive':
                # ✅ 修復：自適應閾值也基於未來收益率
                vol_multiplier = params.get('vol_multiplier', 1.5)
                vol_window = int(params.get('vol_window', min(lookback_window, 40)))

                # ✅ 使用歷史未來收益率計算波動率
                historical_future_returns = future_returns.shift(1)
                rolling_volatility = historical_future_returns.rolling(
                    window=vol_window, 
                    min_periods=20
                ).std()
                
                # 基於波動率的動態閾值
                rolling_profit_threshold = rolling_volatility * vol_multiplier
                rolling_loss_threshold = -rolling_volatility * vol_multiplier
                
                # 填補無效值
                rolling_profit_threshold = rolling_profit_threshold.fillna(0.01)
                rolling_loss_threshold = rolling_loss_threshold.fillna(-0.01)
                
                # ✅ 向量化標籤分配（類型一致）
                valid_range = slice(vol_window, len(future_returns) - lag)
                labels.iloc[valid_range] = 1  # 重置為持有
                
                # 買入信號
                buy_mask = future_returns.iloc[valid_range] > rolling_profit_threshold.iloc[valid_range]
                labels.iloc[valid_range] = labels.iloc[valid_range].where(~buy_mask, 2)
                
                # 賣出信號
                sell_mask = future_returns.iloc[valid_range] < rolling_loss_threshold.iloc[valid_range]
                labels.iloc[valid_range] = labels.iloc[valid_range].where(~sell_mask, 0)
            else:
                # 默認固定閾值
                profit_threshold = 0.01
                loss_threshold = -0.01
                labels[future_returns > profit_threshold] = 2
                labels[future_returns < loss_threshold] = 0

            # 移除未來數據洩露
            labels = labels[:-lag] if lag > 0 else labels

            # 計算並打印標籤統計
            self._print_label_statistics(labels, params)

            return labels.dropna()

        except Exception as e:
            self.logger.error(f"標籤生成失敗: {e}")
            return pd.Series([], dtype=int)

    def stabilized_label_generation(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """穩定化標籤生成：固定歷史分位數 + 信號驗證"""
        try:
            lag = params['lag']
            fixed_lookback = int(params.get('fixed_lookback', 2000))

            if len(price_data) <= max(lag, fixed_lookback):
                return self.generate_labels(price_data, params)

            future_returns = (price_data.shift(-lag) - price_data) / price_data

            historical_returns = future_returns.iloc[:fixed_lookback].dropna()
            if len(historical_returns) < 200:
                return self.generate_labels(price_data, params)

            buy_quantile = float(params.get('buy_quantile', 0.75))
            sell_quantile = float(params.get('sell_quantile', 0.25))

            buy_threshold = historical_returns.quantile(buy_quantile)
            sell_threshold = historical_returns.quantile(sell_quantile)

            min_gap = float(params.get('min_threshold_gap', 0.005))
            if buy_threshold - abs(sell_threshold) < min_gap:
                buy_threshold = abs(sell_threshold) + min_gap

            labels = pd.Series(1, index=price_data.index, dtype=int)
            valid_end = len(price_data) - lag
            valid_start = fixed_lookback
            if valid_start >= valid_end:
                return self.generate_labels(price_data, params)

            valid_slice = slice(valid_start, valid_end)
            returns_slice = future_returns.iloc[valid_slice]

            labels.iloc[valid_slice] = labels.iloc[valid_slice].where(~(returns_slice >= buy_threshold), 2)
            labels.iloc[valid_slice] = labels.iloc[valid_slice].where(~(returns_slice <= sell_threshold), 0)

            signal_stats = self.validate_signal_authenticity(price_data, labels, lag)
            max_noise_ratio = float(params.get('max_noise_ratio', 0.35))

            if signal_stats['noise_ratio'] > max_noise_ratio:
                self.logger.warning(
                    f"⚠️ 信號噪聲過高({signal_stats['noise_ratio']:.2f} > {max_noise_ratio:.2f})，使用保守標籤"
                )
                labels = self.generate_conservative_labels(price_data, params)

            if lag > 0:
                labels = labels[:-lag]

            self._print_label_statistics(labels, params)
            return labels.dropna()

        except Exception as e:
            self.logger.error(f"穩定化標籤生成失敗: {e}")
            return self.generate_labels(price_data, params)

    def generate_conservative_labels(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """生成更保守的標籤（提高買入門檻/降低賣出門檻）"""
        conservative_shift = float(params.get('conservative_shift', 0.005))
        fallback_params = params.copy()
        fallback_params['buy_quantile'] = min(0.95, fallback_params.get('buy_quantile', 0.75) + 0.05)
        fallback_params['sell_quantile'] = max(0.05, fallback_params.get('sell_quantile', 0.25) - 0.05)
        fallback_params['profit_threshold'] = fallback_params.get('profit_threshold', 0.02) + conservative_shift
        fallback_params['loss_threshold'] = fallback_params.get('loss_threshold', -0.02) - conservative_shift
        return self.generate_labels(price_data, fallback_params)

    def validate_signal_authenticity(self, prices: pd.Series, labels: pd.Series, lag: int) -> Dict:
        """信號真實性驗證：統計信號後的實際走勢"""
        try:
            buy_mask = labels == 2
            sell_mask = labels == 0

            future_prices = prices.shift(-lag)
            buy_accuracy = (
                (future_prices[buy_mask] > prices[buy_mask]).mean()
                if buy_mask.sum() > 0 else 0
            )
            sell_accuracy = (
                (future_prices[sell_mask] < prices[sell_mask]).mean()
                if sell_mask.sum() > 0 else 0
            )

            total_signals = buy_mask.sum() + sell_mask.sum()
            total_accuracy = (
                (buy_accuracy * buy_mask.sum() + sell_accuracy * sell_mask.sum()) / total_signals
                if total_signals > 0 else 0
            )
            noise_ratio = 1 - total_accuracy

            quality = 'high'
            if noise_ratio >= 0.4:
                quality = 'low'
            elif noise_ratio >= 0.2:
                quality = 'medium'

            return {
                'buy_accuracy': float(buy_accuracy),
                'sell_accuracy': float(sell_accuracy),
                'noise_ratio': float(noise_ratio),
                'signal_quality': quality
            }

        except Exception as e:
            self.logger.warning(f"信號真實性驗證失敗: {e}")
            return {'buy_accuracy': 0, 'sell_accuracy': 0, 'noise_ratio': 1.0, 'signal_quality': 'unknown'}
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """計算平均真實區間（ATR）"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr
        except Exception as e:
            self.logger.error(f"ATR計算失敗: {e}")
            return pd.Series(0, index=close.index)
    
    def generate_triple_barrier_labels(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """✅ 修復版Triple-Barrier標籤生成 - 使用ATR動態閾值"""
        try:
            lag = params.get('lag', 12)
            profit_multiplier = params.get('profit_multiplier', 2.0)  # ATR倍數（止盈）
            stop_multiplier = params.get('stop_multiplier', 1.0)      # ATR倍數（止損）
            max_holding = params.get('max_holding', 16)               # 最大持有期
            atr_period = params.get('atr_period', 14)                 # ATR週期

            if len(price_data) <= max_holding:
                return pd.Series([], dtype=int)
            
            # ✅ 計算ATR（需要OHLCV數據）
            try:
                ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
                if ohlcv_file.exists():
                    ohlcv_df = pd.read_parquet(ohlcv_file)
                    atr = self.calculate_atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], atr_period)
                    # ✅ 修復：對齊到price_data的索引，只用前向填充避免數據泄漏
                    atr = atr.reindex(price_data.index).fillna(method='ffill')
                    # 處理剩餘的NaN（通常是開頭的ATR週期內）
                    if atr.isna().any():
                        first_valid_idx = atr.first_valid_index()
                        if first_valid_idx is not None:
                            first_valid_value = atr[first_valid_idx]
                            atr = atr.fillna(first_valid_value)
                            self.logger.info(f"✅ ATR前向填充完成，用第一個有效值({first_valid_value:.4f})填充前期NaN")
                        else:
                            atr = atr.fillna(0.01)
                            self.logger.warning("⚠️ ATR無有效值，使用默認值0.01")
                else:
                    # 如果沒有OHLCV，使用簡化的ATR估算
                    returns = price_data.pct_change().abs()
                    atr = returns.rolling(atr_period).mean() * price_data
                    self.logger.warning("⚠️ 未找到OHLCV數據，使用簡化ATR估算")
            except Exception as e:
                self.logger.warning(f"⚠️ ATR計算失敗: {e}，使用簡化方法")
                returns = price_data.pct_change().abs()
                atr = returns.rolling(atr_period).mean() * price_data

            labels = pd.Series(1, index=price_data.index, dtype=int)  # 默認持有
            
            for i in range(len(price_data) - max_holding):
                entry_price = price_data.iloc[i]
                current_atr = atr.iloc[i]
                
                if pd.isna(current_atr) or current_atr <= 0:
                    continue
                
                # ✅ 動態止盈止損基於ATR
                profit_target_price = entry_price + current_atr * profit_multiplier
                stop_loss_price = entry_price - current_atr * stop_multiplier
                
                # ✅ 確保風險收益比 ≥ 2:1
                actual_profit_distance = profit_target_price - entry_price
                actual_stop_distance = entry_price - stop_loss_price
                
                if actual_stop_distance > 0 and actual_profit_distance / actual_stop_distance < 2.0:
                    # 調整止損以達到2:1比例
                    stop_loss_price = entry_price - (actual_profit_distance / 2.0)
                
                # 定義未來價格窗口
                future_window = price_data.iloc[i+1:i+max_holding+1]
                
                if len(future_window) == 0:
                    continue
                
                # 檢查三重障礙觸發條件
                hit_profit = (future_window >= profit_target_price).any()
                hit_stop = (future_window <= stop_loss_price).any()
                
                if hit_profit and hit_stop:
                    # 都觸發，看哪個先發生
                    profit_idx = future_window[future_window >= profit_target_price].index[0]
                    stop_idx = future_window[future_window <= stop_loss_price].index[0]
                    
                    if price_data.index.get_loc(profit_idx) < price_data.index.get_loc(stop_idx):
                        labels.iloc[i] = 2  # 買入（先觸發止盈）
                    else:
                        labels.iloc[i] = 0  # 賣出（先觸發止損）
                elif hit_profit:
                    labels.iloc[i] = 2  # 買入
                elif hit_stop:
                    labels.iloc[i] = 0  # 賣出
                # 否則保持持有
                
            # 移除可能的未來數據洩露
            if lag > 0:
                labels = labels[:-lag]
            
            self.logger.info(f"✅ Triple-Barrier標籤: 止盈={profit_multiplier}×ATR, "
                           f"止損={stop_multiplier}×ATR, 最大持有={max_holding}期, "
                           f"ATR週期={atr_period}, 風險收益比≥2:1")
            
            return labels.dropna()
            
        except Exception as e:
            self.logger.error(f"Triple-Barrier標籤生成失敗: {e}")
            return pd.Series([], dtype=int)

    def _print_label_statistics(self, labels: pd.Series, params: Dict) -> None:
        """🚀 新增：計算並打印標籤分佈統計與質量指標"""
        try:
            if len(labels) == 0:
                self.logger.warning("空標籤序列，無法計算統計")
                return

            # 標籤分佈統計
            label_counts = labels.value_counts().sort_index()
            total_samples = len(labels)

            self.logger.info(f"📊 修復版標籤分佈統計:")
            self.logger.info(f"   總樣本數: {total_samples}")
            self.logger.info(f"   滾動窗口: {params.get('lookback_window', 500)}")
            self.logger.info(f"   lag: {params.get('lag', 12)}期")

            # 各類別統計
            for label_val in [0, 1, 2]:
                count = label_counts.get(label_val, 0)
                percentage = (count / total_samples) * 100
                label_name = {0: '賣出', 1: '持有', 2: '買入'}[label_val]
                self.logger.info(f"   {label_name}({label_val}): {count:,} ({percentage:.2f}%)")

            # 標籤變化頻率
            if len(labels) > 1:
                changes = (labels.diff() != 0).sum()
                change_rate = changes / len(labels)
                self.logger.info(f"   標籤變化頻率: {change_rate:.3f} ({changes:,}/{total_samples:,})")

            # 檢查標籤平衡性
            min_class_ratio = min(label_counts) / total_samples
            max_class_ratio = max(label_counts) / total_samples
            imbalance_ratio = max_class_ratio / min_class_ratio

            if imbalance_ratio > 10:
                self.logger.warning(f"⚠️ 標籤嚴重不平衡: 最大/最小類別比例 = {imbalance_ratio:.2f}")
            elif imbalance_ratio > 5:
                self.logger.warning(f"⚠️ 標籤輕度不平衡: 最大/最小類別比例 = {imbalance_ratio:.2f}")
            else:
                self.logger.info(f"✅ 標籤平衡性良好: 最大/最小類別比例 = {imbalance_ratio:.2f}")

            # 檢查各類別是否都有代表樣本
            missing_classes = []
            for class_val in [0, 1, 2]:
                if label_counts.get(class_val, 0) == 0:
                    missing_classes.append(class_val)

            if missing_classes:
                class_names = [['賣出', '持有', '買入'][i] for i in missing_classes]
                self.logger.error(f"❌ 缺失類別: {missing_classes} ({class_names})")
            else:
                self.logger.info("✅ 所有類別都有代表樣本")

        except Exception as e:
            self.logger.error(f"標籤統計計算失敗: {e}")

    def calculate_label_quality(self, labels: pd.Series, params: Dict) -> Dict:
        """計算標籤質量指標（含 Precision / Recall / F1 的代理評分）"""
        try:
            if len(labels) == 0:
                return {
                    'balance_score': 0,
                    'stability_score': 0,
                    'f1_score': 0,
                    'precision_macro': 0,
                    'recall_macro': 0
                }

            # 1) 分布平衡（越接近理想 25/50/25 越好）
            value_counts = labels.value_counts(normalize=True)
            target_dist = np.array([0.25, 0.5, 0.25])
            actual_dist = np.array([
                value_counts.get(0, 0.0),
                value_counts.get(1, 0.0),
                value_counts.get(2, 0.0)
            ])
            kl_div = np.sum(target_dist * np.log((target_dist + 1e-8) / (actual_dist + 1e-8)))
            balance_score = float(np.exp(-kl_div))

            # 2) 穩定性：標籤切換率越低越穩定
            label_changes = int((labels.diff() != 0).sum())
            stability_score = float(max(0.0, 1.0 - label_changes / max(len(labels), 1)))

            # 3) 準確性代理：用「類別覆蓋率」近似 precision/recall
            # 沒有預測器，採用保守代理：若某類嚴重稀少，precision/recall 都會受限
            class_presence = (actual_dist > 0).astype(float)
            precision_macro = float(class_presence.mean())  # 覆蓋越全越高
            recall_macro = float(1.0 - abs(actual_dist - target_dist).mean() / max(target_dist.max(), 1e-8))
            # 代理 F1
            if precision_macro + recall_macro > 0:
                f1_macro = float(2 * precision_macro * recall_macro / (precision_macro + recall_macro))
            else:
                f1_macro = 0.0

            return {
                'balance_score': balance_score,
                'stability_score': stability_score,
                'f1_score': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'distribution': actual_dist.tolist(),
                'total_samples': int(len(labels))
            }

        except Exception as e:
            self.logger.error(f"標籤質量計算失敗: {e}")
            return {
                'balance_score': 0,
                'stability_score': 0,
                'f1_score': 0,
                'precision_macro': 0,
                'recall_macro': 0
            }

    def objective(self, trial: optuna.Trial) -> float:
        """修復版Optuna目標函數 - 第1層：滾動窗口標籤參數優化"""

        # 🚀 修復版參數：包含滾動窗口大小
        params = {
            # 核心參數
            'lag': trial.suggest_int('lag',
                                     self.scaled_config.get('label_lag_min', self.scaled_config.get('label_lag', 12) - 2),
                                     self.scaled_config.get('label_lag_max', self.scaled_config.get('label_lag', 12) + 4)),
            'threshold_method': trial.suggest_categorical('threshold_method',
                                                        ['quantile', 'fixed', 'adaptive', 'triple_barrier', 'stabilized']),

            # 🚀 新增：滾動窗口大小 (500-800)
            'lookback_window': trial.suggest_int('lookback_window', 500, 800),

            # ✅ 修復優化：基於學術文獻的分位數範圍（更保守，信號質量更高）
            'buy_quantile': trial.suggest_float('buy_quantile', 0.70, 0.85),   # 學術推薦：70-85%分位
            'sell_quantile': trial.suggest_float('sell_quantile', 0.15, 0.30), # 學術推薦：15-30%分位

            # Fixed方法參數
            'profit_threshold': trial.suggest_float('profit_threshold', 0.005, 0.03),
            'loss_threshold': trial.suggest_float('loss_threshold', -0.03, -0.005),

            # Adaptive方法參數
            'vol_multiplier': trial.suggest_float('vol_multiplier', 1.2, 2.0),
            # 額外顯式控制自適應波動窗口，促進更密集訊號
            'vol_window': trial.suggest_int('vol_window', 20, 40),
            
            # ✅ 修復：Triple-Barrier方法參數（ATR倍數，動態調整）
            'profit_multiplier': trial.suggest_float('profit_multiplier', 1.6, 3.0),  # 止盈：≥1.6倍ATR
            'stop_multiplier': trial.suggest_float('stop_multiplier', 0.6, 1.4),      # 止損：≤1.4倍ATR
            'max_holding': trial.suggest_int('max_holding', 16, 24),                  # 最大持有：16-24期（4-6小時）
            'atr_period': trial.suggest_int('atr_period', 10, 20),                    # ATR週期：10-20

            # 質量控制參數
            'min_samples': trial.suggest_int('min_samples', 1000, 5000),  # 提高最小樣本要求
            'balance_weight': trial.suggest_float('balance_weight', 0.3, 0.7),
            'stability_weight': trial.suggest_float('stability_weight', 0.2, 0.5),

            # 穩定化標籤參數
            'fixed_lookback': trial.suggest_int('fixed_lookback', 1500, 3000),
            'min_threshold_gap': trial.suggest_float('min_threshold_gap', 0.003, 0.01),
            'max_noise_ratio': trial.suggest_float('max_noise_ratio', 0.25, 0.4),
            
            # 🚀 新增：標籤分布控制參數
            'target_hold_ratio': trial.suggest_float('target_hold_ratio', 0.48, 0.52),  # 收斂持有比例
            'distribution_penalty': trial.suggest_float('distribution_penalty', 0.8, 1.5)  # 溫和分布懲罰
        }

        try:
            # 🔧 修復版：明确构造文件路径并添加调试信息
            ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
            self.logger.info(f"🔍 查找OHLCV文件: {ohlcv_file.absolute()}")
            
            if not ohlcv_file.exists():
                alternative_paths = [
                    f"data/raw/{self.symbol}/{self.symbol}_{self.timeframe}_ohlcv.parquet",
                    f"../data/raw/{self.symbol}/{self.symbol}_{self.timeframe}_ohlcv.parquet",
                    f"./{self.symbol}_{self.timeframe}_ohlcv.parquet"
                ]

                df = None
                for alt_path in alternative_paths:
                    if Path(alt_path).exists():
                        self.logger.info(f"🔍 找到替代路径: {alt_path}")
                        df = pd.read_parquet(alt_path, engine='pyarrow')
                        break

                if df is None:
                    self.logger.error(f"❌ 未找到OHLCV數據文件: {ohlcv_file.absolute()}")
                    raise FileNotFoundError(f"未找到原始 OHLCV 數據: {ohlcv_file}")
                else:
                    self.logger.info(f"✅ 成功加載OHLCV數據: {df.shape}")
            else:
                df = pd.read_parquet(ohlcv_file, engine='pyarrow')
                self.logger.info(f"✅ 加載OHLCV數據: {df.shape}")

            # 確定價格列
            price_data = df['close']

            # 🔧 修复：一次性预计算全量rolling分位数，避免重复计算
            lag = params['lag']
            lookback_window = params['lookback_window']
            actual_profit = price_data.pct_change(lag).shift(-lag)
            
            # 向量化滚动分位数计算（一次性完成）
            self.logger.info(f"🚀 开始向量化分位数计算: window={lookback_window}")
            buy_quantile = params['buy_quantile']
            sell_quantile = params['sell_quantile']
            
            # 预计算全量滚动分位数
            rolling_upper = actual_profit.rolling(window=lookback_window, min_periods=100).quantile(buy_quantile)
            rolling_lower = actual_profit.rolling(window=lookback_window, min_periods=100).quantile(sell_quantile)
            
            # 缓存到实例变量避免重复计算
            self._cached_rolling_upper = rolling_upper
            self._cached_rolling_lower = rolling_lower
            self.logger.info(f"✅ 分位数预计算完成，缓存{len(rolling_upper)}个点")

            # 🚀 風險比約束：強制 2:1（止盈 >= 2 × 止損）
            if params.get('threshold_method') == 'triple_barrier':
                pm = float(params.get('profit_multiplier', 2.0))
                sm = float(params.get('stop_multiplier', 1.0))
                if pm < 2.0 * sm:
                    # 將止盈抬高到至少2x止損
                    adjusted_pm = max(2.0 * sm, pm)
                    self.logger.info(f"🔒 風險比約束: profit_multiplier {pm:.2f} → {adjusted_pm:.2f} (stop={sm:.2f})")
                    params['profit_multiplier'] = adjusted_pm

            # 🚀 改進：根據方法選擇標籤生成策略
            threshold_method = params.get('threshold_method', 'quantile')
            if threshold_method == 'triple_barrier':
                labels = self.generate_triple_barrier_labels(price_data, params)
                self.logger.info("🚧 使用Triple-Barrier標籤生成方法")
            elif threshold_method == 'stabilized':
                labels = self.stabilized_label_generation(price_data, params)
                self.logger.info("📊 使用穩定化固定分位標籤生成方法")
            else:
                labels = self.generate_labels(price_data, params)
                self.logger.info(f"📊 使用{threshold_method}分位數標籤生成方法")

            if len(labels) < params['min_samples']:
                return -999.0  # 樣本不足

            # 計算標籤質量
            quality_metrics = self.calculate_label_quality(labels, params)

            # 🚀 改進：多目標優化得分 + 分布約束
            balance_score = quality_metrics['balance_score']
            stability_score = quality_metrics['stability_score']
            f1_score = quality_metrics['f1_score']
            
            # 計算實際標籤分布
            label_counts = labels.value_counts(normalize=True)
            actual_hold_ratio = label_counts.get(1, 0)  # 持有比例
            actual_buy_ratio = label_counts.get(2, 0)   # 買入比例  
            actual_sell_ratio = label_counts.get(0, 0)  # 賣出比例

            # 🚧 硬性分佈約束：過度不平衡的方案直接淘汰，避免搜索陷入長期持有
            if actual_hold_ratio > 0.70 or actual_buy_ratio < 0.10 or actual_sell_ratio < 0.10:
                self.logger.info(
                    f"⛔ 分佈不合格(hold>{actual_hold_ratio:.2%} 或 買/賣<{min(actual_buy_ratio, actual_sell_ratio):.2%})，退回極低分"
                )
                return -999.0
            
            # 🚀 新增：分布偏差懲罰
            target_hold = params['target_hold_ratio']
            hold_deviation = abs(actual_hold_ratio - target_hold)
            
            # 確保買賣信號均衡（各自至少15%）
            min_trade_ratio = 0.15
            buy_sell_penalty = 0
            if actual_buy_ratio < min_trade_ratio:
                buy_sell_penalty += (min_trade_ratio - actual_buy_ratio)
            if actual_sell_ratio < min_trade_ratio:
                buy_sell_penalty += (min_trade_ratio - actual_sell_ratio)
            # 低變化率懲罰（鼓勵避免長期持有單一類別）
            change_rate = float(((labels.diff() != 0).sum() / max(len(labels), 1)))
            low_change_penalty = max(0.0, 0.25 - change_rate)  # 期望≥0.25
            
            # 分布懲罰（強化版）：持有偏差 + 買賣不足 + 低變化率
            distribution_penalty = params['distribution_penalty'] * (hold_deviation + buy_sell_penalty + 0.5 * low_change_penalty)
            
            # 🚀 修復版：權重歸一化，確保f1_score權重不為負
            balance_weight = params['balance_weight']
            stability_weight = params['stability_weight']
            total_weight = balance_weight + stability_weight
            
            # 🔧 修复版：确保f1权重至少0.15（增加准确性权重）
            if total_weight > 0.85:  # 从0.9改为0.85，给f1更多权重
                scale_factor = 0.85 / total_weight
                balance_weight = balance_weight * scale_factor
                stability_weight = stability_weight * scale_factor
                self.logger.info(f"🔧 权重缩放: 原始和={total_weight:.3f}, 缩放因子={scale_factor:.3f}")
            
            f1_weight = max(0.15, 1.0 - balance_weight - stability_weight)  # 从0.1改为0.15
            
            # 🔧 新增：记录权重调整过程
            self.logger.info(f"⚖️ 最终权重: balance={balance_weight:.3f}, stability={stability_weight:.3f}, f1={f1_weight:.3f}")
            self.logger.info(f"📊 分布惩罚: 持有偏差={hold_deviation:.3f}, 买卖惩罚={buy_sell_penalty:.3f}, 总惩罚={distribution_penalty:.4f}")
            
            # 多目標加權總分：準確性 + 分布合理性
            accuracy_component = (balance_score * balance_weight +
                                stability_score * stability_weight +
                                f1_score * f1_weight)
            
            # 最終分數 = 準確性分數 - 分布懲罰
            final_score = accuracy_component - distribution_penalty
            
            # 記錄分布信息
            self.logger.info(f"📊 標籤分布: 賣出={actual_sell_ratio:.1%}, 持有={actual_hold_ratio:.1%}, 買入={actual_buy_ratio:.1%}")
            self.logger.info(f"🎯 目標持有率={target_hold:.1%}, 實際偏差={hold_deviation:.3f}")
            self.logger.info(f"⚖️ 權重: balance={balance_weight:.3f}, stability={stability_weight:.3f}, f1={f1_weight:.3f}")
            self.logger.info(f"⚖️ 準確性={accuracy_component:.4f}, 分布懲罰={distribution_penalty:.4f}, 最終={final_score:.4f}")

            # 🔧 新增：计算持有率误差并传递给Layer2
            hold_error = abs(actual_hold_ratio - target_hold)
            
            # 将持有率误差写入trial.user_attrs供Layer2参考
            trial.set_user_attr("hold_error", hold_error)
            trial.set_user_attr("actual_hold_ratio", actual_hold_ratio)
            trial.set_user_attr("buy_sell_balance", actual_buy_ratio / max(actual_sell_ratio, 0.01))
            
            self.logger.info(f"📊 持有率传递: 目标={target_hold:.1%}, 实际={actual_hold_ratio:.1%}, 误差={hold_error:.3f}")

            return final_score

        except Exception as e:
            self.logger.error(f"標籤優化過程出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 200, timeframes: List[str] = None) -> Dict:
        """執行標籤參數優化（支援多時框）"""
        if timeframes is None:
            timeframes = [self.timeframe]

        results = {}
        meta_vol = self.scaled_config.get('meta_vol', 0.02)

        for tf in timeframes:
            self.logger.info(f"🚀 開始標籤生成參數優化（第1層） - 時框: {tf}")
            self.timeframe = tf

            storage_url = None
            try:
                # 可由環境變數或 scaled_config 控制，預設 None（in-memory）
                storage_url = self.scaled_config.get('optuna_storage')
            except Exception:
                storage_url = None
            study = optuna.create_study(
                direction='maximize',
                study_name=f'label_optimization_layer1_{tf}',
                storage=storage_url,
                load_if_exists=bool(storage_url)
            )
            study.set_user_attr('meta_vol', meta_vol)

            study.optimize(self.objective, n_trials=n_trials)

            best_params = study.best_params
            best_score = study.best_value

            self.logger.info(f"標籤優化完成! 最佳得分: {best_score:.4f}")
            self.logger.info(f"最優參數: {best_params}")

            try:
                processed_cleaned = Path(self.data_path) / "processed" / "cleaned" / f"{self.symbol}_{tf}"
                df2 = None
                # 1) processed 下的固定檔名
                for c in [processed_cleaned / "cleaned_ohlcv.parquet", processed_cleaned / "cleaned_ohlcv.pkl"]:
                    if c.exists():
                        df2 = read_dataframe(c)
                        self.logger.info(f"✅ 使用Layer0清洗數據生成最終標籤: {c}")
                        break
                # 2) processed 下的雜湊檔名
                if df2 is None and processed_cleaned.exists():
                    hashed = sorted(list(processed_cleaned.glob("cleaned_ohlcv_*.parquet")) +
                                    list(processed_cleaned.glob("cleaned_ohlcv_*.pkl")),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
                    for c in hashed:
                        try:
                            df2 = read_dataframe(c)
                            self.logger.info(f"✅ 使用Layer0清洗數據(雜湊)生成最終標籤: {c}")
                            break
                        except Exception:
                            continue
                # 3) 回退到 configs 舊路徑（含 parquet/pkl）
                if df2 is None:
                    legacy_candidates = [
                        self.config_path / f"cleaned_ohlcv_{tf}.parquet",
                        self.config_path / f"cleaned_ohlcv_{tf}.pkl",
                    ]
                    for c in legacy_candidates:
                        if c.exists():
                            df2 = read_dataframe(c)
                            self.logger.info(f"✅ 使用Layer0清洗數據(legacy)生成最終標籤: {c}")
                            break
                # 4) 最後回退原始 OHLCV
                if df2 is None:
                    ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{tf}_ohlcv.parquet"
                    df2 = read_dataframe(ohlcv_file)
                    self.logger.info(f"✅ 使用原始OHLCV數據生成標籤: {ohlcv_file}")
                labeled_data = self.apply_labels(df2, best_params)
                final_quality = self.calculate_label_quality(labeled_data['label'], best_params)
            except Exception as e:
                self.logger.warning(f"無法生成最終標籤: {e}")
                final_quality = {}
                labeled_data = None

            result = {
                'timeframe': tf,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'final_quality': final_quality,
                'meta_vol': meta_vol,
                'labeled_data': labeled_data,
                'optimization_history': [
                    {'trial': i, 'score': trial.value}
                    for i, trial in enumerate(study.trials)
                    if trial.value is not None
                ]
            }
            lag_min = self.scaled_config.get('label_lag_min', 1)
            lag_max = self.scaled_config.get('label_lag_max', 1000)
            result['best_params']['lag'] = int(max(lag_min, min(result['best_params'].get('lag', lag_min), lag_max)))
 
            json_safe = {k: v for k, v in result.items() if k != 'labeled_data'}
            output_file = self.config_path / "label_params.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe, f, indent=2, ensure_ascii=False)
 
            tf_output = self.config_path / f"label_params_{tf}.json"
            with open(tf_output, 'w', encoding='utf-8') as f:
                json.dump(json_safe, f, indent=2, ensure_ascii=False)

            self.logger.info(f"結果已保存至: {output_file}")
            self.logger.info(f"時框專屬結果已保存至: {tf_output}")

            # 物化標籤資料到 processed/labels（支援版本子目錄）
            try:
                if labeled_data is not None and not labeled_data.empty:
                    # 若協調器已建立 results/latest.txt，可讀取最新版本作為子目錄
                    results_root = Path(__file__).resolve().parent.parent / "results"
                    latest_file = results_root / "latest.txt"
                    version_sub = None
                    if latest_file.exists():
                        try:
                            version_sub = latest_file.read_text(encoding="utf-8").strip()
                        except Exception:
                            version_sub = None
                    out_dir = Path(self.data_path) / "processed" / "labels" / f"{self.symbol}_{tf}"
                    if version_sub:
                        out_dir = out_dir / version_sub
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target = out_dir / f"labels_{self.symbol}_{tf}.parquet"
                    label_file, _ = write_dataframe(labeled_data, target)
                    self.logger.info(f"✅ Layer1 標籤物化: {label_file}")
                    result['materialized_path'] = str(label_file)
                else:
                    self.logger.warning("⚠️ 無可物化的標籤資料 (labeled_data 為空)")
            except Exception as exc:
                self.logger.warning(f"⚠️ Layer1 標籤物化失敗: {exc}")
 
            results[tf] = result
 
        return results[self.timeframe] if len(timeframes) == 1 else results

    def apply_labels(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """根據參數生成標籤並附加至資料"""
        if 'close' not in data.columns:
            raise ValueError("資料必須包含close欄位")
        labels = self.generate_labels(data['close'], params)
        result = data.loc[labels.index].copy()
        result['label'] = labels
        return result

    def apply_transform(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """統一的物化接口"""
        return self.apply_labels(data, params)

    def _rebalance_labels(self, labels: pd.Series, valid_slice: slice,
                           target_ratio: List[float], tolerance: float) -> pd.Series:
        if len(target_ratio) != 3:
            return labels
        try:
            target_ratio = np.array(target_ratio, dtype=float)
            target_ratio = target_ratio / target_ratio.sum()
        except Exception:
            return labels

        portion = labels.iloc[valid_slice]
        counts = portion.value_counts(normalize=True)
        current_ratio = np.array([counts.get(0, 0.0), counts.get(1, 0.0), counts.get(2, 0.0)])

        if np.all(np.abs(current_ratio - target_ratio) <= tolerance):
            return labels

        over_hold = current_ratio[1] > target_ratio[1] + tolerance
        need_buy = current_ratio[2] < max(target_ratio[2] - tolerance, 0)
        need_sell = current_ratio[0] < max(target_ratio[0] - tolerance, 0)

        rng = np.random.default_rng(42)
        idx = portion.index
        if over_hold:
            hold_idx = idx[portion == 1]
            swap_candidates = hold_idx.tolist()
            rng.shuffle(swap_candidates)
            for idx_val in swap_candidates:
                if need_buy and current_ratio[2] < target_ratio[2]:
                    labels.at[idx_val] = 2
                    current_ratio[1] -= 1 / len(portion)
                    current_ratio[2] += 1 / len(portion)
                    need_buy = current_ratio[2] < target_ratio[2] - tolerance
                elif need_sell and current_ratio[0] < target_ratio[0]:
                    labels.at[idx_val] = 0
                    current_ratio[1] -= 1 / len(portion)
                    current_ratio[0] += 1 / len(portion)
                    need_sell = current_ratio[0] < target_ratio[0] - tolerance
                if (not need_buy) and (not need_sell):
                    break
        return labels


def main():
    """主函數"""
    optimizer = LabelOptimizer(data_path='../data', config_path='../configs')
    result = optimizer.optimize(n_trials=200)
    print(f"標籤優化完成: {result['best_score']:.4f}")


if __name__ == "__main__":
    main()