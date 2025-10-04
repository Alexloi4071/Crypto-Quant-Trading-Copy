# -*- coding: utf-8 -*-
"""
Layer0: 加密貨幣數據清洗參數優化器 (基礎層)
專門針對 OHLCV 數據的基礎清洗和驗證
必須在所有其他優化層之前執行
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from optuna_system.utils.io_utils import write_dataframe, read_dataframe

warnings.filterwarnings('ignore')


class DataCleaningOptimizer:
    """Layer0: 加密貨幣數據清洗參數優化器 - 基礎層優化"""

    def __init__(self, data_path: str, config_path: str = "configs/",
                 symbol: str = "BTCUSDT", timeframe: str = "15m",
                 scaled_config: Dict = None):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.scaled_config = scaled_config or {}

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

    def apply_cleaning(self, original_data: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """組裝並執行完整清洗流程，回傳清洗後 DataFrame。"""
        if original_data is None or len(original_data) == 0:
            return pd.DataFrame()
        step1_data = self.check_price_anomalies(original_data, params)
        step2_data = self.check_ohlc_logic(step1_data, params)
        step3_data = self.check_timestamp_continuity(step2_data, params)
        step4_data = self.enhanced_price_cleaning(step3_data, params)
        step5_data = self.check_volume_anomalies(step4_data, params)
        cleaned_data = self.apply_missing_value_treatment(step5_data, params)
        cleaned_data = self.volume_void_detection(cleaned_data, params)
        return cleaned_data

    def apply_transform(self, original_data: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """物化介面別名，與其他層對齊。"""
        return self.apply_cleaning(original_data, **params)

    def load_raw_ohlcv_data(self) -> pd.DataFrame:
        """加載原始 OHLCV 數據"""
        try:
            cleaned_candidate = self.config_path / f"cleaned_ohlcv_{self.timeframe}.parquet"
            if cleaned_candidate.exists():
                data_df = read_dataframe(cleaned_candidate)
                self.logger.info(f"✅ 使用Layer0清洗數據: {cleaned_candidate}")
                self.logger.info(f"原始OHLCV數據: {data_df.shape}")
                return data_df

            ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
            self.logger.info(f"🔍 查找OHLCV文件: {ohlcv_file.absolute()}")

            if ohlcv_file.exists():
                data_df = read_dataframe(ohlcv_file)
                self.logger.info(f"✅ 加載原始OHLCV數據: {ohlcv_file}")
            else:
                # 尝试其他可能的路径
                alternative_paths = [
                    f"data/raw/{self.symbol}/{self.symbol}_{self.timeframe}_ohlcv.parquet",
                    f"../data/raw/{self.symbol}/{self.symbol}_{self.timeframe}_ohlcv.parquet",
                    f"./{self.symbol}_{self.timeframe}_ohlcv.parquet"
                ]
                
                data_df = None
                for alt_path in alternative_paths:
                    if Path(alt_path).exists():
                        self.logger.info(f"🔍 找到替代路径: {alt_path}")
                        data_df = read_dataframe(Path(alt_path))
                        break
                
                if data_df is None:
                    # 生成模擬 OHLCV 數據用於測試
                    self.logger.warning(f"❌ 未找到OHLCV數據文件: {ohlcv_file.absolute()}")
                    self.logger.warning("🔄 生成模擬數據用於測試")
                    data_df = self._generate_mock_ohlcv_data()
                else:
                    self.logger.info(f"✅ 成功加載OHLCV數據: {data_df.shape}")

            self.logger.info(f"原始OHLCV數據: {data_df.shape}")
            return data_df

        except Exception as e:
            self.logger.error(f"OHLCV數據加載失敗: {e}")
            return pd.DataFrame()

    def _generate_mock_ohlcv_data(self) -> pd.DataFrame:
        """生成包含異常值的模擬OHLCV數據"""
        np.random.seed(42)
        n_samples = 2000

        # 生成基礎價格序列
        base_price = 50000  # BTC基礎價格
        price_changes = np.random.normal(0, 0.02, n_samples)  # 2%標準差
        prices = [base_price]

        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # 確保價格不會太低

        prices = np.array(prices[1:])

        # 生成 OHLCV 數據
        data = []
        for i, close in enumerate(prices):
            # 正常情況下的 OHLC 關係
            volatility = np.random.uniform(0.005, 0.03)  # 0.5%-3%波動

            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)

            # 確保 OHLC 邏輯正確
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # 正常成交量
            volume = np.random.uniform(100, 10000)

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        # 添加一些異常數據來測試清洗器
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)

        for idx in anomaly_indices:
            if idx < len(data):
                anomaly_type = np.random.choice(['zero_price', 'extreme_price', 'ohlc_violation', 'zero_volume'])

                if anomaly_type == 'zero_price':
                    # 價格為0的異常
                    data[idx]['close'] = 0
                elif anomaly_type == 'extreme_price':
                    # 極端價格異常
                    data[idx]['close'] *= np.random.choice([100, 0.01])
                elif anomaly_type == 'ohlc_violation':
                    # OHLC關係錯誤
                    data[idx]['high'] = data[idx]['low'] * 0.5  # high < low
                elif anomaly_type == 'zero_volume':
                    # 成交量為0
                    data[idx]['volume'] = 0

        # 創建 DataFrame
        dates = pd.date_range('2022-01-01', periods=n_samples, freq='15min')
        df = pd.DataFrame(data, index=dates)

        # 添加一些缺失值
        missing_mask = np.random.random(len(df)) < 0.005  # 0.5%缺失
        df.loc[missing_mask, 'volume'] = np.nan

        return df

    def check_price_anomalies(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """檢查和處理價格異常"""
        try:
            cleaned_data = data.copy()

            # 1. 移除價格為0或負數的記錄
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in cleaned_data.columns:
                    # 標記異常值
                    zero_mask = (cleaned_data[col] <= 0)
                    if zero_mask.sum() > 0:
                        self.logger.warning(f"發現 {zero_mask.sum()} 個 {col} 價格 <= 0 的異常值")

                        # 處理方式根據參數決定
                        if params.get('zero_price_action', 'drop') == 'drop':
                            cleaned_data = cleaned_data[~zero_mask]
                        elif params.get('zero_price_action', 'drop') == 'forward_fill':
                            cleaned_data.loc[zero_mask, col] = np.nan
                            cleaned_data[col] = cleaned_data[col].ffill()

            # 2. 移除極端價格異常（基於移動中位數）
            price_change_threshold = params.get('extreme_price_threshold', 10.0)

            for col in price_cols:
                if col in cleaned_data.columns and len(cleaned_data) > 10:
                    # 計算價格變化率
                    price_change = cleaned_data[col].pct_change().abs()

                    # 基於滾動中位數的異常檢測
                    rolling_median = price_change.rolling(window=20, min_periods=5).median()
                    extreme_mask = price_change > (rolling_median * price_change_threshold)

                    if extreme_mask.sum() > 0:
                        self.logger.warning(f"發現 {extreme_mask.sum()} 個 {col} 極端價格變化")

                        if params.get('extreme_price_action', 'cap') == 'cap':
                            # 限制在合理範圍內
                            max_change = rolling_median * price_change_threshold
                            prev_prices = cleaned_data[col].shift(1)

                            # 向上調整
                            up_mask = extreme_mask & (cleaned_data[col] > prev_prices)
                            cleaned_data.loc[up_mask, col] = prev_prices[up_mask] * (1 + max_change[up_mask])

                            # 向下調整
                            down_mask = extreme_mask & (cleaned_data[col] < prev_prices)
                            cleaned_data.loc[down_mask, col] = prev_prices[down_mask] * (1 - max_change[down_mask])

            return cleaned_data

        except Exception as e:
            self.logger.error(f"價格異常檢查失敗: {e}")
            return data

    def check_ohlc_logic(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """檢查和修復OHLC邏輯關係"""
        try:
            cleaned_data = data.copy()

            # OHLC邏輯檢查: high >= max(open, close), low <= min(open, close)
            if all(col in cleaned_data.columns for col in ['open', 'high', 'low', 'close']):

                # 檢查 high 是否真的是最高價
                max_oc = np.maximum(cleaned_data['open'], cleaned_data['close'])
                high_violation = cleaned_data['high'] < max_oc

                # 檢查 low 是否真的是最低價
                min_oc = np.minimum(cleaned_data['open'], cleaned_data['close'])
                low_violation = cleaned_data['low'] > min_oc

                violations = high_violation | low_violation

                if violations.sum() > 0:
                    self.logger.warning(f"發現 {violations.sum()} 個OHLC邏輯錯誤")

                    if params.get('ohlc_fix_method', 'adjust') == 'adjust':
                        # 調整 high 和 low 以符合邏輯
                        cleaned_data.loc[high_violation, 'high'] = max_oc[high_violation]
                        cleaned_data.loc[low_violation, 'low'] = min_oc[low_violation]

                    elif params.get('ohlc_fix_method', 'adjust') == 'drop':
                        # 移除邏輯錯誤的記錄
                        cleaned_data = cleaned_data[~violations]

            return cleaned_data

        except Exception as e:
            self.logger.error(f"OHLC邏輯檢查失敗: {e}")
            return data

    def check_timestamp_continuity(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """檢查時間戳連續性"""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                self.logger.warning("索引不是DatetimeIndex，跳過時間連續性檢查")
                return data

            # 計算時間間隔
            time_diffs = data.index.to_series().diff()
            expected_freq = pd.Timedelta(self.timeframe)

            # 找出時間間隔異常的位置
            gap_threshold = expected_freq * params.get('timestamp_gap_multiplier', 5.0)
            large_gaps = time_diffs > gap_threshold

            if large_gaps.sum() > 0:
                self.logger.warning(f"發現 {large_gaps.sum()} 個時間戳間隔異常")

            return data

        except Exception as e:
            self.logger.error(f"時間戳連續性檢查失敗: {e}")
            return data

    def check_volume_anomalies(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """檢查成交量異常"""
        try:
            cleaned_data = data.copy()

            if 'volume' not in cleaned_data.columns:
                return cleaned_data

            # 1. 處理成交量為0或負數
            zero_volume_mask = cleaned_data['volume'] <= 0
            if zero_volume_mask.sum() > 0:
                self.logger.warning(f"發現 {zero_volume_mask.sum()} 個成交量 <= 0 的記錄")

                if params.get('zero_volume_action', 'fill_median') == 'fill_median':
                    # 用中位數填充
                    median_volume = cleaned_data['volume'][cleaned_data['volume'] > 0].median()
                    cleaned_data.loc[zero_volume_mask, 'volume'] = median_volume
                elif params.get('zero_volume_action', 'fill_median') == 'drop':
                    cleaned_data = cleaned_data[~zero_volume_mask]

            # 2. 檢查極端成交量
            if len(cleaned_data) > 0:
                volume_median = cleaned_data['volume'].median()
                volume_std = cleaned_data['volume'].std()

                # 基於中位數和標準差的異常檢測
                extreme_threshold = params.get('volume_extreme_multiplier', 5.0)
                extreme_volume_mask = cleaned_data['volume'] > (volume_median + extreme_threshold * volume_std)

                if extreme_volume_mask.sum() > 0:
                    self.logger.warning(f"發現 {extreme_volume_mask.sum()} 個極端成交量")

                    if params.get('extreme_volume_action', 'cap') == 'cap':
                        # 限制在合理範圍內
                        max_volume = volume_median + extreme_threshold * volume_std
                        cleaned_data.loc[extreme_volume_mask, 'volume'] = max_volume

            return cleaned_data

        except Exception as e:
            self.logger.error(f"成交量異常檢查失敗: {e}")
            return data

    def enhanced_price_cleaning(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """強化版價格清洗：Winsorize + 跳變檢測"""
        try:
            cleaned = data.copy()
            price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in cleaned.columns]
            if not price_cols:
                return cleaned

            window = max(20, int(params.get('winsorize_window', 200)))
            lower_q = float(params.get('winsorize_lower', 0.01))
            upper_q = float(params.get('winsorize_upper', 0.99))
            min_periods = max(5, window // 5)

            for col in price_cols:
                rolling_lower = cleaned[col].rolling(window, min_periods=min_periods).quantile(lower_q)
                rolling_upper = cleaned[col].rolling(window, min_periods=min_periods).quantile(upper_q)

                global_lower = cleaned[col].quantile(lower_q)
                global_upper = cleaned[col].quantile(upper_q)

                cleaned[col] = cleaned[col].clip(
                    lower=rolling_lower.fillna(global_lower),
                    upper=rolling_upper.fillna(global_upper)
                )

            if 'close' not in cleaned.columns:
                return cleaned

            returns = cleaned['close'].pct_change()
            jump_threshold = float(params.get('jump_threshold', 0.05))
            jump_mask = returns.abs() > jump_threshold

            if jump_mask.any():
                jump_indices = cleaned.index[jump_mask.fillna(False)]
                for idx in jump_indices:
                    if idx == cleaned.index[0] or idx == cleaned.index[-1]:
                        continue
                    prev_idx = cleaned.index.get_loc(idx) - 1
                    next_idx = cleaned.index.get_loc(idx) + 1
                    if prev_idx < 0 or next_idx >= len(cleaned):
                        continue
                    prev_price = cleaned.iloc[prev_idx]['close']
                    next_price = cleaned.iloc[next_idx]['close']
                    if prev_price <= 0 or next_price <= 0:
                        continue
                    cleaned.at[idx, 'close'] = np.sqrt(prev_price * next_price)

            return cleaned

        except Exception as e:
            self.logger.warning(f"強化價格清洗失敗: {e}")
            return data

    def volume_void_detection(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """成交量空洞檢測與填補"""
        try:
            if 'volume' not in data.columns or 'close' not in data.columns:
                return data

            cleaned = data.copy()
            vol_window = max(10, int(params.get('vol_window', 50)))
            void_threshold = float(params.get('void_threshold', 0.1))
            vol_multiplier = float(params.get('vol_vol_multiplier', 5.0))

            vol_ma = cleaned['volume'].rolling(vol_window, min_periods=1).mean()
            void_mask = cleaned['volume'] < (vol_ma * void_threshold)

            if void_mask.any():
                price_volatility = cleaned['close'].pct_change().abs().rolling(vol_window, min_periods=1).mean()
                estimated_volume = (vol_ma * (1 + price_volatility.fillna(0) * vol_multiplier)).fillna(vol_ma)
                cleaned.loc[void_mask, 'volume'] = estimated_volume[void_mask]

            return cleaned

        except Exception as e:
            self.logger.warning(f"成交量空洞檢測失敗: {e}")
            return data

    def evaluate_ohlcv_cleaning_effectiveness(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict:
        """評估OHLCV數據清洗效果"""
        try:
            metrics = {}

            # 數據保留率
            metrics['data_retention_rate'] = len(cleaned) / max(len(original), 1)

            # 異常值移除效果
            if all(col in original.columns for col in ['open', 'high', 'low', 'close']):
                # 價格異常檢查
                original_zero_prices = (original[['open', 'high', 'low', 'close']] <= 0).sum().sum()
                cleaned_zero_prices = (cleaned[['open', 'high', 'low', 'close']] <= 0).sum().sum()
                metrics['zero_prices_removed'] = original_zero_prices - cleaned_zero_prices

                # OHLC邏輯一致性
                if len(cleaned) > 0:
                    max_oc = np.maximum(cleaned['open'], cleaned['close'])
                    min_oc = np.minimum(cleaned['open'], cleaned['close'])

                    ohlc_consistent = ((cleaned['high'] >= max_oc) & (cleaned['low'] <= min_oc)).mean()
                    metrics['ohlc_consistency_rate'] = ohlc_consistent
                else:
                    metrics['ohlc_consistency_rate'] = 0

            # 成交量質量
            if 'volume' in original.columns and 'volume' in cleaned.columns:
                original_zero_volume = (original['volume'] <= 0).sum()
                cleaned_zero_volume = (cleaned['volume'] <= 0).sum()
                metrics['zero_volume_removed'] = original_zero_volume - cleaned_zero_volume

            # 數據完整性
            original_missing = original.isnull().sum().sum()
            cleaned_missing = cleaned.isnull().sum().sum()
            metrics['missing_data_handled'] = original_missing - cleaned_missing

            return metrics

        except Exception as e:
            self.logger.error(f"OHLCV清洗效果評估失敗: {e}")
            return {'error': str(e)}

    def apply_missing_value_treatment(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """🚀 修復版：時間序列導向的缺失值處理"""
        try:
            method = params['impute_method']

            # 統計原始缺失比例
            total_values = data.size
            missing_before = data.isnull().sum().sum()
            missing_ratio_before = missing_before / total_values if total_values > 0 else 0

            self.logger.info(f"📊 原始缺失值: {missing_before:,}/{total_values:,} ({missing_ratio_before:.2%})")

            if method == 'drop':
                # 刪除缺失值（保留原邏輯）
                threshold = params.get('missing_threshold', 0.1)
                missing_ratio = data.isnull().sum() / len(data)
                cols_to_keep = missing_ratio[missing_ratio <= threshold].index
                cleaned_data = data[cols_to_keep].dropna()

            elif method in ['forward_fill', 'ffill']:
                # 🚀 修復版：時間序列優先填補策略
                # 第一步：前向填充（保持時間序列連續性）
                cleaned_data = data.ffill()

                # 第二步：滾動中位數填補剩餘缺失
                window = max(params.get('impute_window', 20), 10)  # 確保窗口足夠大

                for col in cleaned_data.columns:
                    remaining_missing = cleaned_data[col].isnull()
                    if remaining_missing.any():
                        # 滾動中位數填補
                        rolling_median = data[col].rolling(window=window, min_periods=3).median()
                        cleaned_data.loc[remaining_missing, col] = rolling_median[remaining_missing]

                        # 最後備選：全局中位數
                        still_missing = cleaned_data[col].isnull()
                        if still_missing.any():
                            global_median = data[col].median()
                            if not pd.isna(global_median):
                                cleaned_data.loc[still_missing, col] = global_median

                self.logger.info(f"✅ 時間序列填補: ffill + 滾動中位數 (窗口={window})")

            elif method == 'median':
                # 🚀 修復版：時間序列友好的中位數填充
                cleaned_data = data.ffill()  # 先前向填充

                # 再用滾動中位數填充剩餘
                window = params.get('impute_window', 20)
                for col in cleaned_data.columns:
                    remaining_missing = cleaned_data[col].isnull()
                    if remaining_missing.any():
                        rolling_median = data[col].rolling(window=window, min_periods=3).median()
                        cleaned_data.loc[remaining_missing, col] = rolling_median[remaining_missing]

                # 最終全局中位數填充
                for col in cleaned_data.columns:
                    global_median = data[col].median()
                    if not pd.isna(global_median):
                        cleaned_data[col] = cleaned_data[col].fillna(global_median)

                self.logger.info(f"✅ 時間序列中位數填補 (滾動窗口={window})")

            elif method == 'rolling_mean':
                # 🚀 修復版：先ffill再滾動均值
                cleaned_data = data.ffill()
                window = params.get('impute_window', 20)

                for col in cleaned_data.columns:
                    remaining_missing = cleaned_data[col].isnull()
                    if remaining_missing.any():
                        rolling_mean = data[col].rolling(window=window, min_periods=3).mean()
                        cleaned_data.loc[remaining_missing, col] = rolling_mean[remaining_missing]

                self.logger.info(f"✅ ffill + 滾動均值填補 (窗口={window})")

            elif method == 'knn':
                # KNN填充（保留，但提醒時間序列風險）
                n_neighbors = params.get('knn_neighbors', 5)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                filled_values = imputer.fit_transform(data)
                cleaned_data = pd.DataFrame(filled_values, columns=data.columns, index=data.index)
                self.logger.warning("⚠️ KNN填補可能破壞時間序列特性")

            else:
                # 🚀 默認：時間序列最優策略
                cleaned_data = data.ffill()  # 前向填充
                remaining_missing = cleaned_data.isnull().sum().sum()
                if remaining_missing > 0:
                    # 滾動中位數填充剩餘
                    for col in cleaned_data.columns:
                        rolling_median = data[col].rolling(window=20, min_periods=3).median()
                        cleaned_data[col] = cleaned_data[col].fillna(rolling_median)

                self.logger.info("✅ 默認時間序列填補: ffill + 滾動中位數")

            # 統計填補效果
            missing_after = cleaned_data.isnull().sum().sum()
            missing_ratio_after = missing_after / cleaned_data.size if cleaned_data.size > 0 else 0
            fill_success_rate = (missing_before - missing_after) / missing_before if missing_before > 0 else 1.0

            self.logger.info(f"📈 填補效果: {missing_after:,}剩餘 ({missing_ratio_after:.2%}), 成功率={fill_success_rate:.1%}")

            # 最終保障：確保無缺失值（避免對價格欄位用0填補）
            if missing_after > 0:
                price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in cleaned_data.columns]
                non_price_cols = [c for c in cleaned_data.columns if c not in price_cols]

                # 價格欄位：再次前向填補；如仍有缺失，用首個有效值填補開頭缺失
                if price_cols:
                    cleaned_data[price_cols] = cleaned_data[price_cols].ffill()
                    for col in price_cols:
                        if cleaned_data[col].isnull().any():
                            series = cleaned_data[col]
                            first_valid = series.dropna().iloc[0] if series.dropna().size > 0 else None
                            if first_valid is not None:
                                cleaned_data[col] = series.fillna(first_valid)
                            else:
                                # 無有效值時，保留為NaN並記錄
                                self.logger.warning(f"⚠️ 無可用價格填補值: {col} 仍有缺失")

                # 非價格欄位：允許0填補作為最後兜底
                if non_price_cols:
                    residual_missing = cleaned_data[non_price_cols].isnull().sum().sum()
                    if residual_missing > 0:
                        cleaned_data[non_price_cols] = cleaned_data[non_price_cols].fillna(0)
                        self.logger.warning(f"⚠️ 對非價格欄位用0填補了{int(residual_missing)}個頑固缺失值")

            return cleaned_data

        except Exception as e:
            self.logger.error(f"缺失值處理失敗: {e}")
            # 安全備選方案：價格欄位僅ffill，非價格欄位允許0兜底
            fallback = data.ffill()
            price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in fallback.columns]
            non_price_cols = [c for c in fallback.columns if c not in price_cols]
            if price_cols:
                for col in price_cols:
                    if fallback[col].isnull().any():
                        series = fallback[col]
                        first_valid = series.dropna().iloc[0] if series.dropna().size > 0 else None
                        if first_valid is not None:
                            fallback[col] = series.fillna(first_valid)
            if non_price_cols:
                fallback[non_price_cols] = fallback[non_price_cols].fillna(0)
            return fallback

    def apply_outlier_treatment(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """應用異常值處理"""
        try:
            method = params['outlier_method']

            if method == 'iqr':
                # 🚀 修復版：滾動窗口IQR方法（避免極端段落處理不當）
                multiplier = params.get('iqr_multiplier', 1.5)
                window = params.get('rolling_outlier_window', 1000)  # 滾動窗口大小

                cleaned_data = data.copy()

                for col in data.columns:
                    # 基於滾動窗口計算動態四分位數
                    rolling_q1 = data[col].rolling(window=window, min_periods=50).quantile(0.25)
                    rolling_q3 = data[col].rolling(window=window, min_periods=50).quantile(0.75)
                    rolling_iqr = rolling_q3 - rolling_q1

                    # 動態邊界
                    lower_bound = rolling_q1 - multiplier * rolling_iqr
                    upper_bound = rolling_q3 + multiplier * rolling_iqr

                    # 標記並修復異常值
                    outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outlier_count = outlier_mask.sum()

                    if outlier_count > 0:
                        # 裁剪到動態邊界
                        cleaned_data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                        self.logger.info(f"  {col}: 修復{outlier_count}個滾動IQR異常值")

                self.logger.info(f"✅ 滾動窗口IQR異常值檢測完成 (窗口={window})")

            elif method == 'zscore':
                # 🚀 修復版：滾動窗口Z-score方法（避免極端段落處理不當）
                threshold = params.get('zscore_threshold', 3.0)
                window = params.get('rolling_outlier_window', 1000)  # 滾動窗口大小

                cleaned_data = data.copy()

                for col in data.columns:
                    # 基於滾動窗口計算動態統計
                    rolling_median = data[col].rolling(window=window, min_periods=50).median()
                    rolling_std = data[col].rolling(window=window, min_periods=50).std()

                    # 計算滾動Z-score
                    z_scores = np.abs((data[col] - rolling_median) / (rolling_std + 1e-8))

                    # 標記異常值
                    outlier_mask = z_scores > threshold
                    outlier_count = outlier_mask.sum()

                    if outlier_count > 0:
                        # 用滾動中位數替換異常值
                        cleaned_data.loc[outlier_mask, col] = rolling_median[outlier_mask]
                        self.logger.info(f"  {col}: 修復{outlier_count}個滾動Z-score異常值")

                self.logger.info(f"✅ 滾動窗口異常值檢測完成 (窗口={window})")

            elif method == 'percentile':
                # 百分位數裁剪
                lower_pct = params.get('lower_percentile', 5)
                upper_pct = params.get('upper_percentile', 95)

                lower_bound = data.quantile(lower_pct / 100)
                upper_bound = data.quantile(upper_pct / 100)

                cleaned_data = data.clip(lower=lower_bound, upper=upper_bound, axis=1)

            elif method == 'none':
                # 不處理異常值
                cleaned_data = data

            else:
                # 默認IQR方法
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_data = data.clip(lower=lower_bound, upper=upper_bound, axis=1)

            return cleaned_data

        except Exception as e:
            self.logger.error(f"異常值處理失敗: {e}")
            return data

    def apply_smoothing(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """應用數據平滑"""
        try:
            method = params.get('smooth_method', 'none')
            smooth_min = max(1, int(self.scaled_config.get('cleaning_smooth_window_min', 5)))

            if method == 'rolling_mean':
                smooth_window = params.get('smooth_window', smooth_min)
                smoothed_data = data.rolling(window=smooth_window, min_periods=1).mean()

            elif method == 'ewm':
                alpha = params.get('exp_alpha', 0.3)
                smoothed_data = data.ewm(alpha=alpha).mean()

            elif method == 'savgol':
                # 簡化的Savitzky-Golay濾波（使用滾動多項式近似）
                smooth_window = params.get('smooth_window', smooth_min)
                smoothed_data = data.rolling(window=smooth_window, min_periods=1).apply(
                    lambda x: np.polyval(np.polyfit(range(len(x)), x, 2), len(x)//2), raw=True
                )

            elif method == 'none':
                smoothed_data = data

            else:
                # 默認不平滑
                smoothed_data = data

            return smoothed_data

        except Exception as e:
            self.logger.error(f"數據平滑失敗: {e}")
            return data

    def evaluate_cleaning_quality(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict:
        """評估數據清洗質量"""
        try:
            # 數據完整性
            original_missing = original.isnull().sum().sum()
            cleaned_missing = cleaned.isnull().sum().sum()
            completeness_score = 1 - (cleaned_missing / max(original.size, 1))

            # 數據保真度（通過重構誤差衡量）
            common_cols = original.columns.intersection(cleaned.columns)
            if len(common_cols) > 0:
                original_subset = original[common_cols].fillna(0)
                cleaned_subset = cleaned[common_cols].fillna(0)

                # 計算標準化後的MSE
                scaler = StandardScaler()
                original_scaled = scaler.fit_transform(original_subset)
                cleaned_scaled = scaler.transform(cleaned_subset)

                mse = mean_squared_error(original_scaled, cleaned_scaled)
                fidelity_score = np.exp(-mse)  # 轉換為0-1分數
            else:
                fidelity_score = 0

            # 數據穩定性（方差變化）
            original_var = original.var().mean()
            cleaned_var = cleaned.var().mean()

            if original_var > 0:
                variance_ratio = cleaned_var / original_var
                stability_score = 1 - abs(1 - variance_ratio)  # 接近1的比例得高分
            else:
                stability_score = 0

            return {
                'completeness_score': completeness_score,
                'fidelity_score': fidelity_score,
                'stability_score': stability_score,
                'data_reduction': (original.size - cleaned.size) / original.size,
                'missing_reduction': (original_missing - cleaned_missing) / max(original_missing, 1)
            }

        except Exception as e:
            self.logger.error(f"清洗質量評估失敗: {e}")
            return {
                'completeness_score': 0,
                'fidelity_score': 0,
                'stability_score': 0,
                'data_reduction': 0,
                'missing_reduction': 0
            }

    def objective(self, trial: optuna.Trial) -> float:
        """修复版：支持与后续层联动的清洗参数优化"""
        
        # 🚀 新增：读取Layer1参数实现平滑窗口联动
        layer1_lookback = 500  # 默认值
        try:
            label_config = self.config_path / "label_params.json"
            if label_config.exists():
                with open(label_config, 'r', encoding='utf-8') as f:
                    layer1_result = json.load(f)
                    layer1_lookback = layer1_result.get('best_params', {}).get('lookback_window', 500)
                    self.logger.info(f"🔗 Layer0读取Layer1 lookback_window: {layer1_lookback}")
        except Exception as e:
            self.logger.warning(f"无法读取Layer1参数: {e}")

        window_min = self.scaled_config.get('cleaning_impute_window', 10)
        smooth_min = self.scaled_config.get('cleaning_smooth_window_min', 5)
        smooth_max = self.scaled_config.get('cleaning_smooth_window_max', 20)
        params = {
            # 缺失值處理
            'impute_method': trial.suggest_categorical('impute_method',
                                                     ['forward_fill', 'mean', 'median', 'knn', 'rolling_mean']),
            'impute_window': trial.suggest_int('impute_window', max(3, window_min // 2), max(30, window_min * 2)),
            'missing_threshold': trial.suggest_float('missing_threshold', 0.05, 0.3),
            'knn_neighbors': trial.suggest_int('knn_neighbors', 3, 10),

            # ✅ 優化：價格清洗（針對加密貨幣大幅波動）
            'winsorize_window': trial.suggest_int('winsorize_window', 80, 300),
            'winsorize_lower': trial.suggest_float('winsorize_lower', 0.001, 0.02),
            'winsorize_upper': trial.suggest_float('winsorize_upper', 0.98, 0.999),
            'jump_threshold': trial.suggest_float('jump_threshold', 0.05, 0.12),  # 放寬跳躍閾值

            # ✅ 優化：滾動窗口異常值處理（針對加密貨幣高波動特性）
            'outlier_method': trial.suggest_categorical('outlier_method',
                                                      ['iqr', 'zscore', 'percentile', 'none']),
            'iqr_multiplier': trial.suggest_float('iqr_multiplier', 2.5, 4.0),  # 更保守，避免過度清洗
            'zscore_threshold': trial.suggest_float('zscore_threshold', 2.5, 5.0),  # 放寬閾值
            
            # 🔧 修复2：基于数据保留率动态调整缺失值阈值
            'rolling_outlier_window': trial.suggest_int('rolling_outlier_window', 
                                                       max(100, layer1_lookback // 5),  # 联动
                                                       max(1000, layer1_lookback // 2)), # 联动
            'lower_percentile': trial.suggest_float('lower_percentile', 1, 10),
            'upper_percentile': trial.suggest_float('upper_percentile', 90, 99),

            # 成交量空洞檢測
            'vol_window': trial.suggest_int('vol_window', 30, 120),
            'void_threshold': trial.suggest_float('void_threshold', 0.05, 0.3),
            'vol_vol_multiplier': trial.suggest_float('vol_vol_multiplier', 2.0, 8.0),

            # 🔧 修复1：平滑窗口与Layer1 lookback_window联动
            'smooth_method': trial.suggest_categorical('smooth_method', ['rolling_mean', 'savgol', 'exp']),
            'smooth_window': trial.suggest_int('smooth_window', 
                                             max(5, smooth_min),
                                             max(20, smooth_max)),
            'exp_alpha': trial.suggest_float('exp_alpha', 0.01, 0.2),  # 限制α範圍防止過度平滑

            # 質量權重
            'completeness_weight': trial.suggest_float('completeness_weight', 0.3, 0.7),
            'fidelity_weight': trial.suggest_float('fidelity_weight', 0.2, 0.6),
            'stability_weight': trial.suggest_float('stability_weight', 0.1, 0.4)
        }

        try:
            # 加載原始OHLCV數據
            original_data = self.load_raw_ohlcv_data()

            if len(original_data) == 0:
                return -999.0

            # 執行OHLCV專用清洗步驟
            step1_data = self.check_price_anomalies(original_data, params)
            step2_data = self.check_ohlc_logic(step1_data, params)
            step3_data = self.check_timestamp_continuity(step2_data, params)
            step4_data = self.enhanced_price_cleaning(step3_data, params)
            step5_data = self.check_volume_anomalies(step4_data, params)
            cleaned_data = self.volume_void_detection(step5_data, params)

            if len(cleaned_data) == 0:
                return -999.0
            
            # 🔧 新增：动态调整缺失值阈值
            data_retention_rate = len(cleaned_data) / len(original_data) if len(original_data) > 0 else 1.0
            
            # 如果数据丢失过多，动态降低缺失值阈值
            if data_retention_rate < 0.8:
                adjusted_threshold = params.get('missing_threshold', 0.1) * 0.8
                self.logger.info(f"🔧 数据保留率低({data_retention_rate:.1%})，降低缺失值阈值至{adjusted_threshold:.3f}")
                params['missing_threshold'] = adjusted_threshold
                # 重新清洗（简化版，仅处理缺失值）
                if 'missing_threshold' in params:
                    missing_ratio = cleaned_data.isnull().sum().sum() / cleaned_data.size if cleaned_data.size > 0 else 0
                    if missing_ratio > adjusted_threshold:
                        cleaned_data = cleaned_data.dropna()
                    data_retention_rate = len(cleaned_data) / len(original_data)
            
            # 将清洗统计信息写入trial.user_attrs供后续层使用
            trial.set_user_attr("data_retention_rate", data_retention_rate)
            trial.set_user_attr("smooth_window_used", params['smooth_window'])
            trial.set_user_attr("outlier_window_used", params['rolling_outlier_window'])

            # 評估OHLCV清洗效果
            effectiveness = self.evaluate_ohlcv_cleaning_effectiveness(original_data, cleaned_data)

            # 計算綜合得分
            retention_score = effectiveness.get('data_retention_rate', 0)
            consistency_score = effectiveness.get('ohlc_consistency_rate', 0)

            # 質量得分：基於異常值移除效果
            quality_metrics = [
                effectiveness.get('zero_prices_removed', 0),
                effectiveness.get('zero_volume_removed', 0),
                effectiveness.get('missing_data_handled', 0)
            ]
            quality_score = min(1.0, sum(quality_metrics) / max(len(original_data) * 0.1, 1))

            # 加權綜合得分
            final_score = (retention_score * params['completeness_weight'] +
                          consistency_score * params['fidelity_weight'] +
                          quality_score * params['stability_weight'])

            # ✅ 修復：計算清洗後統計信息（僅使用歷史數據，無泄漏）
            try:
                # 計算清洗後的"當前"收益率統計（非未來收益率）
                if 'close' in cleaned_data.columns and len(cleaned_data) > 100:
                    # ✅ 使用歷史收益率（非未來收益率）
                    current_returns = cleaned_data['close'].pct_change()
                    current_returns_clean = current_returns.dropna()
                    
                    if len(current_returns_clean) > 100:
                        # 清洗後的歷史波動率統計
                        cleaned_volatility = float(current_returns_clean.std())
                        cleaned_mean_return = float(current_returns_clean.mean())
                        cleaned_q90 = float(current_returns_clean.quantile(0.9))
                        cleaned_q10 = float(current_returns_clean.quantile(0.1))
                        
                        # 保存到trial.user_attrs（無泄漏）
                        trial.set_user_attr("cleaned_volatility", cleaned_volatility)
                        trial.set_user_attr("cleaned_mean_return", cleaned_mean_return)
                        trial.set_user_attr("cleaned_q90", cleaned_q90)
                        trial.set_user_attr("cleaned_q10", cleaned_q10)
                        trial.set_user_attr("cleaned_data_length", len(cleaned_data))
                        
                        self.logger.info(f"📊 清洗統計（歷史波動率）: 波動率={cleaned_volatility:.4f}, "
                                       f"均值={cleaned_mean_return:.6f}, "
                                       f"90%分位={cleaned_q90:.4f}, 10%分位={cleaned_q10:.4f}")
            except Exception as e:
                self.logger.warning(f"清洗統計計算失敗: {e}")

            return final_score

        except Exception as e:
            self.logger.error(f"數據清洗優化過程出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 25) -> Dict:
        """執行Layer0加密貨幣數據清洗參數優化"""
        self.logger.info("🚀 開始Layer0數據清洗參數優化...")

        # 創建研究
        storage_url = None
        try:
            storage_url = self.scaled_config.get('optuna_storage')
        except Exception:
            storage_url = None
        study = optuna.create_study(
            direction='maximize',
            study_name='layer0_crypto_data_cleaning',
            storage=storage_url,
            load_if_exists=bool(storage_url)
        )

        # 執行優化
        study.optimize(self.objective, n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"✅ Layer0數據清洗優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"最優參數: {best_params}")

        # 使用最優參數重新評估
        try:
            original_data = self.load_raw_ohlcv_data()
            final_cleaned_data = self.apply_cleaning(original_data, **best_params)

            final_effectiveness = self.evaluate_ohlcv_cleaning_effectiveness(original_data, final_cleaned_data)

            detailed_results = {
                'cleaning_effectiveness': final_effectiveness,
                'original_data_shape': original_data.shape,
                'cleaned_data_shape': final_cleaned_data.shape,
                'data_retention_rate': len(final_cleaned_data) / len(original_data)
            }
        except Exception as e:
            self.logger.warning(f"無法獲取詳細結果: {e}")
            detailed_results = {}
            final_effectiveness = {}
            final_cleaned_data = None

        cleaned_file_path = None
        if final_cleaned_data is not None:
            try:
                # 1) 保持舊位置（向後兼容）
                target_path = self.config_path / f"cleaned_ohlcv_{self.timeframe}.parquet"
                cleaned_file_path, _ = write_dataframe(final_cleaned_data, target_path)
                self.logger.info(f"✅ Layer0清洗數據已保存: {cleaned_file_path}")

                # 2) 新位置（processed）
                # 若存在最新版本號，將清洗數據也寫入版本子目錄
                results_root = Path(__file__).resolve().parent.parent / "results"
                latest_file = results_root / "latest.txt"
                version_sub = None
                if latest_file.exists():
                    try:
                        version_sub = latest_file.read_text(encoding="utf-8").strip()
                    except Exception:
                        version_sub = None
                processed_dir = self.data_path / "processed" / "cleaned" / f"{self.symbol}_{self.timeframe}"
                if version_sub:
                    processed_dir = processed_dir / version_sub
                processed_dir.mkdir(parents=True, exist_ok=True)
                processed_target = processed_dir / "cleaned_ohlcv.parquet"
                processed_file, _ = write_dataframe(final_cleaned_data, processed_target)
                self.logger.info(f"✅ Layer0清洗數據已同步至: {processed_file}")
            except Exception as e:
                self.logger.warning(f"⚠️ 清洗數據保存失敗: {e}")
                cleaned_file_path = None

        # 保存結果 (修復JSON序列化問題)
        def convert_numpy_types(obj):
            """轉換numpy類型為Python原生類型"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        result = {
            'layer': 0,
            'description': 'Layer0: 加密貨幣數據清洗參數優化',
            'best_params': convert_numpy_types(best_params),
            'best_score': convert_numpy_types(best_score),
            'n_trials': convert_numpy_types(n_trials),
            'detailed_results': convert_numpy_types(detailed_results),
            'metrics': convert_numpy_types(final_effectiveness),
            'cleaned_file': str(cleaned_file_path) if cleaned_file_path else None,
            'optimization_history': [
                {'trial': i, 'score': convert_numpy_types(trial.value)}
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ]
        }

        # 保存到JSON文件（保持統一命名）
        output_file = self.config_path / "cleaning_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"✅ Layer0結果已保存至: {output_file}")

        return result


def main():
    """主函數"""
    optimizer = DataCleaningOptimizer(
        data_path='../data',
        config_path='../configs',
        symbol='BTCUSDT',
        timeframe='15m'
    )
    result = optimizer.optimize(n_trials=25)
    print(f"Layer0數據清洗優化完成: {result['best_score']:.4f}")


if __name__ == "__main__":
    main()