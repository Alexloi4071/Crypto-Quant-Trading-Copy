#!/usr/bin/env python3
"""
標籤生成器 - 簡化版本，專注於標籤生成邏輯
原 Optuna 優化功能已遷移至 optuna_system/optimizers/optuna_label.py

負責：
- 基於價格數據生成交易標籤
- 計算標籤穩定性和平衡性
- 數據平衡處理 (SMOTE等)
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from typing import Dict, List, Tuple, Any
import warnings

# Import balancing libraries (合併自LabelBalancer功能)
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

warnings.filterwarnings('ignore')

from .config import get_config


class LabelGenerator:
    """標籤生成器 - 專注於標籤生成和平衡處理"""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = get_config("label", timeframe)

    def generate_labels(
        self,
        price_data: pd.Series,
        lag: int,
        profit_threshold: float,
        loss_threshold: float,
        label_type: str,
        threshold_method: str = "fixed"
    ) -> pd.Series:
        """
        基於未來實際盈利能力生成交易標籤
        
        Args:
            price_data: 價格序列
            lag: 預測未來N期的收益
            profit_threshold: 最小盈利閾值 (正值) 或分位數 (0-1)
            loss_threshold: 最大虧損閾值 (負值) 或分位數 (0-1)
            label_type: 標籤類型 ("binary" 或 "ternary")
            threshold_method: 閾值計算方法 ("fixed", "quantile", "adaptive")
        """
        try:
            print(f"🎯 生成標籤: 基於未來{lag}期實際盈利能力")
            
            # 🎯 核心邏輯: 計算未來實際收益 (考慮交易成本)
            future_prices = price_data.shift(-lag)  # 未來lag期價格
            future_returns = (future_prices / price_data) - 1  # 未來原始收益率
            
            # 考慮實際交易成本
            total_trading_cost = 0.0006  # 0.04% (taker) + 0.02% (slippage)
            actual_profit = future_returns - total_trading_cost
            
            # 動態閾值計算
            if threshold_method == "adaptive":
                dynamic_profit_threshold, dynamic_loss_threshold = self._calculate_adaptive_thresholds(
                    price_data, profit_threshold, loss_threshold
                )
            elif threshold_method == "quantile":
                dynamic_profit_threshold, dynamic_loss_threshold = self._calculate_quantile_thresholds(
                    actual_profit, profit_threshold, loss_threshold
                )
            else:
                # 固定閾值
                dynamic_profit_threshold = profit_threshold
                dynamic_loss_threshold = loss_threshold
                print(f"📊 使用固定閾值: 盈利>{profit_threshold:.4f}, 虧損<{loss_threshold:.4f}")
            
            # 🏷️ 根據實際盈利能力生成標籤
            if label_type == "binary":
                # 二分類: 盈利(1) vs 不盈利(0)
                labels = (actual_profit > dynamic_profit_threshold).astype(int)
                print("🏷️ 二分類標籤: 1=盈利, 0=不盈利")
            else:
                # 三分類: 買入(2) vs 持有(1) vs 賣出(0)
                labels = pd.Series(
                    np.where(actual_profit > dynamic_profit_threshold, 2,      # 盈利足夠 -> 買入
                            np.where(actual_profit < dynamic_loss_threshold, 0,  # 虧損過大 -> 賣出
                                    1)),  # 其他情況 -> 持有
                    index=price_data.index
                )
                print("🏷️ 三分類標籤: 2=買入(盈利), 1=持有(中性), 0=賣出(虧損)")
            
            # 移除未來數據洩漏
            if lag > 0:
                labels = labels[:-lag]
                print(f"🔧 移除最後{lag}期數據避免洩漏，剩餘{len(labels)}條記錄")
            
            # 標籤分佈統計
            self._print_label_distribution(labels)
            
            # 處理NaN值
            labels = labels.fillna(1)  # 默認為持有
            
            return labels
            
        except Exception as e:
            print(f"❌ 標籤生成失敗: {e}")
            import traceback
            traceback.print_exc()
            # 返回默認標籤 (全部持有)
            return pd.Series(index=price_data.index[:-lag] if lag > 0 else price_data.index, 
                           data=1, dtype=int)

    def _calculate_adaptive_thresholds(self, price_data: pd.Series, 
                                     profit_threshold: float, loss_threshold: float) -> Tuple[float, float]:
        """計算自適應閾值"""
        try:
            lookback_window = min(100, len(price_data) // 4)
            rolling_volatility = price_data.pct_change().rolling(lookback_window, min_periods=20).std()
            
            # 基礎閾值 + 波動率調整
            volatility_factor = rolling_volatility / rolling_volatility.median()
            dynamic_profit_threshold = profit_threshold * (1 + volatility_factor * 0.5)
            dynamic_loss_threshold = loss_threshold * (1 + volatility_factor * 0.5)
            
            # 避免極端值
            dynamic_profit_threshold = np.clip(dynamic_profit_threshold, 
                                             profit_threshold * 0.5, profit_threshold * 3)
            dynamic_loss_threshold = np.clip(dynamic_loss_threshold, 
                                           loss_threshold * 3, loss_threshold * 0.5)
            
            print(f"📊 使用自適應閾值: 盈利={profit_threshold:.4f}±波動調整, 虧損={loss_threshold:.4f}±波動調整")
            return dynamic_profit_threshold.iloc[-1], dynamic_loss_threshold.iloc[-1]
            
        except Exception as e:
            print(f"⚠️ 自適應閾值計算失敗: {e}")
            return profit_threshold, loss_threshold

    def _calculate_quantile_thresholds(self, actual_profit: pd.Series, 
                                     pos_q: float, neg_q: float) -> Tuple[float, float]:
        """計算分位數閾值"""
        try:
            # 使用已扣成本的實際收益分布計算上下閾值
            q_high = actual_profit.quantile(pos_q)
            q_low = actual_profit.quantile(neg_q)
            
            # 安全處理：確保方向性合理
            if q_high <= 0:
                q_high = max(0.0005, actual_profit.quantile(min(0.9, max(0.6, pos_q))))
            if q_low >= 0:
                q_low = min(-0.0005, actual_profit.quantile(max(0.1, min(0.4, neg_q))))

            print(f"📊 使用分位數閾值: 上界={q_high:.5f} (q={pos_q:.3f}), 下界={q_low:.5f} (q={neg_q:.3f})")
            return q_high, q_low
            
        except Exception as e:
            print(f"⚠️ 分位數閾值計算失敗: {e}")
            return 0.005, -0.005  # 默認閾值

    def _print_label_distribution(self, labels: pd.Series):
        """打印標籤分佈統計"""
        label_counts = labels.value_counts().sort_index()
        total_samples = len(labels.dropna())
        print(f"📊 標籤分佈:")
        for label, count in label_counts.items():
            percentage = count / total_samples * 100
            label_name = {0: "賣出", 1: "持有", 2: "買入"}.get(label, f"類別{label}")
            print(f"   {label_name}({label}): {count:,} ({percentage:.1f}%)")
        
        # 檢查標籤平衡性
        if len(label_counts) < 2:
            print("⚠️ 警告: 只產生一個類別，可能閾值設置不當")
        elif label_counts.max() / total_samples > 0.9:
            print("⚠️ 警告: 數據嚴重不平衡，主導類別超過90%")

    def calculate_label_stability(self, labels: pd.Series, window: int = 10) -> float:
        """計算標籤穩定性"""
        try:
            if len(labels) < window * 2:
                return 0.0

            label_changes = labels.diff().abs()
            rolling_std = label_changes.rolling(window=window).std()

            avg_std = rolling_std.mean()
            max_possible_std = np.sqrt(len(labels.unique()) - 1)

            stability = 1 - (avg_std / max(max_possible_std, 0.001))
            return max(0.0, min(1.0, stability))

        except Exception as e:
            print(f"穩定性計算失敗: {e}")
            return 0.0

    def calculate_label_balance(self, labels: pd.Series) -> float:
        """
        計算標籤平衡性分數
        
        使用平衡分數 = min(class_ratios) / max(class_ratios)
        目標：使各類別分佈更均衡，避免極度不平衡
        """
        try:
            if len(labels) == 0:
                return 0.0
            
            # 計算各類別比例
            label_counts = labels.value_counts()
            total_samples = len(labels)
            
            if len(label_counts) < 2:
                return 0.0  # 只有一個類別，完全不平衡
            
            class_ratios = label_counts / total_samples
            min_ratio = class_ratios.min()
            max_ratio = class_ratios.max()
            
            # 計算平衡分數：理想情況下各類別相等，分數為1.0
            balance_score = min_ratio / max_ratio if max_ratio > 0 else 0.0
            
            # 額外獎勵：如果最小類別 >= 20%，給予額外分數
            if min_ratio >= 0.20:
                balance_score *= 1.2
            
            # 懲罰極度不平衡：如果最大類別 > 90%，嚴重懲罰
            if max_ratio > 0.90:
                balance_score *= 0.1
            elif max_ratio > 0.80:
                balance_score *= 0.5
            
            return max(0.0, min(1.0, balance_score))
            
        except Exception as e:
            print(f"平衡性計算失敗: {e}")
            return 0.0


    def balance_labels(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote',
                      **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """
        標籤平衡處理
        
        Args:
            X: 特徵DataFrame
            y: 標籤Series  
            method: 平衡方法 ('smote', 'adasyn', 'undersample', 'oversample')
            **kwargs: 其他參數
            
        Returns:
            平衡後的 (X, y)
        """
        try:
            print(f"🔧 使用 {method} 方法平衡標籤...")
            
            # 記錄原始分佈
            original_dist = y.value_counts().sort_index()
            print(f"📊 原始分佈: {dict(original_dist)}")
            
            # 檢查是否需要平衡
            balance_score = self.calculate_label_balance(y)
            if balance_score > 0.4:  # 如果已經相對平衡
                print(f"✅ 數據已相對平衡 (分數: {balance_score:.3f})，跳過平衡處理")
                return X, y
            
            # 根據方法進行平衡
            if method == 'smote' and IMBALANCED_LEARN_AVAILABLE:
                return self._apply_smote(X, y, **kwargs)
            elif method == 'adasyn' and IMBALANCED_LEARN_AVAILABLE:
                return self._apply_adasyn(X, y, **kwargs)
            elif method == 'undersample':
                return self._apply_undersample(X, y, **kwargs)
            elif method == 'oversample':
                return self._apply_oversample(X, y, **kwargs)
            else:
                print(f"⚠️ 方法 {method} 不可用或未安裝依賴，使用簡單過採樣")
                return self._apply_oversample(X, y, **kwargs)
                
        except Exception as e:
            print(f"❌ 標籤平衡失敗: {e}")
            return X, y

    def _apply_smote(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """應用SMOTE過採樣"""
        try:
            smote = SMOTE(random_state=42, k_neighbors=max(1, min(5, y.value_counts().min()-1)))
            X_balanced, y_balanced = smote.fit_resample(X, y)
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
            
            print(f"✅ SMOTE完成，樣本數: {len(X)} -> {len(X_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"⚠️  SMOTE失敗: {e}，改用簡單過採樣")
            return self._apply_oversample(X, y, **kwargs)

    def _apply_adasyn(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """應用ADASYN自適應過採樣"""
        try:
            adasyn = ADASYN(random_state=42)
            X_balanced, y_balanced = adasyn.fit_resample(X, y)
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced)
            
            print(f"✅ ADASYN完成，樣本數: {len(X)} -> {len(X_balanced)}")
            return X_balanced, y_balanced
        except Exception as e:
            print(f"⚠️  ADASYN失敗: {e}，改用簡單過採樣")
            return self._apply_oversample(X, y, **kwargs)

    def _apply_undersample(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """應用隨機欠採樣"""
        try:
            # 找到最小類別的樣本數
            min_samples = y.value_counts().min()
            
            X_balanced = pd.DataFrame()
            y_balanced = pd.Series(dtype=y.dtype)
            
            for label in y.unique():
                label_mask = y == label
                X_label = X[label_mask]
                y_label = y[label_mask]
                
                if len(X_label) > min_samples:
                    # 隨機採樣到最小數量
                    indices = np.random.choice(len(X_label), min_samples, replace=False)
                    X_sampled = X_label.iloc[indices]
                    y_sampled = y_label.iloc[indices]
                else:
                    X_sampled = X_label
                    y_sampled = y_label
                
                X_balanced = pd.concat([X_balanced, X_sampled], ignore_index=True)
                y_balanced = pd.concat([y_balanced, y_sampled], ignore_index=True)
            
            print(f"✅ 欠採樣完成，樣本數: {len(X)} -> {len(X_balanced)}")
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ 欠採樣失敗: {e}")
            return X, y

    def _apply_oversample(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """應用簡單過採樣"""
        try:
            # 找到最大類別的樣本數
            max_samples = y.value_counts().max()
            
            X_balanced = pd.DataFrame()
            y_balanced = pd.Series(dtype=y.dtype)
            
            for label in y.unique():
                label_mask = y == label
                X_label = X[label_mask]
                y_label = y[label_mask]
                
                if len(X_label) < max_samples:
                    # 過採樣到最大數量
                    X_resampled = resample(X_label, replace=True, n_samples=max_samples, random_state=42)
                    y_resampled = resample(y_label, replace=True, n_samples=max_samples, random_state=42)
                else:
                    X_resampled = X_label
                    y_resampled = y_label
                
                X_balanced = pd.concat([X_balanced, X_resampled], ignore_index=True)
                y_balanced = pd.concat([y_balanced, y_resampled], ignore_index=True)
            
            print(f"✅ 過採樣完成，樣本數: {len(X)} -> {len(X_balanced)}")
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ 過採樣失敗: {e}")
            return X, y

    def get_available_balance_methods(self) -> List[str]:
        """獲取可用的平衡方法列表"""
        methods = ['oversample', 'undersample']
        
        if IMBALANCED_LEARN_AVAILABLE:
            methods.extend(['smote', 'adasyn'])
            
        return methods

    def generate_report(self, labels: pd.Series) -> str:
        """生成標籤報告"""
        if labels is None or len(labels) == 0:
            return "標籤數據為空"
        
        label_counts = labels.value_counts().sort_index()
        total_samples = len(labels)
        stability = self.calculate_label_stability(labels)
        balance_score = self.calculate_label_balance(labels)
        
        report = f"""
🏷️ 標籤生成報告 - {self.symbol} {self.timeframe}
{'='*50}
📊 標籤分佈:"""

        for label, count in label_counts.items():
            percentage = count / total_samples * 100
            label_name = {0: "賣出", 1: "持有", 2: "買入"}.get(label, f"類別{label}")
            report += f"\n├─ {label_name}({label}): {count:,} ({percentage:.1f}%)"

        report += f"""

📈 質量指標:
├─ 標籤穩定性: {stability:.4f}
├─ 平衡性分數: {balance_score:.4f}
└─ 總樣本數: {total_samples:,}

💡 建議:"""

        if balance_score < 0.3:
            report += "\n├─ 標籤嚴重不平衡，建議調整閾值或使用SMOTE"
        elif balance_score < 0.5:
            report += "\n├─ 標籤輕微不平衡，可考慮數據平衡處理"
        else:
            report += "\n├─ 標籤分佈良好"

        if stability < 0.5:
            report += "\n└─ 標籤穩定性較低，可能需要調整lag參數"
        else:
            report += "\n└─ 標籤穩定性良好"

        return report


# 舊類名保持兼容性
LabelOptimizer = LabelGenerator  # 別名，保持向後兼容
