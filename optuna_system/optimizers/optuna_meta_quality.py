# -*- coding: utf-8 -*-
"""
Meta Quality Optimizer (Layer 1B)
質量評估器：評估 Primary 信號是否值得執行（二分類）

Meta-Labeling 架構的第二層：Meta Model
- 目標：評估 Primary 信號質量（執行 vs 跳過）
- 輸入：Primary 信號 (1/-1) + 市場特徵
- 輸出：1 (執行) / 0 (跳過)

參考文獻：
- Marcos López de Prado (2018), "Advances in Financial Machine Learning", Ch.3
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
import pandas as pd

from optuna_system.utils.io_utils import read_dataframe


class MetaQualityOptimizer:
    """
    Layer 1B: Meta Model - 質量評估器
    
    目標：評估 Primary 信號質量（執行 vs 跳過）
    輸入：Primary 信號 (1/-1) + 市場特徵
    輸出：1 (執行) / 0 (跳過)
    """
    
    def __init__(
        self,
        data_path: str,
        config_path: str = "configs/",
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        scaled_config: Dict = None
    ):
        """初始化 Meta Model 優化器"""
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.symbol = symbol
        self.timeframe = timeframe
        self.scaled_config = scaled_config or {}
        
        # 由外部設定（MetaLabelOptimizer 調用）
        self.primary_signals = None
        self.price_data = None
    
    def set_primary_signals(self, signals: pd.Series):
        """設定 Primary 信號（由 MetaLabelOptimizer 調用）"""
        self.primary_signals = signals
        self.logger.info(f"✅ Primary 信號已設定: {len(signals)} 筆")
    
    def _compute_final_labels(
        self, 
        primary_signals: pd.Series,  # 1/-1
        meta_quality: pd.Series       # 1/0
    ) -> pd.Series:
        """
        計算最終的三分類標籤（與 Coordinator 的邏輯一致）
        
        Args:
            primary_signals: Primary Model 的二分類信號 (1=買入, -1=賣出)
            meta_quality: Meta Model 的質量評估 (1=執行, 0=跳過)
            
        Returns:
            pd.Series: 最終三分類標籤 (0=賣出, 1=持有, 2=買入)
        """
        final_label = pd.Series(1, index=primary_signals.index)  # 默認持有
        
        # 只有 Meta 通過的信號才執行
        final_label[(primary_signals == 1) & (meta_quality == 1)] = 2   # 買入
        final_label[(primary_signals == -1) & (meta_quality == 1)] = 0  # 賣出
        
        return final_label
    
    def _select_best_from_pareto(self, study: optuna.Study) -> optuna.Trial:
        """
        從 Pareto 前沿選擇最佳折衷解（使用膝點法）
        
        Args:
            study: Optuna 多目標優化的 Study 對象
            
        Returns:
            optuna.Trial: 選中的最佳 trial
        """
        # 獲取 Pareto 前沿的所有 trials
        pareto_trials = study.best_trials
        
        if len(pareto_trials) == 0:
            self.logger.warning("⚠️ 未找到 Pareto 前沿解，返回所有試驗中的最佳")
            return max(study.trials, key=lambda t: t.values[0] if t.values else -999)
        
        self.logger.info(f"📊 Pareto 前沿包含 {len(pareto_trials)} 個解")
        
        # 方法：膝點法（Knee Point）- 找到曲線轉折最大的點
        # 標準化目標值到 [0, 1]
        obj1_values = np.array([t.values[0] for t in pareto_trials])  # 性能（最大化）
        obj2_values = np.array([t.values[1] for t in pareto_trials])  # 偏差（最小化）
        
        obj1_norm = (obj1_values - obj1_values.min()) / (obj1_values.max() - obj1_values.min() + 1e-6)
        obj2_norm = (obj2_values - obj2_values.min()) / (obj2_values.max() - obj2_values.min() + 1e-6)
        
        # 理想點：最大性能 + 最小偏差
        ideal_point = np.array([1.0, 0.0])
        
        # 計算每個解到理想點的距離
        distances = np.sqrt(
            (obj1_norm - ideal_point[0])**2 + 
            (obj2_norm - ideal_point[1])**2
        )
        
        # 選擇距離理想點最近的解
        best_idx = np.argmin(distances)
        best_trial = pareto_trials[best_idx]
        
        self.logger.info(
            f"✅ 選中折衷解: 性能={best_trial.values[0]:.4f}, "
            f"標籤偏差={best_trial.values[1]:.2%}"
        )
        
        # 記錄 Pareto 前沿的統計信息
        self.logger.info(
            f"📈 Pareto 前沿統計:\n"
            f"   性能範圍: {obj1_values.min():.4f} ~ {obj1_values.max():.4f}\n"
            f"   偏差範圍: {obj2_values.min():.2%} ~ {obj2_values.max():.2%}"
        )
        
        return best_trial
    
    def _calculate_atr(self, close: pd.Series, period: int = 14) -> pd.Series:
        """計算 ATR（簡化版）"""
        if self.price_data is None:
            # 使用價格標準差估算
            returns = close.pct_change().abs()
            atr = returns.rolling(period).mean() * close
            return atr
        
        try:
            high = self.price_data['high']
            low = self.price_data['low']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr
        except Exception as e:
            self.logger.warning(f"⚠️ ATR 計算失敗: {e}，使用簡化方法")
            returns = close.pct_change().abs()
            atr = returns.rolling(period).mean() * close
            return atr
    
    def generate_meta_features(
        self,
        primary_signals: pd.Series,
        price_data: pd.DataFrame,
        params: Dict
    ) -> pd.DataFrame:
        """
        生成元特徵（用於評估信號質量）
        
        元特徵類型：
        1. 信號強度：ATR 倍數、止盈距離
        2. 市場環境：波動率、趨勢強度、成交量
        3. 信號一致性：價格動量
        4. 歷史表現：近期勝率（滾動窗口）
        
        Returns:
            pd.DataFrame: 元特徵矩陣
        """
        meta_features = pd.DataFrame(index=primary_signals.index)
        close = price_data['close']
        
        # === 特徵組 1：信號強度特徵 ===
        atr_period = params.get('atr_period', 14)
        profit_multiplier = params.get('profit_multiplier', 2.0)
        
        # 1.1 信號強度（止盈距離 / 價格）
        atr = self._calculate_atr(close, atr_period)
        atr = atr.reindex(primary_signals.index).fillna(method='ffill').fillna(close.std() * 0.02)
        meta_features['signal_strength'] = (atr * profit_multiplier) / close
        
        # 1.2 風險回報比
        stop_multiplier = params.get('stop_multiplier', 1.5)
        meta_features['risk_reward_ratio'] = profit_multiplier / stop_multiplier
        
        # === 特徵組 2：市場環境特徵 ===
        # 2.1 波動率（20 期標準差）
        returns = close.pct_change()
        meta_features['volatility'] = returns.rolling(20).std()
        
        # 2.2 趨勢強度（價格 / 50 期均線 - 1）
        sma_50 = close.rolling(50).mean()
        meta_features['trend_strength'] = (close / sma_50 - 1)
        
        # 2.3 成交量比率（當前成交量 / 20 期均量）
        if 'volume' in price_data.columns:
            volume = price_data['volume']
            avg_volume = volume.rolling(20).mean()
            meta_features['volume_ratio'] = volume / avg_volume
        else:
            meta_features['volume_ratio'] = 1.0
        
        # === 特徵組 3：價格動量特徵 ===
        # 3.1 短期動量（5 期收益率）
        meta_features['momentum_5'] = close.pct_change(5)
        
        # 3.2 中期動量（20 期收益率）
        meta_features['momentum_20'] = close.pct_change(20)
        
        # === 特徵組 4：歷史表現特徵 ===
        # 4.1 近期勝率（滾動窗口）
        meta_features['recent_winrate'] = self._calculate_rolling_winrate(
            primary_signals, price_data, window=20, lag=params.get('lag', 12)
        )
        
        # 4.2 信號與動量一致性
        signal_direction = primary_signals  # 1 或 -1
        momentum_direction = np.sign(meta_features['momentum_5'])
        meta_features['signal_momentum_alignment'] = (signal_direction == momentum_direction).astype(int)
        
        # 填充缺失值
        meta_features = meta_features.fillna(method='ffill').fillna(0)
        
        return meta_features
    
    def _calculate_rolling_winrate(
        self,
        signals: pd.Series,
        price_data: pd.DataFrame,
        window: int = 20,
        lag: int = 12
    ) -> pd.Series:
        """
        計算滾動勝率（避免未來數據洩露）
        
        Args:
            signals: Primary 信號 (1/-1)
            price_data: 價格數據
            window: 滾動窗口大小
            lag: 未來收益的 lag 期數
        
        Returns:
            pd.Series: 滾動勝率
        """
        close = price_data['close']
        
        # 計算未來收益
        future_returns = close.pct_change(lag).shift(-lag)
        
        # 計算信號收益
        signal_returns = signals.shift(1) * future_returns
        
        # 勝負判斷（收益 > 0 為勝）
        wins = (signal_returns > 0).astype(int)
        
        # 滾動勝率
        rolling_winrate = wins.rolling(window, min_periods=5).mean()
        
        return rolling_winrate
    
    def generate_meta_labels(
        self,
        primary_signals: pd.Series,
        price_data: pd.DataFrame,
        params: Dict
    ) -> pd.Series:
        """
        生成元標籤（訓練目標）
        
        邏輯：
        - 執行 Primary 信號能獲利 → 1 (執行)
        - 否則 → 0 (跳過)
        
        Returns:
            pd.Series: 1=好信號（執行），0=壞信號（跳過）
        """
        close = price_data['close']
        lag = params.get('lag', 12)
        
        # 計算未來收益
        future_returns = close.pct_change(lag).shift(-lag)
        
        # 計算信號收益
        signal_returns = primary_signals.shift(1) * future_returns
        
        # 定義「好信號」：獲利超過 2× 交易成本
        transaction_cost = params.get('transaction_cost_bps', 10) / 10000
        
        # 元標籤：信號收益 > 雙邊交易成本 → 1（執行）
        meta_labels = (signal_returns > transaction_cost * 2).astype(int)
        
        return meta_labels
    
    def calculate_filtered_trading_metrics(
        self,
        quality_predictions: pd.Series,
        primary_signals: pd.Series,
        price_data: pd.DataFrame,
        params: Dict
    ) -> Dict:
        """
        🎯 計算Meta過濾後的交易質量指標（優化目標重構 - Phase 2）
        
        這是新增的核心函數，用於計算Meta Model過濾後的交易效果。
        
        Args:
            quality_predictions: Meta模型預測的質量（1=高質量執行，0=低質量跳過）
            primary_signals: Primary模型的原始信號（1=買入，-1=賣出）
            price_data: 價格數據（包含OHLCV）
            params: 參數
        
        Returns:
            Dict: 過濾後的交易質量指標
        """
        try:
            # 應用Meta過濾
            filtered_signals = primary_signals.copy()
            filtered_signals[quality_predictions == 0] = 0  # 低質量信號設為0（不執行）
            
            # 提取參數
            lag = params.get('lag', 12)
            
            # 計算執行率
            total_primary = (primary_signals != 0).sum()
            total_executed = (filtered_signals != 0).sum()
            execution_ratio = total_executed / total_primary if total_primary > 0 else 0.0
            
            # 如果過濾後交易太少，返回默認值
            if total_executed < 30:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'execution_ratio': execution_ratio,
                    'sharpe': 0.0
                }
            
            # 計算未來收益（用於評估）
            future_returns = price_data['close'].pct_change(lag).shift(-lag)
            future_returns = future_returns.loc[filtered_signals.index]
            
            # 計算交易收益
            trade_returns = []
            for idx in filtered_signals.index:
                if filtered_signals[idx] != 0:
                    signal = filtered_signals[idx]
                    ret = future_returns.loc[idx]
                    if not np.isnan(ret):
                        trade_return = signal * ret
                        trade_returns.append(trade_return)
            
            if len(trade_returns) == 0:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'execution_ratio': execution_ratio,
                    'sharpe': 0.0
                }
            
            trade_returns = pd.Series(trade_returns)
            
            # 計算勝率
            wins = trade_returns[trade_returns > 0]
            losses = trade_returns[trade_returns <= 0]
            win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0.0
            
            # 計算盈利因子
            total_wins = wins.sum() if len(wins) > 0 else 0.0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
            profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
            profit_factor = min(profit_factor, 10.0)  # 上限10
            
            # 計算平均盈虧
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
            
            # 計算Sharpe
            sharpe = (trade_returns.mean() / (trade_returns.std() + 1e-6)) * np.sqrt(252)
            sharpe = max(min(sharpe, 10.0), -10.0)  # 限制在[-10, 10]
            
            # 計算總盈虧
            total_pnl = trade_returns.sum()
            
            return {
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'total_trades': len(trade_returns),
                'total_pnl': float(total_pnl),
                'execution_ratio': float(execution_ratio),
                'sharpe': float(sharpe)
            }
        
        except Exception as e:
            self.logger.error(f"❌ 過濾後交易質量計算失敗: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_trades': 0,
                'total_pnl': 0.0,
                'execution_ratio': 0.0,
                'sharpe': 0.0
            }
    
    def evaluate_quality(
        self,
        meta_features: pd.DataFrame,
        params: Dict
    ) -> pd.Series:
        """
        使用元特徵評估質量（閾值分類器）
        
        評分公式：
        quality_score = w1 × signal_strength 
                      + w2 × |trend_strength|
                      + w3 × recent_winrate
                      + w4 × signal_momentum_alignment
        
        Returns:
            pd.Series: 1=高質量（執行），0=低質量（跳過）
        """
        # 提取權重
        strength_weight = params.get('strength_weight', 0.3)
        trend_weight = params.get('trend_weight', 0.3)
        winrate_weight = params.get('winrate_weight', 0.2)
        alignment_weight = params.get('alignment_weight', 0.2)
        
        # 計算質量分數
        quality_score = (
            meta_features['signal_strength'] * strength_weight +
            meta_features['trend_strength'].abs() * trend_weight +
            meta_features['recent_winrate'] * winrate_weight +
            meta_features['signal_momentum_alignment'] * alignment_weight
        )
        
        # 二分類：分數 > 閾值 → 執行
        quality_threshold = params.get('quality_threshold', 0.5)
        quality_labels = (quality_score > quality_threshold).astype(int)
        
        return quality_labels
    
    def objective(self, trial: optuna.Trial) -> Tuple[float, float]:
        """
        🎯 多目標優化目標函數（優化目標重構 - Phase 2）
        
        ✅ 新優化目標：
        1. 最大化：過濾後的交易質量（Win Rate 35% + Profit Factor 30% + Execution Ratio 20% + F1 10% + Sharpe 5%）
        2. 最小化：標籤分布偏差
        
        ❌ 已修改的舊目標：
        - F1從50%降低到10%（保留但不是主要目標）
        - Sharpe從5%保持5%（但使用過濾後的Sharpe）
        - 新增Win Rate 35%
        - 新增Profit Factor 30%
        - 新增Execution Ratio 20%
        
        Returns:
            Tuple[float, float]: (性能得分, 標籤偏差)
        """
        if self.primary_signals is None or self.price_data is None:
            self.logger.error("❌ 必須先設定 Primary 信號和價格數據")
            return -999.0, 999.0
        
        # 🔧 參數搜索空間（優化目標重構 - Phase 2）
        params = {
            'lag': 12,  # 從 Primary Model 繼承
            'atr_period': 14,
            'profit_multiplier': 2.0,
            'stop_multiplier': 1.5,
            # ✅ 修改：降低交易成本範圍（從5.0-15.0降低到3.0-6.0）
            'transaction_cost_bps': trial.suggest_float('transaction_cost_bps', 3.0, 6.0),
            
            # Meta Model 特定參數
            'strength_weight': trial.suggest_float('strength_weight', 0.1, 0.5),
            'trend_weight': trial.suggest_float('trend_weight', 0.1, 0.4),
            'winrate_weight': trial.suggest_float('winrate_weight', 0.1, 0.4),
            'alignment_weight': trial.suggest_float('alignment_weight', 0.05, 0.3),
            
            # 🎯 P0修復：降低quality_threshold避免Meta過度保守
            # 問題分析（v63測試）：
            # - 舊範圍0.32-0.48導致90%信號變持有，執行率僅9.5%
            # - 召回率僅13.1%，錯過大量交易機會
            # 
            # 修復方案：
            # - 降低到0.20-0.35範圍
            # - 預期執行率提升到15-25%
            # - 持有比例從90%降到70-80%
            # - 保持高精度（77.6%）同時提升召回率
            # 
            # 學術依據：Meta-Labeling應過濾低質量信號，但不應過度保守
            # 參考：López de Prado (2018) Ch.3 - Meta-Labeling
            'quality_threshold': trial.suggest_float('quality_threshold', 0.20, 0.35),
        }
        
        try:
            # 生成元特徵和元標籤
            meta_features = self.generate_meta_features(
                self.primary_signals, self.price_data, params
            )
            meta_labels = self.generate_meta_labels(
                self.primary_signals, self.price_data, params
            )
            
            # 預測質量
            predicted_quality = self.evaluate_quality(meta_features, params)
            
            # 確保有足夠的樣本（放宽约束：100→50）
            if len(predicted_quality) < 50:
                self.logger.warning(f"⚠️ 樣本數過少: {len(predicted_quality)} < 50")
                return -999.0, 999.0
            
            # 🎯 目標 1：過濾後的交易質量（優化目標重構 - Phase 2）
            # ✅ 新增：計算Meta過濾後的交易質量指標
            filtered_metrics = self.calculate_filtered_trading_metrics(
                predicted_quality,
                self.primary_signals,
                self.price_data,
                params
            )
            
            # ✅ 保留F1計算（用於監控，但權重降低）
            tp = ((predicted_quality == 1) & (meta_labels == 1)).sum()
            fp = ((predicted_quality == 1) & (meta_labels == 0)).sum()
            fn = ((predicted_quality == 0) & (meta_labels == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            
            # ✅ 硬約束檢查（過濾後交易太少）
            if filtered_metrics['total_trades'] < 30:
                self.logger.warning(f"⚠️ 過濾後交易數過少: {filtered_metrics['total_trades']} < 30")
                return -999.0, 999.0
            
            if filtered_metrics['execution_ratio'] < 0.15:
                self.logger.warning(f"⚠️ 執行率過低: {filtered_metrics['execution_ratio']:.1%} < 15%")
                return -999.0, 999.0
            
            # ✅ 新性能得分（交易質量主導）
            # 權重配置：交易質量65% + 執行率20% + ML指標10% + Sharpe 5%
            performance_score_base = (
                # 過濾後的交易質量（65%）
                filtered_metrics['win_rate'] * 0.35 +                      # 過濾後勝率 35%
                (filtered_metrics['profit_factor'] / 3.0) * 0.30 +         # 過濾後盈利因子 30%（歸一化）
                
                # 執行率（20%）- 確保不會過度過濾
                filtered_metrics['execution_ratio'] * 0.20 +
                
                # ML指標（10%）- 保留但降低權重
                f1 * 0.10 +
                
                # Sharpe（5%）- 輔助指標
                max(0, min(filtered_metrics['sharpe'], 10)) / 10 * 0.05  # 歸一化到0-1
            )
            
            # ✅ 記錄過濾後的交易質量指標
            trial.set_user_attr("filtered_win_rate", filtered_metrics['win_rate'])
            trial.set_user_attr("filtered_profit_factor", filtered_metrics['profit_factor'])
            trial.set_user_attr("filtered_total_trades", filtered_metrics['total_trades'])
            trial.set_user_attr("filtered_sharpe", filtered_metrics['sharpe'])
            
            # 🎯 目標 2：標籤分布偏差（需要最小化）
            execution_ratio = (predicted_quality == 1).sum() / len(predicted_quality)
            
            # 🔧 P0修复v2：改为软约束（惩罚而非拒绝）
            # 问题分析：
            # - 硬约束（返回-999）导致Optuna无解空间
            # - 即使execution_ratio=7.6%在范围内，仍可能因为其他原因失败
            # 
            # 修复方案：软约束（允许探索，但惩罚偏离）
            # - 理想范围：5-40%
            # - 偏离惩罚：线性降低performance_score
            # - 不再直接返回-999
            
            # 计算execution_ratio偏离惩罚
            exec_penalty = 0.0
            if execution_ratio < 0.05:
                # 太低（<5%）：惩罚力度随偏离增大
                exec_penalty = (0.05 - execution_ratio) * 2.0  # 最多-0.1
            elif execution_ratio > 0.40:
                # 太高（>40%）：惩罚力度随偏离增大
                exec_penalty = (execution_ratio - 0.40) * 1.0  # 最多-0.6
            
            # 记录约束状态（软约束，不再硬拒绝）
            trial.set_user_attr("execution_ratio", execution_ratio)
            trial.set_user_attr("exec_penalty", exec_penalty)
            trial.set_user_attr("constraint_violated", False)  # 软约束不算违反
            
            # 計算最終三分類標籤的分布
            final_labels = self._compute_final_labels(self.primary_signals, predicted_quality)
            
            label_dist = {}
            for i in [0, 1, 2]:
                label_dist[i] = (final_labels == i).sum() / len(final_labels)
            
            # 與目標分布的最大偏差
            target_dist = [0.25, 0.50, 0.25]
            max_deviation = max(
                abs(label_dist[0] - target_dist[0]),
                abs(label_dist[1] - target_dist[1]),
                abs(label_dist[2] - target_dist[2])
            )
            
            # 应用软约束惩罚
            performance_score = performance_score_base - exec_penalty
            
            # 記錄詳細信息
            trial.set_user_attr("f1_score", f1)
            trial.set_user_attr("precision", precision)
            trial.set_user_attr("recall", recall)
            trial.set_user_attr("sharpe", filtered_metrics['sharpe'])  # ✅ 修复：使用filtered_metrics
            trial.set_user_attr("execution_ratio", execution_ratio)
            trial.set_user_attr("label_deviation", max_deviation)
            trial.set_user_attr("buy_pct", label_dist[2])
            trial.set_user_attr("hold_pct", label_dist[1])
            trial.set_user_attr("sell_pct", label_dist[0])
            trial.set_user_attr("performance_score_base", performance_score_base)
            trial.set_user_attr("performance_score_final", performance_score)
            trial.set_user_attr("constraint_violated", False)
            
            return performance_score, max_deviation
            
        except Exception as e:
            self.logger.warning(f"⚠️ Meta Model 評估失敗: {e}")
            return -999.0, 999.0
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """
        執行多目標優化（NSGA-II）
        
        Args:
            n_trials: 試驗次數（推薦 400-600）
            
        Returns:
            Dict: 優化結果，包含最佳參數和性能指標
        """
        self.logger.info("🚀 Meta Model 多目標優化開始（NSGA-II）...")
        self.logger.info(f"   目標1: 最大化模型性能（F1 + Sharpe）")
        self.logger.info(f"   目標2: 最小化標籤分布偏差")
        self.logger.info(f"   總試驗數: {n_trials}")
        
        if self.primary_signals is None:
            raise ValueError("❌ 必須先設定 Primary 信號 (set_primary_signals)")
        
        # 🎯 使用 NSGA-II 採樣器
        # population_size: 建議為 30-40，試驗數越多可以稍大
        population_size = min(40, max(30, n_trials // 15))
        
        sampler = NSGAIISampler(
            population_size=population_size,
            mutation_prob=None,  # 自動調整
            crossover_prob=0.9,
            swapping_prob=0.5,
            seed=42  # 可重現性
        )
        
        study = optuna.create_study(
            directions=['maximize', 'minimize'],  # 最大化性能，最小化偏差
            sampler=sampler,
            study_name=f'meta_quality_multiobjective_{self.timeframe}'
        )
        
        self.logger.info(f"   種群大小: {population_size}")
        self.logger.info(f"   預計代數: {n_trials // population_size}")
        
        # 執行優化
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=False)
        
        # 從 Pareto 前沿選擇最佳折衷解
        best_trial = self._select_best_from_pareto(study)
        
        best_params = best_trial.params
        performance_score = best_trial.values[0]
        label_deviation = best_trial.values[1]
        
        self.logger.info(f"✅ Meta Model 優化完成!")
        self.logger.info(f"📋 最佳參數: {best_params}")
        self.logger.info(f"📊 性能得分: {performance_score:.4f}")
        self.logger.info(f"📊 標籤偏差: {label_deviation:.2%}")
        
        # 統計 Pareto 前沿
        pareto_count = len(study.best_trials)
        self.logger.info(f"🎯 Pareto 前沿解數量: {pareto_count}/{n_trials}")
        
        # 返回結果
        return {
            'best_params': best_params,
            'best_score': performance_score,
            'label_deviation': label_deviation,
            'n_trials': n_trials,
            'pareto_front_size': pareto_count,
            'study': study,
            'f1_score': best_trial.user_attrs.get('f1_score', 0),
            'precision': best_trial.user_attrs.get('precision', 0),
            'recall': best_trial.user_attrs.get('recall', 0),
            'sharpe': best_trial.user_attrs.get('sharpe', 0),
            'execution_ratio': best_trial.user_attrs.get('execution_ratio', 0),
            'buy_pct': best_trial.user_attrs.get('buy_pct', 0),
            'hold_pct': best_trial.user_attrs.get('hold_pct', 0),
            'sell_pct': best_trial.user_attrs.get('sell_pct', 0)
        }
    
    def apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """統一物化接口（Coordinator 調用）"""
        if self.primary_signals is None:
            raise ValueError("必須先調用 set_primary_signals() 設置 Primary 信號")
        
        meta_features = self.generate_meta_features(self.primary_signals, data, params)
        meta_quality = self.evaluate_quality(meta_features, params)
        
        result = data.loc[meta_quality.index].copy()
        result['meta_quality'] = meta_quality
        return result


if __name__ == "__main__":
    # 獨立測試腳本
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    print("🚀 Meta Quality Optimizer 測試")
    print("⚠️ 需要先運行 Primary Model 生成信號")
    
    # 這裡需要從 Primary Model 獲取信號
    # 實際使用時由 MetaLabelOptimizer 協調
    
    print("✅ Meta Model 框架已準備就緒")
    print("📋 待整合到 MetaLabelOptimizer 中使用")
