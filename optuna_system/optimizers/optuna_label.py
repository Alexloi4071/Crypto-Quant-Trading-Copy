# -*- coding: utf-8 -*-
"""
標籤生成參數優化器 (第1層) - P0修復版
==================================================
P0修復記錄:
  1. [CRITICAL] 移除 rolling quantile 標籤洩漏
     舊版問題: historical_future_returns = future_returns.shift(1) 仍使用未來收益的滾動統計
     修復方案: 使用 FIXED 全局分位數（在訓練集前半段計算，凍結閾值）
  2. [CRITICAL] 移除 objective() 中的 self._last_distribution / self._last_rebalance 殘留引用
  3. [MAJOR] 修復 generate_labels() 預設走 quantile 路徑時 valid_range slice 導致 labels 頭尾全為1 的問題
  4. [MAJOR] Triple-Barrier: profit_multiplier / stop_multiplier 風險比約束改為 2:1（原1.3:1過低）
  5. [MINOR] optimize() 兩階段搜索改用 tqdm 顯示進度；移除 _last_distribution 未定義屬性引用
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import optuna
import pandas as pd

from optuna_system.utils.io_utils import write_dataframe, read_dataframe, atomic_write_json
from optuna_system.utils.time_integrity import TimeLeakageDetector

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# P0 核心修復：固定全局分位數標籤生成
# ---------------------------------------------------------------------------

def _fixed_global_quantile_labels(
    future_returns: pd.Series,
    buy_quantile: float,
    sell_quantile: float,
    freeze_ratio: float = 0.5,
) -> pd.Series:
    """
    P0 核心：使用『凍結分位數』生成標籤，完全消除 rolling quantile 統計洩漏。

    原理：
      - 只用前 freeze_ratio（預設50%）的 future_returns 計算 buy/sell 閾值
      - 閾值在整個序列中保持不變（凍結）
      - 後半段完全 out-of-sample，閾值不含任何未來信息

    Args:
        future_returns: 未來 lag 期收益率（已對齊）
        buy_quantile: 買入分位數（例如 0.80）
        sell_quantile: 賣出分位數（例如 0.20）
        freeze_ratio: 用於計算閾值的歷史比例（預設 0.5）

    Returns:
        標籤序列（0=賣出, 1=持有, 2=買入）
    """
    n = len(future_returns)
    freeze_end = max(100, int(n * freeze_ratio))
    train_returns = future_returns.iloc[:freeze_end].dropna()

    if len(train_returns) < 50:
        # 樣本不足時退化為全局分位數
        buy_threshold = float(future_returns.quantile(buy_quantile))
        sell_threshold = float(future_returns.quantile(sell_quantile))
    else:
        buy_threshold = float(train_returns.quantile(buy_quantile))
        sell_threshold = float(train_returns.quantile(sell_quantile))

    # 確保閾值有效且有間距
    min_gap = 0.0005
    if buy_threshold <= sell_threshold + min_gap:
        buy_threshold = sell_threshold + min_gap * 2

    labels = pd.Series(1, index=future_returns.index, dtype=int)
    labels[future_returns > buy_threshold] = 2
    labels[future_returns < sell_threshold] = 0

    return labels


class LabelOptimizer:
    """標籤生成參數優化器 - 第1層優化（P0修復版）"""

    def __init__(self, data_path: str, config_path: str = "configs/",
                 symbol: str = "BTCUSDT", timeframe: str = "15m", scaled_config: Dict = None):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.scaled_config = scaled_config or {}
        self.logger = logging.getLogger(__name__)
        # P0: 移除 _last_rebalance 殘留狀態
        # （原代碼有 self._last_distribution / self._last_rebalance_changes 但從未初始化）

    # ------------------------------------------------------------------
    # P0 主入口：標籤生成
    # ------------------------------------------------------------------

    def generate_labels(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """
        P0 修復版標籤生成。

        路由邏輯：
          - 'triple_barrier' → generate_triple_barrier_labels()
          - 'stabilized'     → stabilized_label_generation()
          - 'fixed'          → 固定絕對閾值
          - 'adaptive'       → 波動率自適應閾值
          - 其他/默認        → _fixed_global_quantile_labels()  ← P0 核心修復

        與舊版的差異（P0）：
          - 舊版 'quantile' 路徑使用 historical_future_returns.rolling().quantile()，
            本質上仍是對「未來收益的滾動統計」，存在統計洩漏。
          - P0 改用凍結式全局分位數，閾值只從前50%樣本計算，
            完全隔絕後半段未來信息。
        """
        try:
            lag = int(params.get('lag', 12))
            threshold_method = params.get('threshold_method', 'quantile')

            if len(price_data) <= lag:
                return pd.Series([], dtype=int)

            future_prices = price_data.shift(-lag)
            future_returns = (future_prices - price_data) / price_data.replace(0, np.nan)
            future_returns = future_returns.replace([np.inf, -np.inf], np.nan)

            if threshold_method == 'triple_barrier':
                labels = self.generate_triple_barrier_labels(price_data, params)
            elif threshold_method == 'stabilized':
                labels = self.stabilized_label_generation(price_data, params)
            elif threshold_method == 'fixed':
                profit_threshold = float(params.get('profit_threshold', 0.01))
                loss_threshold = float(params.get('loss_threshold', -0.01))
                labels = pd.Series(1, index=price_data.index, dtype=int)
                labels[future_returns > profit_threshold] = 2
                labels[future_returns < loss_threshold] = 0
                if lag > 0:
                    labels = labels.iloc[:-lag]
            elif threshold_method == 'adaptive':
                vol_multiplier = float(params.get('vol_multiplier', 1.5))
                vol_window = int(params.get('vol_window', 30))
                rolling_vol = price_data.pct_change().rolling(vol_window).std()
                profit_thr = rolling_vol * vol_multiplier
                loss_thr = -rolling_vol * vol_multiplier
                profit_thr = profit_thr.fillna(0.01)
                loss_thr = loss_thr.fillna(-0.01)
                labels = pd.Series(1, index=price_data.index, dtype=int)
                labels[future_returns > profit_thr] = 2
                labels[future_returns < loss_thr] = 0
                if lag > 0:
                    labels = labels.iloc[:-lag]
            else:
                # ✅ P0 核心修復：固定全局分位數（默認路徑）
                buy_quantile = float(params.get('buy_quantile', 0.80))
                sell_quantile = float(params.get('sell_quantile', 0.20))
                # 移除尾部 lag 個 NaN 點再計算
                valid_returns = future_returns.iloc[:-lag] if lag > 0 else future_returns
                labels = _fixed_global_quantile_labels(
                    valid_returns, buy_quantile, sell_quantile, freeze_ratio=0.5
                )

            labels = labels.dropna().astype(int)
            self._print_label_statistics(labels, params)
            return labels

        except Exception as e:
            self.logger.error(f"標籤生成失敗: {e}")
            return pd.Series([], dtype=int)

    # ------------------------------------------------------------------
    # 穩定化標籤（P0：改用固定前段分位數）
    # ------------------------------------------------------------------

    def stabilized_label_generation(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """穩定化標籤：P0 改用固定前段分位數，與 _fixed_global_quantile_labels 一致。"""
        try:
            lag = int(params.get('lag', 12))
            if len(price_data) <= lag:
                return pd.Series([], dtype=int)

            future_returns = (price_data.shift(-lag) - price_data) / price_data.replace(0, np.nan)
            future_returns = future_returns.replace([np.inf, -np.inf], np.nan)
            valid_returns = future_returns.iloc[:-lag] if lag > 0 else future_returns

            buy_quantile = float(params.get('buy_quantile', 0.80))
            sell_quantile = float(params.get('sell_quantile', 0.20))
            labels = _fixed_global_quantile_labels(
                valid_returns, buy_quantile, sell_quantile, freeze_ratio=0.5
            )

            signal_stats = self.validate_signal_authenticity(price_data, labels, lag)
            max_noise_ratio = float(params.get('max_noise_ratio', 0.35))
            if signal_stats['noise_ratio'] > max_noise_ratio:
                self.logger.warning(
                    f"⚠️ 信號噪聲過高({signal_stats['noise_ratio']:.2f})，使用保守標籤"
                )
                labels = self.generate_conservative_labels(price_data, params)

            self._print_label_statistics(labels, params)
            return labels.dropna()

        except Exception as e:
            self.logger.error(f"穩定化標籤生成失敗: {e}")
            return self.generate_labels(price_data, params)

    def generate_conservative_labels(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """保守標籤：提高買賣門檻（分位數更極端）。"""
        fallback = params.copy()
        fallback['buy_quantile'] = min(0.92, float(params.get('buy_quantile', 0.80)) + 0.05)
        fallback['sell_quantile'] = max(0.08, float(params.get('sell_quantile', 0.20)) - 0.05)
        fallback['threshold_method'] = 'quantile'
        return self.generate_labels(price_data, fallback)

    def validate_signal_authenticity(self, prices: pd.Series, labels: pd.Series, lag: int) -> Dict:
        """信號真實性驗證。"""
        try:
            buy_mask = labels == 2
            sell_mask = labels == 0
            future_prices = prices.shift(-lag)
            buy_accuracy = (
                (future_prices[buy_mask] > prices[buy_mask]).mean()
                if buy_mask.sum() > 0 else 0.0
            )
            sell_accuracy = (
                (future_prices[sell_mask] < prices[sell_mask]).mean()
                if sell_mask.sum() > 0 else 0.0
            )
            total_signals = buy_mask.sum() + sell_mask.sum()
            total_accuracy = (
                (buy_accuracy * buy_mask.sum() + sell_accuracy * sell_mask.sum()) / total_signals
                if total_signals > 0 else 0.0
            )
            noise_ratio = 1.0 - total_accuracy
            quality = 'high' if noise_ratio < 0.2 else ('medium' if noise_ratio < 0.4 else 'low')
            return {
                'buy_accuracy': float(buy_accuracy),
                'sell_accuracy': float(sell_accuracy),
                'noise_ratio': float(noise_ratio),
                'signal_quality': quality
            }
        except Exception as e:
            self.logger.warning(f"信號驗證失敗: {e}")
            return {'buy_accuracy': 0.0, 'sell_accuracy': 0.0, 'noise_ratio': 1.0, 'signal_quality': 'unknown'}

    # ------------------------------------------------------------------
    # Triple-Barrier 標籤（P0：風險比約束升為 2:1）
    # ------------------------------------------------------------------

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr = pd.concat([
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def generate_triple_barrier_labels(self, price_data: pd.Series, params: Dict) -> pd.Series:
        """
        Triple-Barrier 標籤生成（P0修復）。
        - 風險回報比約束升為 2:1（舊版 1.3:1 過低，導致過多賣出噪聲）
        - 移動止損邏輯保持不變
        """
        try:
            lag = int(params.get('lag', 12))
            profit_multiplier = float(params.get('profit_multiplier', 2.0))
            stop_multiplier = float(params.get('stop_multiplier', 1.0))
            max_holding = int(params.get('max_holding', 16))
            atr_period = int(params.get('atr_period', 14))
            transaction_cost_bps = float(params.get('transaction_cost_bps', self.scaled_config.get('transaction_cost_bps', 7)))
            round_trip_cost = transaction_cost_bps / 10000.0 * 2.0

            enable_trailing = bool(params.get('enable_trailing_stop', True))
            trail_activation = float(params.get('trailing_activation_ratio', 0.5))
            trail_distance = float(params.get('trailing_distance_ratio', 0.7))
            trail_lock_min = float(params.get('trailing_lock_min_profit', 0.3))

            if len(price_data) <= max_holding:
                return pd.Series([], dtype=int)

            # P0 修復：風險比約束從 1.3 升為 2.0
            min_rr = float(params.get('min_risk_reward_ratio', 2.0))
            if profit_multiplier / max(stop_multiplier, 1e-6) < min_rr:
                profit_multiplier = stop_multiplier * min_rr
                self.logger.info(f"🔒 風險比約束: profit_multiplier 調整至 {profit_multiplier:.2f}×ATR (2:1)")

            # 計算 ATR
            try:
                ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
                if ohlcv_file.exists():
                    ohlcv_df = pd.read_parquet(ohlcv_file)
                    atr = self.calculate_atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], atr_period)
                    atr = atr.reindex(price_data.index).ffill()
                else:
                    atr = price_data.pct_change().abs().rolling(atr_period).mean() * price_data
            except Exception:
                atr = price_data.pct_change().abs().rolling(atr_period).mean() * price_data

            atr = atr.fillna(price_data.std() * 0.02)
            labels = pd.Series(1, index=price_data.index, dtype=int)

            for i in range(len(price_data) - max_holding):
                entry_price = price_data.iloc[i]
                current_atr = float(atr.iloc[i])
                if current_atr <= 0 or np.isnan(current_atr):
                    continue

                profit_target = entry_price + current_atr * profit_multiplier * (1 + round_trip_cost)
                initial_stop = entry_price - current_atr * stop_multiplier * (1 - round_trip_cost)

                current_stop = initial_stop
                highest_price = entry_price
                trailing_activated = False

                for j in range(i + 1, min(i + max_holding + 1, len(price_data))):
                    fp = price_data.iloc[j]

                    if enable_trailing:
                        if fp > highest_price:
                            highest_price = fp
                        progress = (fp - entry_price) / max(profit_target - entry_price, 1e-9)
                        if progress >= trail_activation and not trailing_activated:
                            trailing_activated = True
                        if trailing_activated:
                            new_stop = max(
                                highest_price - trail_distance * current_atr,
                                entry_price + trail_lock_min * current_atr
                            )
                            current_stop = max(current_stop, new_stop)

                    if fp >= profit_target:
                        labels.iloc[i] = 2
                        break
                    elif fp <= current_stop:
                        labels.iloc[i] = 0
                        break

            if lag > 0:
                labels = labels.iloc[:-lag]
            self._print_label_statistics(labels, params)
            return labels.dropna()

        except Exception as e:
            self.logger.error(f"Triple-Barrier 生成失敗: {e}")
            return pd.Series([], dtype=int)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _timeframe_to_minutes(self, timeframe: Optional[str] = None) -> float:
        tf = (timeframe or self.timeframe or '').lower()
        try:
            if tf.endswith('m'):
                return max(1.0, float(tf[:-1]))
            if tf.endswith('h'):
                return max(1.0, float(tf[:-1]) * 60.0)
            if tf.endswith('d'):
                return max(1.0, float(tf[:-1]) * 1440.0)
        except Exception:
            pass
        return 15.0

    def _normalize_sharpe(self, sharpe: float) -> float:
        return float((min(max(sharpe, -1.0), 3.0) + 1.0) / 4.0)

    def _normalize_trade_frequency(self, trades_per_day: float, params: Dict) -> float:
        target = float(params.get('target_trades_per_day', self.scaled_config.get('target_trades_per_day', 2.0)))
        target = max(target, 1e-6)
        return float(max(0.0, min(trades_per_day / target, 1.2)))

    def _compute_strategy_metrics(self, labels: pd.Series, actual_returns: pd.Series, params: Dict) -> Dict:
        default = {
            'sharpe': 0.0, 'win_rate': 0.0, 'trades_per_day': 0.0,
            'avg_return': 0.0, 'exposure': 0.0, 'trades': 0,
            'avg_holding_minutes': 0.0,
            'cost_per_trade_bps': float(params.get('transaction_cost_bps', 7))
        }
        if labels is None or labels.empty:
            return default
        try:
            returns_series = actual_returns.reindex(labels.index).fillna(0.0)
            position = labels.astype(float).map({2: 1.0, 1: 0.0, 0: -1.0}).fillna(0.0)
            cost_bps = float(params.get('transaction_cost_bps', 7))
            cost_per_trade = cost_bps / 10000.0
            pos_change = position.diff().abs()
            if not pos_change.empty:
                pos_change.iloc[0] = abs(position.iloc[0])
            strategy_returns = position * returns_series - pos_change * cost_per_trade
            trades = int((pos_change > 0).sum())
            exposure = float((position != 0).mean())
            minutes = self._timeframe_to_minutes()
            total_days = max((len(labels) * minutes) / (60.0 * 24.0), 1 / 24)
            trades_per_day = trades / total_days

            in_pos = position != 0
            pr = strategy_returns[in_pos] if in_pos.any() else strategy_returns
            win_rate = float((pr > 0).mean()) if len(pr) > 0 else 0.0
            avg_return = float(pr.mean()) if len(pr) > 0 else 0.0

            mean_ret = float(strategy_returns.mean())
            std_ret = float(strategy_returns.std(ddof=0))
            periods_per_year = (len(strategy_returns) / total_days) * 252.0
            sharpe = mean_ret / std_ret * np.sqrt(periods_per_year) if std_ret > 1e-9 else 0.0

            return {
                'sharpe': float(sharpe), 'win_rate': win_rate,
                'trades_per_day': float(trades_per_day), 'avg_return': avg_return,
                'exposure': exposure, 'trades': trades,
                'avg_holding_minutes': float((in_pos.sum() / max(trades, 1)) * minutes),
                'cost_per_trade_bps': cost_bps
            }
        except Exception as e:
            self.logger.warning(f"策略指標計算失敗: {e}")
            return default

    def _print_label_statistics(self, labels: pd.Series, params: Dict) -> None:
        if labels.empty:
            return
        counts = labels.value_counts().sort_index()
        total = len(labels)
        self.logger.info(f"📊 標籤分佈 (lag={params.get('lag', '?')}, method={params.get('threshold_method', '?')}):"  )
        for v, name in [(0, '賣出'), (1, '持有'), (2, '買入')]:
            c = counts.get(v, 0)
            self.logger.info(f"   {name}({v}): {c:,} ({c/total*100:.1f}%)")
        if len(labels) > 1:
            changes = int((labels.diff() != 0).sum())
            self.logger.info(f"   標籤變化率: {changes/total:.3f}")
        vals = counts.values
        imbalance = max(vals) / max(min(vals), 1)
        if imbalance > 10:
            self.logger.warning(f"⚠️ 嚴重不平衡: {imbalance:.1f}x")
        elif imbalance > 5:
            self.logger.warning(f"⚠️ 輕度不平衡: {imbalance:.1f}x")

    def calculate_label_quality(self, labels: pd.Series, params: Dict) -> Dict:
        if labels.empty:
            return {'balance_score': 0.0, 'stability_score': 0.0, 'f1_score': 0.0,
                    'precision_macro': 0.0, 'recall_macro': 0.0}
        vc = labels.value_counts(normalize=True)
        actual = np.array([vc.get(0, 0.0), vc.get(1, 0.0), vc.get(2, 0.0)])
        target = np.array([0.20, 0.60, 0.20])  # P0 修正：更符合加密市場真實分布
        kl = np.sum(target * np.log((target + 1e-8) / (actual + 1e-8)))
        balance_score = float(np.exp(-kl))
        changes = int((labels.diff() != 0).sum())
        stability_score = float(max(0.0, 1.0 - changes / max(len(labels), 1)))
        presence = (actual > 0.02).astype(float)
        precision_macro = float(presence.mean())
        recall_macro = float(1.0 - abs(actual - target).mean() / max(target.max(), 1e-8))
        f1 = (2 * precision_macro * recall_macro / max(precision_macro + recall_macro, 1e-8))
        return {
            'balance_score': balance_score, 'stability_score': stability_score,
            'f1_score': float(f1), 'precision_macro': precision_macro,
            'recall_macro': recall_macro, 'distribution': actual.tolist(),
            'total_samples': int(len(labels))
        }

    # ------------------------------------------------------------------
    # Optuna 目標函數（P0 修復：移除殘留 self._last_distribution 引用）
    # ------------------------------------------------------------------

    def objective(self, trial: optuna.Trial) -> float:
        """P0 修復版目標函數：移除 rolling quantile 洩漏，移除 _last_distribution 殘留引用。"""
        lag_min = self.scaled_config.get('label_lag_min', 6)
        lag_max = self.scaled_config.get('label_lag_max', 36)
        buy_q_min = self.scaled_config.get('label_buy_q_min', 0.70)
        buy_q_max = self.scaled_config.get('label_buy_q_max', 0.90)
        sell_q_min = self.scaled_config.get('label_sell_q_min', 0.10)
        sell_q_max = self.scaled_config.get('label_sell_q_max', 0.30)

        params = {
            'lag': trial.suggest_int('lag', lag_min, max(lag_min + 1, lag_max)),
            'threshold_method': trial.suggest_categorical(
                'threshold_method',
                ['quantile', 'fixed', 'adaptive', 'triple_barrier', 'stabilized']
            ),
            'buy_quantile': trial.suggest_float('buy_quantile', buy_q_min, buy_q_max),
            'sell_quantile': trial.suggest_float('sell_quantile', sell_q_min, sell_q_max),
            'profit_threshold': trial.suggest_float('profit_threshold', 0.005, 0.03),
            'loss_threshold': trial.suggest_float('loss_threshold', -0.03, -0.005),
            'vol_multiplier': trial.suggest_float('vol_multiplier', 1.2, 2.0),
            'vol_window': trial.suggest_int('vol_window', 20, 40),
            # Triple-barrier params
            'profit_multiplier': trial.suggest_float('profit_multiplier', 1.6, 2.4),
            'stop_multiplier': trial.suggest_float('stop_multiplier', 0.8, 1.3),
            'max_holding': trial.suggest_int('max_holding', 16, 24),
            'atr_period': trial.suggest_int('atr_period', 12, 18),
            'enable_trailing_stop': True,
            'trailing_activation_ratio': trial.suggest_float('trailing_activation_ratio', 0.4, 0.7),
            'trailing_distance_ratio': trial.suggest_float('trailing_distance_ratio', 0.5, 0.9),
            'trailing_lock_min_profit': trial.suggest_float('trailing_lock_min_profit', 0.2, 0.5),
            'min_risk_reward_ratio': 2.0,  # P0 固定 2:1
            # Scoring weights
            'min_samples': trial.suggest_int('min_samples', 800, 3000),
            'balance_weight': trial.suggest_float('balance_weight', 0.2, 0.5),
            'stability_weight': trial.suggest_float('stability_weight', 0.1, 0.4),
            'sharpe_weight': trial.suggest_float('sharpe_weight', 0.30, 0.45),
            'win_weight': trial.suggest_float('win_weight', 0.20, 0.35),
            'trade_weight': trial.suggest_float('trade_weight', 0.10, 0.25),
            'label_weight': trial.suggest_float('label_weight', 0.15, 0.30),
            'target_trades_per_day': trial.suggest_float('target_trades_per_day', 2.0, 4.0),
            'max_noise_ratio': trial.suggest_float('max_noise_ratio', 0.25, 0.40),
            'transaction_cost_bps': trial.suggest_float(
                'transaction_cost_bps',
                float(self.scaled_config.get('transaction_cost_bps_min', 4.0)),
                float(self.scaled_config.get('transaction_cost_bps_max', 12.0))
            ),
            'target_hold_ratio': trial.suggest_float('target_hold_ratio', 0.50, 0.70),
            'distribution_penalty': trial.suggest_float('distribution_penalty', 0.5, 1.2),
        }

        try:
            ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
            if not ohlcv_file.exists():
                raise FileNotFoundError(f"未找到 OHLCV: {ohlcv_file}")
            df = pd.read_parquet(ohlcv_file)
            price_data = df['close']

            labels = self.generate_labels(price_data, params)
            if len(labels) < params['min_samples']:
                return -999.0

            quality_metrics = self.calculate_label_quality(labels, params)

            lag = params['lag']
            actual_returns = price_data.pct_change(lag).shift(-lag)
            strategy_metrics = self._compute_strategy_metrics(labels, actual_returns, params)
            sharpe_norm = self._normalize_sharpe(strategy_metrics['sharpe'])
            trade_freq_norm = self._normalize_trade_frequency(strategy_metrics['trades_per_day'], params)
            win_rate = strategy_metrics['win_rate']

            # 權重歸一化
            sw = params['sharpe_weight']
            ww = params['win_weight']
            tw = params['trade_weight']
            lw = params['label_weight']
            total_w = sw + ww + tw + lw
            if total_w <= 0:
                sw, ww, tw, lw = 0.35, 0.25, 0.20, 0.20
                total_w = 1.0
            sw /= total_w; ww /= total_w; tw /= total_w; lw /= total_w

            bw = params['balance_weight']
            stw = params['stability_weight']
            f1w = max(0.15, 1.0 - bw - stw)

            vc = labels.value_counts(normalize=True)
            actual_buy = vc.get(2, 0.0)
            actual_sell = vc.get(0, 0.0)
            actual_hold = vc.get(1, 0.0)

            # 極端不平衡直接拒絕
            if actual_buy < 0.05 or actual_sell < 0.05:
                return -999.0

            # 軟性分布懲罰
            target_hold = params['target_hold_ratio']
            hold_dev = abs(actual_hold - target_hold)
            buy_sell_pen = max(0.0, 0.15 - actual_buy) + max(0.0, 0.15 - actual_sell)
            change_rate = float((labels.diff() != 0).sum() / max(len(labels), 1))
            low_change_pen = max(0.0, 0.20 - change_rate)
            distribution_penalty = params['distribution_penalty'] * (hold_dev + buy_sell_pen + 0.5 * low_change_pen)

            label_comp = (quality_metrics['balance_score'] * bw +
                          quality_metrics['stability_score'] * stw +
                          quality_metrics['f1_score'] * f1w)
            kpi_comp = sharpe_norm * sw + win_rate * ww + trade_freq_norm * tw
            final_score = label_comp * lw + kpi_comp - distribution_penalty

            self.logger.info(
                f"📊 分布: 賣={actual_sell:.1%} 持={actual_hold:.1%} 買={actual_buy:.1%} | "
                f"Sharpe={strategy_metrics['sharpe']:.2f} WR={win_rate:.2f} | Score={final_score:.4f}"
            )

            trial.set_user_attr('sharpe', strategy_metrics['sharpe'])
            trial.set_user_attr('win_rate', win_rate)
            trial.set_user_attr('trades_per_day', strategy_metrics['trades_per_day'])
            trial.set_user_attr('actual_hold_ratio', actual_hold)
            trial.set_user_attr('label_component', label_comp)
            trial.set_user_attr('kpi_component', kpi_comp)
            trial.set_user_attr('distribution_penalty', distribution_penalty)
            return final_score

        except Exception as e:
            self.logger.error(f"目標函數失敗: {e}")
            return -999.0

    # ------------------------------------------------------------------
    # optimize() 主流程
    # ------------------------------------------------------------------

    def optimize(self, n_trials: int = 200, timeframes: List[str] = None) -> Dict:
        """執行標籤參數優化（P0修復：移除 _last_distribution 殘留引用）。"""
        if timeframes is None:
            timeframes = [self.timeframe]

        results = {}
        meta_vol = self.scaled_config.get('meta_vol', 0.02)

        for tf in timeframes:
            self.logger.info(f"🚀 Layer1 標籤優化開始 - 時框: {tf}")
            self.timeframe = tf
            storage_url = self.scaled_config.get('optuna_storage')

            study = optuna.create_study(
                direction='maximize',
                study_name=f'label_optimization_layer1_{tf}',
                storage=storage_url,
                load_if_exists=bool(storage_url)
            )
            study.set_user_attr('meta_vol', meta_vol)

            enable_two_stage = self.scaled_config.get('enable_two_stage_search', True)
            if enable_two_stage and n_trials >= 20:
                n_stage1 = int(n_trials * 0.4)
                n_stage2 = n_trials - n_stage1
                self.logger.info(f"🔍 Stage 1: 探索 ({n_stage1} trials)")
                study.optimize(self.objective, n_trials=n_stage1, show_progress_bar=True)

                if study.best_trial:
                    best_lag = study.best_params.get('lag', 12)
                    best_bq = study.best_params.get('buy_quantile', 0.80)
                    best_sq = study.best_params.get('sell_quantile', 0.20)
                    lag_margin = 4
                    q_margin = 0.08
                    self.scaled_config['label_lag_min'] = max(1, best_lag - lag_margin)
                    self.scaled_config['label_lag_max'] = best_lag + lag_margin
                    self.scaled_config['label_buy_q_min'] = max(0.60, best_bq - q_margin)
                    self.scaled_config['label_buy_q_max'] = min(0.95, best_bq + q_margin)
                    self.scaled_config['label_sell_q_min'] = max(0.05, best_sq - q_margin)
                    self.scaled_config['label_sell_q_max'] = min(0.40, best_sq + q_margin)
                    self.logger.info(f"🎯 Stage 2: 精搜 ({n_stage2} trials, lag≈{best_lag})")
                    study.optimize(self.objective, n_trials=n_stage2, show_progress_bar=True)
            else:
                study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            best_score = study.best_value
            self.logger.info(f"✅ Layer1 完成 | 最佳分數={best_score:.4f} | 參數={best_params}")

            # 生成最終標籤
            labeled_data = None
            final_quality = {}
            try:
                # 優先使用 Layer0 清洗數據
                processed_dir = Path(self.data_path) / "processed" / "cleaned" / f"{self.symbol}_{tf}"
                df2 = None
                for c in sorted(processed_dir.glob("cleaned_ohlcv*.parquet") if processed_dir.exists() else [],
                                 key=lambda p: p.stat().st_mtime, reverse=True):
                    try:
                        df2 = read_dataframe(c)
                        break
                    except Exception:
                        continue
                if df2 is None:
                    ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{tf}_ohlcv.parquet"
                    df2 = read_dataframe(ohlcv_file)
                labeled_data = self.apply_labels(df2, best_params)
                final_quality = self.calculate_label_quality(labeled_data['label'], best_params)
            except Exception as e:
                self.logger.warning(f"最終標籤生成失敗: {e}")

            result = {
                'timeframe': tf,
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'final_quality': final_quality,
                'meta_vol': meta_vol,
                'labeled_data': labeled_data,
                'optimization_history': [
                    {'trial': i, 'score': t.value}
                    for i, t in enumerate(study.trials) if t.value is not None
                ]
            }

            lag_min = self.scaled_config.get('label_lag_min', 1)
            lag_max_v = self.scaled_config.get('label_lag_max', 1000)
            result['best_params']['lag'] = int(max(lag_min, min(
                result['best_params'].get('lag', lag_min), lag_max_v)))

            json_safe = {k: v for k, v in result.items() if k != 'labeled_data'}
            for fn in [self.config_path / "label_params.json",
                       self.config_path / f"label_params_{tf}.json"]:
                try:
                    atomic_write_json(json_safe, fn)
                except Exception:
                    with open(fn, 'w', encoding='utf-8') as f:
                        json.dump(json_safe, f, indent=2, ensure_ascii=False)

            # 物化標籤
            try:
                if labeled_data is not None and not labeled_data.empty:
                    out_dir = Path(self.data_path) / "processed" / "labels" / f"{self.symbol}_{tf}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target = out_dir / f"labels_{self.symbol}_{tf}.parquet"
                    label_file, _ = write_dataframe(labeled_data, target)
                    result['materialized_path'] = str(label_file)
                    self.logger.info(f"✅ Layer1 標籤物化: {label_file}")
            except Exception as e:
                self.logger.warning(f"標籤物化失敗: {e}")

            results[tf] = result

        return results[self.timeframe] if len(timeframes) == 1 else results

    def apply_labels(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """生成標籤並附加至 OHLCV 資料。"""
        if 'close' not in data.columns:
            raise ValueError("資料必須包含 close 欄位")
        labels = self.generate_labels(data['close'], params)
        if labels.empty:
            raise ValueError("標籤生成失敗，序列為空")
        result = data.loc[labels.index].copy()
        if result.empty:
            raise ValueError("對齊後資料為空")
        result['label'] = labels

        try:
            lag = max(1, int(params.get('lag', 1)))
            aligned_close = data.loc[result.index, 'close']
            actual_returns = aligned_close.pct_change(lag).shift(-lag)
            metrics = self._compute_strategy_metrics(labels, actual_returns, params)
            result['forward_return'] = actual_returns
            result['label_position'] = labels.map({2: 1, 1: 0, 0: -1}).astype(int)
            result.attrs['layer1_metrics'] = metrics
        except Exception as e:
            self.logger.warning(f"附加 KPI 欄位失敗: {e}")

        return result

    def apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return self.apply_labels(data, params)


def main():
    optimizer = LabelOptimizer(data_path='../data', config_path='../configs')
    result = optimizer.optimize(n_trials=200)
    print(f"標籤優化完成: {result['best_score']:.4f}")


if __name__ == "__main__":
    main()
