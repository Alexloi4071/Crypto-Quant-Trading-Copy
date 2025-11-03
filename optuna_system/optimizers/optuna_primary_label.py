# -*- coding: utf-8 -*-
"""
Primary Label Optimizer (Layer 1A)
方向預測器：只預測買入 vs 賣出（二分類）

Meta-Labeling 架構的第一層：Primary Model
- 目標：預測市場方向（買入 vs 賣出）
- 輸出：1 (買入) / -1 (賣出)
- 特點：無「持有」類別，目標 50/50 平衡

參考文獻：
- Marcos López de Prado (2018), "Advances in Financial Machine Learning", Ch.3
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
import pandas as pd

from optuna_system.utils.io_utils import write_dataframe, read_dataframe, atomic_write_json

warnings.filterwarnings('ignore')


class PrimaryLabelOptimizer:
    """
    Layer 1A: Primary Model - 方向預測器
    
    目標：預測市場方向（買入 vs 賣出），二分類
    輸出：1 (買入) / -1 (賣出)
    
    特點：
    - 無「持有」類別
    - 目標 50/50 平衡
    - 只關注方向準確性
    """
    
    def __init__(
        self,
        data_path: str,
        config_path: str = "configs/",
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        scaled_config: Dict = None
    ):
        """初始化 Primary Model 優化器"""
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe
        self.scaled_config = scaled_config or {}
        self.logger = logging.getLogger(__name__)
        
        # 載入價格數據
        self.price_data = None
        self._load_price_data()
    
    def _load_price_data(self):
        """載入清洗後的 OHLCV 數據"""
        try:
            # 優先從 processed/cleaned 載入
            processed_dir = self.data_path / "processed" / "cleaned" / f"{self.symbol}_{self.timeframe}"
            if processed_dir.exists():
                candidates = list(processed_dir.glob("cleaned_ohlcv*.parquet"))
                if candidates:
                    self.price_data = read_dataframe(candidates[0])
                    self.logger.info(f"✅ 載入清洗數據: {candidates[0].name}")
                    return
            
            # 回退到原始數據
            raw_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
            if raw_file.exists():
                self.price_data = read_dataframe(raw_file)
                self.logger.info(f"✅ 載入原始數據: {raw_file.name}")
            else:
                raise FileNotFoundError(f"找不到價格數據: {raw_file}")
        
        except Exception as e:
            self.logger.error(f"❌ 載入價格數據失敗: {e}")
            raise
    
    def _timeframe_to_minutes(self, timeframe: Optional[str] = None) -> float:
        """轉換時間框為分鐘數"""
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
        return 15.0  # 默認15分鐘
    
    def calculate_trading_metrics(
        self,
        signals: pd.Series,
        price_data: pd.Series,
        params: Dict
    ) -> Dict:
        """
        🎯 計算真正的交易質量指標（優化目標重構 - Phase 1）
        
        這是新增的核心函數，用於替代舊的Accuracy/Sharpe優化。
        
        計算內容：
        - 整體胜率、盈利因子、盈亏比
        - 做多/做空分別的胜率、盈利因子
        - 平均盈利/亏损
        - 最大连续盈利/亏损
        
        學術依據：
        - Van Tharp (2008), "Trade Your Way to Financial Freedom"
        - Connors & Alvarez (2009), "Short Term Trading Strategies That Work"
        
        Args:
            signals: Primary信號序列（1=買入, -1=賣出）
            price_data: 價格序列
            params: Triple Barrier參數
        
        Returns:
            Dict: 完整的交易質量指標
        """
        try:
            # 提取參數
            lag = params.get('lag', 12)
            atr_period = params.get('atr_period', 14)
            profit_multiplier = params.get('profit_multiplier', 2.0)
            stop_multiplier = params.get('stop_multiplier', 1.5)
            max_holding = params.get('max_holding', 20)
            transaction_cost_bps = params.get('transaction_cost_bps', 10.0)
            enable_trailing = params.get('enable_trailing_stop', True)
            trail_activation = params.get('trailing_activation_ratio', 0.5)
            trail_distance = params.get('trailing_distance_ratio', 0.7)
            trail_lock_min = params.get('trailing_lock_min_profit', 0.3)
            
            # 計算ATR
            atr = self.calculate_atr(
                self.price_data['high'],
                self.price_data['low'],
                self.price_data['close'],
                atr_period
            )
            atr = atr.reindex(price_data.index).fillna(method='ffill')
            
            # 處理NaN
            if atr.isna().any():
                first_valid_idx = atr.first_valid_index()
                if first_valid_idx is not None:
                    atr = atr.fillna(atr[first_valid_idx])
                else:
                    atr = atr.fillna(price_data.std() * 0.02)
            
            # 交易成本（雙向）
            round_trip_cost = transaction_cost_bps / 10000.0
            
            # 轉換為numpy數組（性能優化）
            price_values = price_data.values
            atr_values = atr.values
            signal_values = signals.values
            
            # 交易記錄
            long_trades = []   # 做多交易
            short_trades = []  # 做空交易
            
            # 逐個入場點模擬交易
            for i in range(len(signals) - max_holding):
                signal = signal_values[i]
                if signal == 0:
                    continue
                
                entry_price = price_values[i]
                current_atr = atr_values[i]
                
                if np.isnan(current_atr) or current_atr <= 0:
                    continue
                
                # 做多交易
                if signal == 1:
                    # 目標價格
                    profit_target = entry_price * (1 + profit_multiplier * current_atr / entry_price)
                    initial_stop = entry_price * (1 - stop_multiplier * current_atr / entry_price)
                    
                    # 考慮交易成本
                    profit_target *= (1 + round_trip_cost)
                    initial_stop *= (1 - round_trip_cost)
                    
                    # 移動止損變量
                    current_stop = initial_stop
                    highest_price = entry_price
                    trailing_activated = False
                    
                    # 未來價格窗口
                    future_window_end = min(i + max_holding + 1, len(price_data))
                    
                    # 逐K線檢查
                    pnl = 0
                    for j in range(i + 1, future_window_end):
                        future_price = price_values[j]
                        
                        # 移動止損邏輯
                        if enable_trailing:
                            if future_price > highest_price:
                                highest_price = future_price
                            
                            profit_progress = (future_price - entry_price) / (profit_target - entry_price)
                            if profit_progress >= trail_activation and not trailing_activated:
                                trailing_activated = True
                            
                            if trailing_activated:
                                new_trail_stop = highest_price * (1 - trail_distance * current_atr / highest_price)
                                min_lock_stop = entry_price * (1 + trail_lock_min * current_atr / entry_price)
                                new_trail_stop = max(new_trail_stop, min_lock_stop)
                                if new_trail_stop > current_stop:
                                    current_stop = new_trail_stop
                        
                        # 觸發止盈
                        if future_price >= profit_target:
                            pnl = (profit_target - entry_price) / entry_price - round_trip_cost
                            long_trades.append({'pnl': pnl, 'type': 'win'})
                            break
                        # 觸發止損
                        elif future_price <= current_stop:
                            pnl = (current_stop - entry_price) / entry_price - round_trip_cost
                            long_trades.append({'pnl': pnl, 'type': 'loss' if pnl < 0 else 'win'})
                            break
                    else:
                        # 超時退出
                        exit_price = price_values[future_window_end - 1]
                        pnl = (exit_price - entry_price) / entry_price - round_trip_cost
                        long_trades.append({'pnl': pnl, 'type': 'win' if pnl > 0 else 'loss'})
                
                # 做空交易
                elif signal == -1:
                    # 目標價格（做空）
                    profit_target = entry_price * (1 - profit_multiplier * current_atr / entry_price)
                    initial_stop = entry_price * (1 + stop_multiplier * current_atr / entry_price)
                    
                    # 考慮交易成本
                    profit_target *= (1 - round_trip_cost)
                    initial_stop *= (1 + round_trip_cost)
                    
                    # 移動止損變量
                    current_stop = initial_stop
                    lowest_price = entry_price
                    trailing_activated = False
                    
                    # 未來價格窗口
                    future_window_end = min(i + max_holding + 1, len(price_data))
                    
                    # 逐K線檢查
                    pnl = 0
                    for j in range(i + 1, future_window_end):
                        future_price = price_values[j]
                        
                        # 移動止損邏輯（做空）
                        if enable_trailing:
                            if future_price < lowest_price:
                                lowest_price = future_price
                            
                            profit_progress = (entry_price - future_price) / (entry_price - profit_target)
                            if profit_progress >= trail_activation and not trailing_activated:
                                trailing_activated = True
                            
                            if trailing_activated:
                                new_trail_stop = lowest_price * (1 + trail_distance * current_atr / lowest_price)
                                min_lock_stop = entry_price * (1 - trail_lock_min * current_atr / entry_price)
                                new_trail_stop = min(new_trail_stop, min_lock_stop)
                                if new_trail_stop < current_stop:
                                    current_stop = new_trail_stop
                        
                        # 觸發止盈（做空）
                        if future_price <= profit_target:
                            pnl = (entry_price - profit_target) / entry_price - round_trip_cost
                            short_trades.append({'pnl': pnl, 'type': 'win'})
                            break
                        # 觸發止損（做空）
                        elif future_price >= current_stop:
                            pnl = (entry_price - current_stop) / entry_price - round_trip_cost
                            short_trades.append({'pnl': pnl, 'type': 'loss' if pnl < 0 else 'win'})
                            break
                    else:
                        # 超時退出
                        exit_price = price_values[future_window_end - 1]
                        pnl = (entry_price - exit_price) / entry_price - round_trip_cost
                        short_trades.append({'pnl': pnl, 'type': 'win' if pnl > 0 else 'loss'})
            
            # 計算做多指標
            long_wins = [t['pnl'] for t in long_trades if t['type'] == 'win']
            long_losses = [t['pnl'] for t in long_trades if t['type'] == 'loss']
            long_win_rate = len(long_wins) / len(long_trades) if long_trades else 0.0
            long_avg_win = np.mean(long_wins) if long_wins else 0.0
            long_avg_loss = abs(np.mean(long_losses)) if long_losses else 0.0
            long_total_profit = sum(long_wins)
            long_total_loss = abs(sum(long_losses))
            long_profit_factor = (long_total_profit / long_total_loss) if long_total_loss > 0 else float('inf')
            
            # 計算做空指標
            short_wins = [t['pnl'] for t in short_trades if t['type'] == 'win']
            short_losses = [t['pnl'] for t in short_trades if t['type'] == 'loss']
            short_win_rate = len(short_wins) / len(short_trades) if short_trades else 0.0
            short_avg_win = np.mean(short_wins) if short_wins else 0.0
            short_avg_loss = abs(np.mean(short_losses)) if short_losses else 0.0
            short_total_profit = sum(short_wins)
            short_total_loss = abs(sum(short_losses))
            short_profit_factor = (short_total_profit / short_total_loss) if short_total_loss > 0 else float('inf')
            
            # 計算整體指標
            all_trades = long_trades + short_trades
            all_wins = long_wins + short_wins
            all_losses = long_losses + short_losses
            
            overall_win_rate = len(all_wins) / len(all_trades) if all_trades else 0.0
            overall_avg_win = np.mean(all_wins) if all_wins else 0.0
            overall_avg_loss = abs(np.mean(all_losses)) if all_losses else 0.0
            overall_total_profit = sum(all_wins)
            overall_total_loss = abs(sum(all_losses))
            overall_profit_factor = (overall_total_profit / overall_total_loss) if overall_total_loss > 0 else float('inf')
            risk_reward_ratio = (overall_avg_win / overall_avg_loss) if overall_avg_loss > 0 else 0.0
            
            # 最大連續盈利/虧損
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_win_streak = 0
            current_loss_streak = 0
            
            for trade in all_trades:
                if trade['type'] == 'win':
                    current_win_streak += 1
                    current_loss_streak = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
                else:
                    current_loss_streak += 1
                    current_win_streak = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
            
            # 計算Sharpe Ratio（基於交易收益序列）
            all_pnls = [t['pnl'] for t in all_trades]
            if len(all_pnls) > 1:
                pnl_std = np.std(all_pnls)
                pnl_mean = np.mean(all_pnls)
                sharpe = (pnl_mean / pnl_std * np.sqrt(252)) if pnl_std > 0 else 0.0
            else:
                sharpe = 0.0
            
            # 返回完整指標
            return {
                # 做多指標
                'long_win_rate': float(long_win_rate),
                'long_avg_win': float(long_avg_win),
                'long_avg_loss': float(long_avg_loss),
                'long_profit_factor': float(min(long_profit_factor, 10.0)),  # 上限10
                'long_total_trades': len(long_trades),
                'long_trades': len(long_trades),  # 向后兼容
                
                # 做空指標
                'short_win_rate': float(short_win_rate),
                'short_avg_win': float(short_avg_win),
                'short_avg_loss': float(short_avg_loss),
                'short_profit_factor': float(min(short_profit_factor, 10.0)),  # 上限10
                'short_total_trades': len(short_trades),
                'short_trades': len(short_trades),  # 向后兼容
                
                # 整體指標
                'overall_win_rate': float(overall_win_rate),
                'overall_profit_factor': float(min(overall_profit_factor, 10.0)),  # 上限10
                'risk_reward_ratio': float(risk_reward_ratio),
                'avg_win': float(overall_avg_win),
                'avg_loss': float(overall_avg_loss),
                'total_trades': len(all_trades),
                'total_profit': float(overall_total_profit),
                'total_loss': float(overall_total_loss),
                'net_profit': float(overall_total_profit - overall_total_loss),
                'total_pnl': float(overall_total_profit - overall_total_loss),  # 向后兼容
                'sharpe': float(min(max(sharpe, -10), 10)),  # Sharpe限制在[-10, 10]
                
                # 連續指標
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
            }
        
        except Exception as e:
            self.logger.error(f"❌ 交易質量指標計算失敗: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # 返回默認值
            return {
                'long_win_rate': 0.0, 'long_avg_win': 0.0, 'long_avg_loss': 0.0,
                'long_profit_factor': 0.0, 'long_total_trades': 0, 'long_trades': 0,
                'short_win_rate': 0.0, 'short_avg_win': 0.0, 'short_avg_loss': 0.0,
                'short_profit_factor': 0.0, 'short_total_trades': 0, 'short_trades': 0,
                'overall_win_rate': 0.0, 'overall_profit_factor': 0.0,
                'risk_reward_ratio': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'total_trades': 0, 'total_profit': 0.0, 'total_loss': 0.0,
                'net_profit': 0.0, 'total_pnl': 0.0, 'sharpe': 0.0,
                'max_consecutive_wins': 0, 'max_consecutive_losses': 0
            }
    
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
        """
        🚀 Triple-Barrier 標籤生成（從 Legacy 複製）
        
        三重障礙標籤生成：
        - 止盈障礙：profit_multiplier × ATR
        - 止損障礙：stop_multiplier × ATR
        - 時間障礙：max_holding 期
        
        Returns:
            pd.Series: 0 (賣出) / 1 (持有) / 2 (買入)
        """
        try:
            # 提取參數
            lag = params.get('lag', 12)
            atr_period = params.get('atr_period', 14)
            profit_multiplier = params.get('profit_multiplier', 2.0)
            stop_multiplier = params.get('stop_multiplier', 1.5)
            max_holding = params.get('max_holding', 20)
            transaction_cost_bps = params.get('transaction_cost_bps', 10.0)
            enable_trailing = params.get('enable_trailing_stop', True)
            trail_activation = params.get('trailing_activation_ratio', 0.5)
            trail_distance = params.get('trailing_distance_ratio', 0.7)
            trail_lock_min = params.get('trailing_lock_min_profit', 0.3)
            
            # 交易成本（雙向）
            round_trip_cost = transaction_cost_bps / 10000.0
            
            # 🔒 風險回報比約束
            min_rr = params.get('min_risk_reward_ratio', 1.3)
            if profit_multiplier / stop_multiplier < min_rr:
                adjusted_profit = stop_multiplier * min_rr
                self.logger.debug(
                    f"🔒 R:R約束: {profit_multiplier/stop_multiplier:.2f}:1 → "
                    f"{adjusted_profit/stop_multiplier:.2f}:1"
                )
                profit_multiplier = adjusted_profit
            
            # 計算ATR
            try:
                atr = self.calculate_atr(
                    self.price_data['high'],
                    self.price_data['low'],
                    self.price_data['close'],
                    atr_period
                )
                atr = atr.reindex(price_data.index).fillna(method='ffill')
                
                # 處理 NaN
                if atr.isna().any():
                    first_valid_idx = atr.first_valid_index()
                    if first_valid_idx is not None:
                        atr = atr.fillna(atr[first_valid_idx])
                    else:
                        atr = atr.fillna(price_data.std() * 0.02)
            except Exception as e:
                self.logger.warning(f"⚠️ ATR計算失敗: {e}，使用簡化估算")
                returns = price_data.pct_change().abs()
                atr = returns.rolling(atr_period).mean() * price_data
            
            # 初始化標籤
            labels = pd.Series(1, index=price_data.index, dtype=int)  # 默認持有
            
            # 統計變量
            stats = {
                'total_signals': 0,
                'profit_hits': 0,
                'initial_stop_hits': 0,
                'trailing_stop_hits': 0,
                'break_even_stops': 0,
                'profit_locks': 0,
                'timeout_holds': 0
            }
            
            # 🚀 性能优化：预先转换为 numpy 数组（避免逐个 iloc 访问）
            price_values = price_data.values
            atr_values = atr.values
            
            # ========== 主循環：逐個入場點模擬 ==========
            for i in range(len(price_data) - max_holding):
                entry_price = price_values[i]
                current_atr = atr_values[i]
                
                if np.isnan(current_atr) or current_atr <= 0:
                    continue
                
                stats['total_signals'] += 1
                
                # 初始止盈止損價格
                profit_target = entry_price + current_atr * profit_multiplier
                initial_stop = entry_price - current_atr * stop_multiplier
                
                # 考慮交易成本
                profit_target *= (1 + round_trip_cost)
                initial_stop *= (1 - round_trip_cost)
                
                # 移動止損變量
                current_stop = initial_stop
                highest_price = entry_price
                trailing_activated = False
                locked_profit = False
                
                # 定義未來價格窗口
                future_window_end = min(i + max_holding + 1, len(price_data))
                
                # ========== 逐K線檢查觸發條件 ==========
                for j in range(i + 1, future_window_end):
                    future_price = price_values[j]
                    current_profit = future_price - entry_price
                    current_profit_atr = current_profit / current_atr
                    
                    # 🚀 移動止損邏輯
                    if enable_trailing:
                        # 更新最高價
                        if future_price > highest_price:
                            highest_price = future_price
                        
                        # 計算盈利進度（相對於目標）
                        profit_progress = (future_price - entry_price) / (profit_target - entry_price)
                        
                        # 啟動條件：盈利達到 trail_activation 比例
                        if profit_progress >= trail_activation and not trailing_activated:
                            trailing_activated = True
                        
                        # 移動止損更新
                        if trailing_activated:
                            # 基本移動止損：距最高點 trail_distance×ATR
                            new_trail_stop = highest_price - trail_distance * current_atr
                            
                            # 確保至少鎖定 trail_lock_min×ATR 利潤
                            min_lock_stop = entry_price + trail_lock_min * current_atr
                            new_trail_stop = max(new_trail_stop, min_lock_stop)
                            
                            # 止損只能上移，不能下移
                            if new_trail_stop > current_stop:
                                # 檢查是否達到保本或鎖利狀態
                                if new_trail_stop >= entry_price and not locked_profit:
                                    locked_profit = True
                                    stats['profit_locks'] += 1
                                
                                current_stop = new_trail_stop
                    
                    # ========== 檢查觸發條件 ==========
                    # 1. 觸發止盈
                    if future_price >= profit_target:
                        labels.iloc[i] = 2  # 買入信號
                        stats['profit_hits'] += 1
                        break
                    
                    # 2. 觸發止損
                    elif future_price <= current_stop:
                        labels.iloc[i] = 0  # 賣出信號
                        
                        # 區分不同類型的止損
                        if trailing_activated:
                            if current_stop >= entry_price:
                                stats['break_even_stops'] += 1
                            else:
                                stats['trailing_stop_hits'] += 1
                        else:
                            stats['initial_stop_hits'] += 1
                        break
                
                else:
                    # 未觸發任何障礙，持有到期
                    stats['timeout_holds'] += 1
            
            # 移除未來數據洩露
            if lag > 0:
                labels = labels[:-lag]
            
            # 統計報告（簡化版）
            if stats['total_signals'] > 0:
                total = stats['total_signals']
                self.logger.info(f"📊 Triple-Barrier 統計: 總信號={total}, "
                               f"止盈={stats['profit_hits']}, "
                               f"止損={stats['initial_stop_hits']}, "
                               f"持有到期={stats['timeout_holds']}")
            
            return labels.dropna()
            
        except Exception as e:
            self.logger.error(f"❌ Triple-Barrier 生成失敗: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.Series([], dtype=int)
    
    def generate_primary_signals(
        self,
        price_data: pd.Series,
        params: Dict
    ) -> pd.Series:
        """
        生成 Primary 信號（二分類）
        
        🔧 P0修復說明：
        - 保留完整的Triple Barrier功能
        - Triple Barrier用於生成"訓練標籤"（這是正確的）
        - 將三分類標籤轉換為二分類信號
        - 注意：這裡生成的是"用於訓練的標籤"，不是"預測信號"
        
        正確的Meta-Labeling流程（López de Prado 2018）：
        1. 使用Triple Barrier生成訓練標籤（可以使用未來信息）✅
        2. 基於歷史特徵訓練模型（在objective函數中實現）
        3. 模型使用歷史特徵預測（無未來信息）
        
        當前函數的作用：
        - 生成訓練標籤（Triple Barrier）
        - 這些標籤會被用於訓練和評估
        
        Returns:
            pd.Series: 1=買入, -1=賣出（訓練標籤）
        """
        # 🔧 步驟 1：生成 Triple Barrier 標籤（保留完整功能）
        labels_3class = self.generate_triple_barrier_labels(price_data, params)
        
        if labels_3class.empty:
            self.logger.warning("⚠️ Triple Barrier 返回空標籤")
            return pd.Series([], dtype=int)
        
        # 🔧 步驟 2：計算未來收益（用於將「持有」分配方向）
        lag = params.get('lag', 12)
        future_prices = price_data.shift(-lag)
        future_returns = (future_prices - price_data) / price_data
        
        # 🔧 步驟 3：三分類 → 二分類轉換（保留原有邏輯）
        binary_signals = pd.Series(0, index=labels_3class.index, dtype=int)
        
        # 原「買入」(2) → 1
        binary_signals[labels_3class == 2] = 1
        
        # 原「賣出」(0) → -1
        binary_signals[labels_3class == 0] = -1
        
        # 原「持有」(1) → 根據未來收益分配
        hold_mask = (labels_3class == 1)
        binary_signals[hold_mask & (future_returns > 0)] = 1   # 未來上漲 → 買入
        binary_signals[hold_mask & (future_returns <= 0)] = -1 # 未來下跌 → 賣出
        
        # 🔧 步驟 4：統計信號分佈
        buy_count = (binary_signals == 1).sum()
        sell_count = (binary_signals == -1).sum()
        total = len(binary_signals)
        buy_ratio = buy_count / total if total > 0 else 0
        sell_ratio = sell_count / total if total > 0 else 0
        
        self.logger.info(f"📊 Primary 信號分佈: 買入={buy_ratio:.1%}, 賣出={sell_ratio:.1%}")
        
        return binary_signals
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        🎯 Optuna 目標函數（優化目標完整重構 - Phase 1）
        
        ✅ 新優化目標（交易質量主導）：
        1. Win Rate（胜率）20%
        2. Profit Factor（盈利因子）30%
        3. Risk:Reward Ratio（盈亏比）10%
        4. Long/Short平衡（各10%）
        5. 平均盈亏比（10%）
        6. 交易頻率（10%）
        
        ❌ 已刪除的舊目標：
        - Accuracy（分類准確率，不代表盈利能力）
        - Sharpe Ratio（整體收益，不反映單筆質量）
        
        學術依據：
        - Van Tharp (2008), "Trade Your Way to Financial Freedom"
        - Connors & Alvarez (2009), "Short Term Trading Strategies That Work"
        - López de Prado (2018), "Advances in Financial Machine Learning", Ch.3
        """
        # 🔧 參數搜索空間（優化目標重構 - 擴大盈虧比範圍）
        params = {
            'lag': trial.suggest_int('lag', 6, 24),
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            
            # ✅ 修改：提高profit_multiplier（從1.5-3.0提高到2.0-4.0）
            'profit_multiplier': trial.suggest_float('profit_multiplier', 2.0, 4.0),
            
            # ✅ 修改：降低stop_multiplier（從1.0-2.0降低到0.8-1.5）
            'stop_multiplier': trial.suggest_float('stop_multiplier', 0.8, 1.5),
            
            # ✅ 修改：擴大max_holding（從10-30擴大到15-40）
            'max_holding': trial.suggest_int('max_holding', 15, 40),
            
            # ✅ 修改：強制啟用移動止損（不再是可選參數）
            'enable_trailing_stop': True,  # 從suggest_categorical改為固定True
            
            # ✅ 修改：優化移動止損參數範圍
            'trailing_activation_ratio': trial.suggest_float('trailing_activation_ratio', 0.4, 0.7),  # 從0.3-0.6改為0.4-0.7
            'trailing_distance_ratio': trial.suggest_float('trailing_distance_ratio', 0.6, 0.9),  # 從0.5-0.9改為0.6-0.9
            'trailing_lock_min_profit': trial.suggest_float('trailing_lock_min_profit', 0.3, 0.6),  # 從0.2-0.5改為0.3-0.6
            
            # ✅ 修改：降低交易成本（從5.0-15.0降低到4.0-8.0，更接近實際）
            'transaction_cost_bps': trial.suggest_float('transaction_cost_bps', 4.0, 8.0),
        }
        
        # ✅ 新增：盈亏比硬約束（確保 Risk:Reward >= 2:1）
        risk_reward = params['profit_multiplier'] / params['stop_multiplier']
        if risk_reward < 2.0:
            # 調整profit_multiplier確保盈亏比>=2:1
            params['profit_multiplier'] = max(params['profit_multiplier'], params['stop_multiplier'] * 2.0)
            risk_reward = params['profit_multiplier'] / params['stop_multiplier']
        
        # 生成 Primary 信號
        try:
            signals = self.generate_primary_signals(self.price_data['close'], params)
        except Exception as e:
            self.logger.warning(f"⚠️ 信號生成失敗: {e}")
            return -999.0
        
        if len(signals) < 100:
            return -999.0
        
        # 🎯 新增：計算真正的交易質量指標（替代Accuracy/Sharpe）
        metrics = self.calculate_trading_metrics(signals, self.price_data['close'], params)
        
        # ✅ 硬約束檢查（確保基本交易質量）
        if metrics['total_trades'] < 50:
            # 交易數太少，無法評估
            return -999.0
        
        if metrics['overall_profit_factor'] < 1.3:
            # 盈利因子<1.3，長期難以盈利
            return -999.0
        
        if metrics['overall_win_rate'] < 0.45:
            # 胜率<45%，配合盈亏比2:1也難以盈利
            return -999.0
        
        # ✅ 新目標函數（交易質量主導，100%權重）
        score = (
            # 盈利能力（60%）- 核心指標
            metrics['overall_profit_factor'] / 3.0 * 0.30 +  # 盈利因子（歸一化），30%
            metrics['overall_win_rate'] * 0.20 +              # 整體胜率，20%
            min(metrics['risk_reward_ratio'] / 3.0, 1.0) * 0.10 +  # 盈亏比（歸一化），10%
            
            # 做多/做空平衡（20%）- 確保雙向都有效
            min(metrics['long_win_rate'], metrics['short_win_rate']) * 0.10 +  # 較弱側胜率，10%
            min(metrics['long_profit_factor'], metrics['short_profit_factor']) / 3.0 * 0.10 +  # 較弱側盈利因子，10%
            
            # 輔助指標（20%）- 風險控制和頻率
            min(metrics['avg_win'] / max(metrics['avg_loss'], 0.001) / 3.0, 1.0) * 0.10 +  # 平均盈亏比，10%
            min(metrics['total_trades'] / 100.0, 1.0) * 0.10  # 交易頻率（歸一化），10%
        )
        
        # ✅ 記錄所有關鍵指標（用於分析）
        # 整體指標
        trial.set_user_attr("overall_win_rate", metrics['overall_win_rate'])
        trial.set_user_attr("profit_factor", metrics['overall_profit_factor'])
        trial.set_user_attr("risk_reward_ratio", metrics['risk_reward_ratio'])
        trial.set_user_attr("avg_win", metrics['avg_win'])
        trial.set_user_attr("avg_loss", metrics['avg_loss'])
        trial.set_user_attr("total_trades", metrics['total_trades'])
        trial.set_user_attr("net_profit", metrics['net_profit'])
        
        # 做多指標
        trial.set_user_attr("long_win_rate", metrics['long_win_rate'])
        trial.set_user_attr("long_profit_factor", metrics['long_profit_factor'])
        trial.set_user_attr("long_total_trades", metrics['long_total_trades'])
        
        # 做空指標
        trial.set_user_attr("short_win_rate", metrics['short_win_rate'])
        trial.set_user_attr("short_profit_factor", metrics['short_profit_factor'])
        trial.set_user_attr("short_total_trades", metrics['short_total_trades'])
        
        # 連續指標
        trial.set_user_attr("max_consecutive_wins", metrics['max_consecutive_wins'])
        trial.set_user_attr("max_consecutive_losses", metrics['max_consecutive_losses'])
        
        # 記錄參數約束結果
        trial.set_user_attr("risk_reward_constraint_met", risk_reward >= 2.0)
        
        return score
    
    def optimize(self, n_trials: int = 100) -> Dict:
        """執行 Primary Model 優化"""
        self.logger.info("🚀 Primary Model (方向預測器) 優化開始...")
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'primary_label_{self.timeframe}'
        )
        
        # 🛡️ 添加超时保护：每个 trial 最多 60 秒，总超时 n_trials * 60 秒
        try:
            study.optimize(
                self.objective, 
                n_trials=n_trials,
                timeout=n_trials * 60,  # 总超时
                catch=(Exception,)  # 捕获单个 trial 的异常但继续优化
            )
        except KeyboardInterrupt:
            self.logger.warning("⚠️ 优化被用户中断")
            if len(study.trials) == 0:
                raise ValueError("没有完成任何 trial，无法继续")
        except Exception as e:
            self.logger.error(f"❌ 优化过程出错: {e}")
            if len(study.trials) == 0:
                raise
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.logger.info(f"✅ Primary 優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"📋 最優參數: {best_params}")
        
        # 獲取最佳 trial 的額外信息
        best_trial = study.best_trial
        accuracy = best_trial.user_attrs.get('accuracy', 0)
        sharpe = best_trial.user_attrs.get('sharpe', 0)
        buy_ratio = best_trial.user_attrs.get('buy_ratio', 0)
        
        self.logger.info(f"📊 最佳性能: accuracy={accuracy:.3f}, sharpe={sharpe:.2f}, buy_ratio={buy_ratio:.2f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials,
            'study': study,
            'accuracy': accuracy,
            'sharpe': sharpe,
            'buy_ratio': buy_ratio
        }
    
    def apply_labels(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """應用 Primary 信號到數據"""
        signals = self.generate_primary_signals(data['close'], params)
        
        result = data.loc[signals.index].copy()
        result['primary_signal'] = signals  # 1/-1
        
        return result
    
    def apply_transform(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """統一物化接口（Coordinator 調用）"""
        return self.apply_labels(data, params)


if __name__ == "__main__":
    # 獨立測試腳本
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    print("🚀 Primary Label Optimizer 測試")
    
    optimizer = PrimaryLabelOptimizer(
        data_path='../../data',
        config_path='../../configs'
    )
    
    print(f"✅ 數據載入成功: {len(optimizer.price_data)} 行")
    print("🔬 開始優化測試（10 trials）...")
    
    result = optimizer.optimize(n_trials=10)
    print(f"\n✅ 優化完成!")
    print(f"   最佳得分: {result['best_score']:.4f}")
    print(f"   方向準確率: {result['accuracy']:.3f}")
    print(f"   Sharpe: {result['sharpe']:.2f}")
    print(f"   買入比例: {result['buy_ratio']:.2f}")
