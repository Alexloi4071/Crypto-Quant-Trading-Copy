# -*- coding: utf-8 -*-
"""
交叉驗證與風控參數優化器 (第4層)
優化CV參數、風險閾值、回測策略參數
"""
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')
import gc


class CVRiskOptimizer:
    """交叉驗證與風控參數優化器 - 第4層優化"""

    def __init__(self, data_path: str, config_path: str = "configs/",
                 symbol: str = "BTCUSDT", timeframe: str = "15m"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)
        self.symbol = symbol
        self.timeframe = timeframe

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

        self.purge_periods = 192  # 48小時15m
        self.embargo_periods = 64  # 16小時
        self.ts_split = TimeSeriesSplit(n_splits=5)
        self.slippage = 0.001  # 0.1%
        self.fees = 0.0005  # 0.05%

    def simulate_backtest_with_risk_params(self, returns: pd.Series, params: Dict) -> Dict:
        """模擬回測並應用風控參數"""
        try:
            # 回測參數
            stop_loss_pct = params['stop_loss_pct']
            take_profit_pct = params['take_profit_pct']
            max_drawdown_limit = params['max_drawdown_limit']
            position_size_pct = params['position_size_pct']
            min_holding_periods = params.get('min_holding_periods', 1)

            # 模擬交易信號（基於簡單移動平均）
            prices = (1 + returns).cumprod() * 100
            sma_short = prices.rolling(window=params.get('sma_short', 10)).mean()
            sma_long = prices.rolling(window=params.get('sma_long', 30)).mean()

            signals = pd.Series(0, index=prices.index)
            signals[sma_short > sma_long] = 1  # 買入信號
            signals[sma_short < sma_long] = -1  # 賣出信號

            # 模擬交易執行
            portfolio_value = 10000.0
            positions = []
            trades = []
            current_position = 0
            entry_price = 0
            hold_count = 0

            portfolio_values = [portfolio_value]

            for i in range(1, len(signals)):
                current_price = prices.iloc[i]
                signal = signals.iloc[i]

                # 持有期限制
                if hold_count > 0:
                    hold_count -= 1

                # 如果有持倉，檢查止損止盈
                if current_position != 0:
                    if current_position > 0:  # 多頭
                        pnl_pct = (current_price - entry_price) / entry_price
                        if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                            # 平倉
                            portfolio_value *= (1 + current_position * pnl_pct)
                            trades.append({
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'pnl_pct': pnl_pct,
                                'reason': 'stop_loss' if pnl_pct <= -stop_loss_pct else 'take_profit'
                            })
                            current_position = 0
                            hold_count = min_holding_periods

                    elif current_position < 0:  # 空頭
                        pnl_pct = (entry_price - current_price) / entry_price
                        if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                            # 平倉
                            portfolio_value *= (1 + abs(current_position) * pnl_pct)
                            trades.append({
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'pnl_pct': pnl_pct,
                                'reason': 'stop_loss' if pnl_pct <= -stop_loss_pct else 'take_profit'
                            })
                            current_position = 0
                            hold_count = min_holding_periods

                # 新信號處理
                if hold_count == 0 and current_position == 0:
                    if signal == 1:  # 買入
                        current_position = position_size_pct
                        entry_price = current_price
                        hold_count = min_holding_periods
                    elif signal == -1:  # 賣出
                        current_position = -position_size_pct
                        entry_price = current_price
                        hold_count = min_holding_periods

                portfolio_values.append(portfolio_value)

            # 計算績效指標
            portfolio_series = pd.Series(portfolio_values)
            returns_series = portfolio_series.pct_change().dropna()

            total_return = (portfolio_value / 10000.0) - 1

            if len(returns_series) > 0:
                sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
                max_drawdown = self.calculate_max_drawdown(portfolio_series)
                volatility = returns_series.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                volatility = 0

            # 風險約束檢查
            risk_violation = max_drawdown > max_drawdown_limit

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'total_trades': len(trades),
                'risk_violation': risk_violation,
                'win_rate': len([t for t in trades if t['pnl_pct'] > 0]) / max(len(trades), 1)
            }

        except Exception as e:
            self.logger.error(f"回測模擬失敗: {e}")
            return {
                'total_return': -999,
                'sharpe_ratio': -999,
                'max_drawdown': 999,
                'volatility': 999,
                'total_trades': 0,
                'risk_violation': True,
                'win_rate': 0
            }

    def calculate_max_drawdown(self, values: pd.Series) -> float:
        """計算最大回撤"""
        rolling_max = values.expanding().max()
        drawdowns = (values - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def evaluate_cv_parameters(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> Dict:
        """評估交叉驗證參數"""
        try:
            n_splits = params['n_splits']
            test_size = params.get('test_size', 0.2)
            purge_pct = params.get('purge_pct', 0.01)

            # 時序交叉驗證
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * test_size))

            cv_scores = []
            stability_scores = []

            for train_idx, val_idx in tscv.split(X):
                # 應用purge（移除訓練和驗證之間的數據）
                purge_size = int(len(train_idx) * purge_pct)
                if purge_size > 0:
                    train_idx = train_idx[:-purge_size]

                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 簡單模型評估（這裡用隨機預測模擬）
                np.random.seed(42)
                y_pred = np.random.choice([0, 1, 2], size=len(y_val))

                score = f1_score(y_val, y_pred, average='weighted')
                cv_scores.append(score)

                # 計算穩定性（預測分佈的一致性）
                pred_dist = np.bincount(y_pred, minlength=3) / len(y_pred)
                target_dist = np.array([0.25, 0.5, 0.25])
                stability = 1 - np.sum(np.abs(pred_dist - target_dist)) / 2
                stability_scores.append(stability)

            return {
                'cv_mean': np.mean(cv_scores) if cv_scores else 0,
                'cv_std': np.std(cv_scores) if cv_scores else 0,
                'stability_mean': np.mean(stability_scores) if stability_scores else 0,
                'n_folds_completed': len(cv_scores)
            }

        except Exception as e:
            self.logger.error(f"CV參數評估失敗: {e}")
            return {'cv_mean': 0, 'cv_std': 0, 'stability_mean': 0, 'n_folds_completed': 0}

    def validate_no_future_leak(self, train_idx, test_idx):
        """檢查索引無未來洩漏"""
        if max(train_idx) >= min(test_idx):
            self.logger.warning('洩漏檢測: train max >= test min')
            return False
        # purge/embargo檢查
        purge_start = max(0, max(train_idx) - self.purge_periods)
        embargo_end = min(len(self.data), max(train_idx) + self.embargo_periods)
        if any(i < purge_start for i in train_idx) or any(i < embargo_end for i in test_idx):
            self.logger.warning('purge/embargo違規')
            return False
        return True

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna目標函數 - 第4層：CV與風控參數優化"""

        # 第4層參數：交叉驗證與風控參數
        params = {
            # 交叉驗證參數
            'n_splits': trial.suggest_int('n_splits', 3, 5),
            'test_size': trial.suggest_float('test_size', 0.2, 0.3),
            'purge_pct': 0.2,  # 48h
            'embargo_pct': 0.067,  # 16h

            # 風控參數
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.01, 0.1),
            'take_profit_pct': trial.suggest_float('take_profit_pct', 0.01, 0.1),
            'max_drawdown_limit': trial.suggest_float('max_drawdown_limit', 0.1, 0.3),
            'position_size_pct': trial.suggest_float('position_size_pct', 0.01, 0.1),
        }

        # Capacity limit clip after suggest
        if params['position_size_pct'] > 0.05:
            params['position_size_pct'] = 0.05
            self.logger.warning('容量限制超過，限於0.05')

        # 回測策略參數
        params['min_holding_periods'] = trial.suggest_int('min_holding_periods', 3, 10)
        params['sma_short'] = trial.suggest_int('sma_short', 10, 30)
        params['sma_long'] = trial.suggest_int('sma_long', 20, 50)

        # 風險控制權重
        params['risk_weight'] = trial.suggest_float('risk_weight', 0.3, 0.7)
        params['return_weight'] = trial.suggest_float('return_weight', 0.3, 0.7)

        try:
            # 讀取真實 OHLCV 並生成特徵與標籤（使用本模組內建簡化版本，與src解耦）
            ohlcv_file = self.data_path / "raw" / self.symbol / f"{self.symbol}_{self.timeframe}_ohlcv.parquet"
            if not ohlcv_file.exists():
                raise FileNotFoundError(f"未找到原始 OHLCV 數據: {ohlcv_file}")

            ohlcv = pd.read_parquet(ohlcv_file)
            # 內建特徵（與Layer2一致的簡化版）
            X_full = pd.DataFrame(index=ohlcv.index)
            close, high, low, volume = ohlcv['close'], ohlcv['high'], ohlcv['low'], ohlcv['volume']
            X_full['ret'] = close.pct_change()
            for w in [5, 10, 20]:
                X_full[f'sma_{w}'] = close.rolling(w).mean()
                X_full[f'rsi_{w}'] = 100 - 100 / (1 + (close.diff().clip(lower=0).rolling(w).mean() / (-(close.diff().clip(upper=0)).rolling(w).mean() + 1e-9)))
            X_full['bb_pos_20'] = (close - (close.rolling(20).mean() - 2 * close.rolling(20).std())) / (4 * close.rolling(20).std() + 1e-9)
            X_full['vol_sma_ratio'] = volume / volume.rolling(20).mean()
            X_full = X_full.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)

            # 內建量化分位標籤（與Layer1口徑一致）
            lag = 3
            future = close.shift(-lag)
            ret_fut = (future - close) / close
            ret_hist = ret_fut.shift(1)
            upper = ret_hist.rolling(500, min_periods=100).quantile(0.75)
            lower = ret_hist.rolling(500, min_periods=100).quantile(0.25)
            y = pd.Series(1, index=close.index)
            y = y.mask(ret_fut > upper, 2).mask(ret_fut < lower, 0).dropna().astype(int)

            # 對齊索引
            common_idx = X_full.index.intersection(y.index)
            X = X_full.loc[common_idx]
            y = y.loc[common_idx]

            # 以收盤價收益作為回測 returns
            returns = close.pct_change().loc[common_idx].fillna(0)

            # 評估CV參數
            cv_results = self.evaluate_cv_parameters(X, y, params)

            if cv_results['n_folds_completed'] < 2:
                return -999.0  # CV失敗

            # 模擬回測
            backtest_results = self.simulate_backtest_with_risk_params(returns, params)

            # 風險約束檢查
            if backtest_results['risk_violation']:
                return -999.0

            if backtest_results['total_trades'] < 5:
                return -999.0  # 交易次數太少

            # 綜合評分
            cv_score = cv_results['cv_mean']
            stability_score = cv_results['stability_mean']
            sharpe_ratio = backtest_results['sharpe_ratio']

            # 風險調整得分
            risk_penalty = backtest_results['max_drawdown'] * 2

            final_score = (cv_score * 0.3 +
                          stability_score * 0.2 +
                          sharpe_ratio * 0.4 +
                          backtest_results['win_rate'] * 0.1 -
                          risk_penalty)

            return final_score

        except Exception as e:
            self.logger.error(f"CV風控優化過程出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 100) -> Dict:
        """執行交叉驗證與風控參數優化"""
        self.logger.info("開始交叉驗證與風控參數優化（第4層）...")

        # 創建研究
        study = optuna.create_study(
            direction='maximize',
            study_name='cv_risk_optimization_layer4'
        )

        # 執行優化
        study.optimize(self.objective, n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"CV風控優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"最優參數: {best_params}")

        # 使用最優參數重新評估
        try:
            # 重新模擬並評估
            np.random.seed(42)
            returns = pd.Series(np.random.randn(1000) * 0.02)
            backtest_results = self.simulate_backtest_with_risk_params(returns, best_params)

            detailed_results = {
                'backtest_simulation': backtest_results
            }
        except Exception as e:
            self.logger.warning(f"無法獲取詳細結果: {e}")
            detailed_results = {}

        # 保存結果
        result = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': n_trials,
            'detailed_results': detailed_results,
            'optimization_history': [
                {'trial': i, 'score': trial.value}
                for i, trial in enumerate(study.trials)
                if trial.value is not None
            ]
        }

        # 保存到JSON文件
        output_file = self.config_path / "cv_risk_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"結果已保存至: {output_file}")

        return result


def main():
    """主函數"""
    optimizer = CVRiskOptimizer(data_path='../data', config_path='../configs')
    result = optimizer.optimize(n_trials=100)
    print(f"CV風控優化完成: {result['best_score']:.4f}")


if __name__ == "__main__":
    main()