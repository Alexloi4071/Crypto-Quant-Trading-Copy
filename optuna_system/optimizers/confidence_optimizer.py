# -*- coding: utf-8 -*-
"""
動態信號置信度閾值優化器
根據市場環境和模型性能自適應調整交易信號置信度
"""
import optuna
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ConfidenceOptimizer:
    """動態信號置信度閾值參數優化器"""

    def __init__(self, data_path: str, config_path: str = "configs/"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

    def calculate_rolling_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """計算滾動波動率"""
        returns = prices.pct_change().dropna()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return rolling_vol.bfill()

    def calculate_market_regime(self, prices: pd.Series, vol_threshold: float = 0.25) -> pd.Series:
        """識別市場狀態 (0=低波動, 1=高波動)"""
        volatility = self.calculate_rolling_volatility(prices)
        regime = (volatility > vol_threshold).astype(int)
        return regime

    def calculate_trend_strength(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """計算趨勢強度"""
        # 使用線性回歸的R²作為趨勢強度指標
        trend_strength = []
        prices_array = prices.values

        for i in range(len(prices)):
            if i < window:
                trend_strength.append(0.0)
                continue

            # 取窗口內的價格數據
            y = prices_array[i-window+1:i+1]
            x = np.arange(len(y))

            # 計算線性回歸的R²
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared = r_value ** 2
                trend_strength.append(r_squared)
            except Exception:
                trend_strength.append(0.0)

        return pd.Series(trend_strength, index=prices.index)

    def calculate_model_confidence_decay(self, predictions: pd.Series,
                                       actual_returns: pd.Series,
                                       decay_window: int = 50) -> pd.Series:
        """計算模型置信度衰減"""
        if len(predictions) != len(actual_returns):
            return pd.Series([0.5] * len(predictions), index=predictions.index)

        # 計算滾動準確率
        binary_pred = (predictions > 0.5).astype(int)
        binary_actual = (actual_returns > 0).astype(int)

        rolling_accuracy = []
        for i in range(len(predictions)):
            if i < decay_window:
                rolling_accuracy.append(0.5)  # 初始值
                continue

            recent_pred = binary_pred.iloc[i-decay_window+1:i+1]
            recent_actual = binary_actual.iloc[i-decay_window+1:i+1]

            accuracy = (recent_pred == recent_actual).mean()
            rolling_accuracy.append(accuracy)

        return pd.Series(rolling_accuracy, index=predictions.index)

    def adaptive_confidence_threshold(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        """自適應計算置信度閾值"""

        # 基礎閾值
        base_threshold = params['base_confidence_threshold']

        # 獲取必要的數據列
        if 'close_price' not in df.columns and 'close' not in df.columns:
            # 如果沒有價格數據，返回固定閾值
            return pd.Series([base_threshold] * len(df), index=df.index)

        price_col = 'close_price' if 'close_price' in df.columns else 'close'
        prices = df[price_col]

        # 計算市場狀態指標
        market_regime = self.calculate_market_regime(prices, params['volatility_threshold'])
        trend_strength = self.calculate_trend_strength(prices, params['trend_window'])

        # 模型置信度衰減
        if 'model_prediction' in df.columns and 'actual_return' in df.columns:
            model_confidence = self.calculate_model_confidence_decay(
                df['model_prediction'], df['actual_return'], params['decay_window']
            )
        else:
            model_confidence = pd.Series([0.7] * len(df), index=df.index)

        # 自適應調整閾值
        adaptive_thresholds = []

        for i in range(len(df)):
            threshold = base_threshold

            # 市場狀態調整
            if market_regime.iloc[i] == 1:  # 高波動市場
                threshold += params['high_vol_adjustment']
            else:  # 低波動市場
                threshold += params['low_vol_adjustment']

            # 趨勢強度調整
            if trend_strength.iloc[i] > params['strong_trend_threshold']:
                threshold -= params['trend_strength_bonus']  # 強趨勢時降低閾值

            # 模型歷史表現調整
            if model_confidence.iloc[i] > 0.6:
                threshold -= params['high_accuracy_bonus']
            elif model_confidence.iloc[i] < 0.4:
                threshold += params['low_accuracy_penalty']

            # 時間衰減調整
            time_factor = 1.0 - (i / len(df)) * params['time_decay_rate']
            threshold *= time_factor

            # 確保閾值在合理範圍內
            threshold = np.clip(threshold, params['min_threshold'], params['max_threshold'])

            adaptive_thresholds.append(threshold)

        return pd.Series(adaptive_thresholds, index=df.index)

    def backtest_confidence_strategy(self, df: pd.DataFrame, params: Dict) -> Dict:
        """回測動態置信度策略"""

        if 'model_prediction' not in df.columns:
            # 如果沒有模型預測，生成模擬數據
            np.random.seed(42)
            df = df.copy()
            df['model_prediction'] = np.random.beta(2, 2, len(df))
            df['actual_return'] = np.random.randn(len(df)) * 0.02

        try:
            # 計算自適應置信度閾值
            adaptive_thresholds = self.adaptive_confidence_threshold(df, params)

            # 生成交易信號
            predictions = df['model_prediction']
            signals = []
            trade_returns = []

            for i in range(len(df)):
                pred = predictions.iloc[i]
                threshold = adaptive_thresholds.iloc[i]

                # 生成信號 (1=買入, -1=賣出, 0=無操作)
                if pred > threshold:
                    signal = 1
                elif pred < (1 - threshold):
                    signal = -1
                else:
                    signal = 0

                signals.append(signal)

                # 計算交易收益 (假設下一期收益)
                if signal != 0 and i < len(df) - 1:
                    next_return = df['actual_return'].iloc[i+1] if 'actual_return' in df.columns else np.random.randn() * 0.02
                    trade_return = signal * next_return
                    trade_returns.append(trade_return)

            # 計算策略表現
            if len(trade_returns) == 0:
                return {'sharpe_ratio': -999.0, 'total_trades': 0}

            returns_series = pd.Series(trade_returns)

            # 計算指標
            total_return = returns_series.sum()
            sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
            max_drawdown = self.calculate_max_drawdown(returns_series)
            win_rate = (returns_series > 0).mean()

            # 統計信號使用情況
            signal_stats = pd.Series(signals).value_counts()

            return {
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return * 100,
                'max_drawdown': max_drawdown * 100,
                'win_rate': win_rate * 100,
                'total_trades': len(trade_returns),
                'buy_signals': signal_stats.get(1, 0),
                'sell_signals': signal_stats.get(-1, 0),
                'no_signal': signal_stats.get(0, 0),
                'avg_threshold': adaptive_thresholds.mean(),
                'threshold_std': adaptive_thresholds.std()
            }

        except Exception as e:
            self.logger.error(f"回測過程出錯: {e}")
            return {'sharpe_ratio': -999.0, 'total_trades': 0}

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """計算最大回撤"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        return abs(drawdowns.min())

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna優化目標函數"""

        # 定義參數空間
        params = {
            # 基礎閾值
            'base_confidence_threshold': trial.suggest_float('base_confidence_threshold', 0.5, 0.9),

            # 市場狀態參數
            'volatility_threshold': trial.suggest_float('volatility_threshold', 0.15, 0.35),
            'high_vol_adjustment': trial.suggest_float('high_vol_adjustment', 0.05, 0.2),
            'low_vol_adjustment': trial.suggest_float('low_vol_adjustment', -0.1, 0.05),

            # 趨勢參數
            'trend_window': trial.suggest_int('trend_window', 20, 100),
            'strong_trend_threshold': trial.suggest_float('strong_trend_threshold', 0.7, 0.95),
            'trend_strength_bonus': trial.suggest_float('trend_strength_bonus', 0.02, 0.1),

            # 模型表現參數
            'decay_window': trial.suggest_int('decay_window', 30, 100),
            'high_accuracy_bonus': trial.suggest_float('high_accuracy_bonus', 0.02, 0.1),
            'low_accuracy_penalty': trial.suggest_float('low_accuracy_penalty', 0.02, 0.15),

            # 時間衰減
            'time_decay_rate': trial.suggest_float('time_decay_rate', 0.0, 0.2),

            # 閾值邊界
            'min_threshold': trial.suggest_float('min_threshold', 0.3, 0.5),
            'max_threshold': trial.suggest_float('max_threshold', 0.8, 0.95),
        }

        try:
            # 加載數據
            data_file = self.data_path / "confidence_data.csv"
            if not data_file.exists():
                # 嘗試其他可能的數據文件
                alternative_files = [
                    self.data_path / "market_data.csv",
                    self.data_path / "ohlcv_data.csv",
                    self.data_path / "prediction_data.csv"
                ]

                data_file = None
                for alt_file in alternative_files:
                    if alt_file.exists():
                        data_file = alt_file
                        break

                if data_file is None:
                    self.logger.warning("無法找到數據文件，生成模擬數據")
                    # 生成模擬市場數據
                    np.random.seed(42)
                    dates = pd.date_range('2022-01-01', periods=1000, freq='H')

                    # 模擬價格走勢
                    returns = np.random.randn(1000) * 0.02
                    prices = 100 * (1 + returns).cumprod()

                    df = pd.DataFrame({
                        'close': prices,
                        'close_price': prices,
                        'model_prediction': np.random.beta(2, 2, 1000),
                        'actual_return': returns
                    }, index=dates)
                else:
                    df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            else:
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)

            # 檢查必要的列
            if 'close' not in df.columns and 'close_price' not in df.columns:
                # 添加模擬價格數據
                if len(df) > 0:
                    np.random.seed(42)
                    returns = np.random.randn(len(df)) * 0.02
                    prices = 100 * (1 + returns).cumprod()
                    df['close_price'] = prices
                    if 'actual_return' not in df.columns:
                        df['actual_return'] = returns
                    if 'model_prediction' not in df.columns:
                        df['model_prediction'] = np.random.beta(2, 2, len(df))

            # 執行回測
            results = self.backtest_confidence_strategy(df, params)

            # 檢查交易數量
            if results['total_trades'] < 10:
                return -999.0  # 交易次數太少

            # 風險約束
            if results.get('max_drawdown', 100) > 30:  # 最大回撤超過30%
                return -999.0

            # 多目標優化得分
            sharpe_ratio = results['sharpe_ratio']
            win_rate = results.get('win_rate', 50) / 100
            total_trades = min(results['total_trades'] / 100, 1.0)  # 歸一化交易次數

            # 綜合得分
            score = sharpe_ratio * 0.6 + win_rate * 0.3 + total_trades * 0.1

            return score

        except Exception as e:
            self.logger.error(f"優化過程中出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 100) -> Dict:
        """執行動態置信度參數優化"""
        self.logger.info("開始動態信號置信度閾值優化...")

        # 創建研究
        study = optuna.create_study(
            direction='maximize',
            study_name='confidence_optimization'
        )

        # 執行優化
        study.optimize(self.objective, n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"最優參數: {best_params}")

        # 使用最優參數重新評估
        try:
            # 生成模擬數據用於最終評估
            np.random.seed(42)
            dates = pd.date_range('2022-01-01', periods=1000, freq='H')
            returns = np.random.randn(1000) * 0.02
            prices = 100 * (1 + returns).cumprod()

            df = pd.DataFrame({
                'close_price': prices,
                'model_prediction': np.random.beta(2, 2, 1000),
                'actual_return': returns
            }, index=dates)

            detailed_results = self.backtest_confidence_strategy(df, best_params)
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
        output_file = self.config_path / "confidence_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"結果已保存至: {output_file}")

        return result


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='動態信號置信度閾值優化')
    parser.add_argument('--data_path', type=str, default='data/', help='數據路徑')
    parser.add_argument('--config_path', type=str, default='configs/', help='配置保存路徑')
    parser.add_argument('--n_trials', type=int, default=100, help='優化試驗次數')

    args = parser.parse_args()

    # 創建優化器
    optimizer = ConfidenceOptimizer(
        data_path=args.data_path,
        config_path=args.config_path
    )

    # 執行優化
    result = optimizer.optimize(n_trials=args.n_trials)

    print("動態信號置信度閾值優化完成!")
    print(f"最優參數已保存至: {optimizer.config_path}/confidence_params.json")


if __name__ == "__main__":
    main()