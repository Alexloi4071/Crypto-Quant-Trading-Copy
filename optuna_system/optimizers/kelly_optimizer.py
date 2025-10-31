# -*- coding: utf-8 -*-
"""
Kelly公式資金管理優化器
基於Kelly準則優化倉位大小和風險管理參數
"""
import optuna
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')


class KellyOptimizer:
    """Kelly公式資金管理參數優化器"""

    def __init__(self, data_path: str, config_path: str = "configs/"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.config_path.mkdir(exist_ok=True)

        # 使用集中日誌 (由上層/入口初始化)，避免重複 basicConfig
        self.logger = logging.getLogger(__name__)

        # 在__init__中添加：
        import os
        from pathlib import Path
        workspace_root = Path(os.getcwd())

    def calculate_kelly_fraction(self, returns: pd.Series, win_rate: float,
                                avg_win: float, avg_loss: float) -> float:
        """計算Kelly分數"""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        # Kelly公式: f = (bp - q) / b
        # 其中 b = 平均盈利/平均虧損, p = 勝率, q = 敗率
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate

        kelly_f = (b * p - q) / b
        return max(0, min(kelly_f, 1))  # 限制在[0,1]範圍內

    def simulate_kelly_strategy(self, returns: pd.Series, params: Dict) -> Dict:
        """模擬Kelly策略表現"""
        try:
            kelly_fraction = params['kelly_fraction']
            lookback = params['kelly_lookback']
            max_position = params['max_kelly_position']
            vol_scaling_min = params.get('vol_scaling_min', 0.5)
            vol_scaling_max = params.get('vol_scaling_max', 2.0)
            target_vol = params.get('target_volatility', 0.15)

            # 模擬交易
            portfolio_values = [10000]  # 初始資金
            positions = []

            for i in range(lookback, len(returns)):
                # 計算歷史績效指標
                hist_returns = returns.iloc[i-lookback:i]
                positive_returns = hist_returns[hist_returns > 0]
                negative_returns = hist_returns[hist_returns < 0]

                if len(positive_returns) == 0 or len(negative_returns) == 0:
                    position_size = 0.0
                else:
                    win_rate = len(positive_returns) / len(hist_returns)
                    avg_win = positive_returns.mean()
                    avg_loss = negative_returns.mean()

                    # 計算Kelly分數
                    kelly_f = self.calculate_kelly_fraction(hist_returns, win_rate, avg_win, avg_loss)

                    # 波動率調整
                    hist_vol = hist_returns.std() * np.sqrt(252)
                    vol_adjustment = target_vol / max(hist_vol, 0.01)
                    vol_adjustment = np.clip(vol_adjustment, vol_scaling_min, vol_scaling_max)

                    # 最終倉位
                    position_size = kelly_f * kelly_fraction * vol_adjustment
                    position_size = min(position_size, max_position)

                positions.append(position_size)

                # 計算收益
                if i < len(returns):
                    portfolio_return = position_size * returns.iloc[i]
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)

            # 計算績效指標
            portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            max_drawdown = self.calculate_max_drawdown(pd.Series(portfolio_values))
            volatility = portfolio_returns.std() * np.sqrt(252)

            # 倉位統計
            position_series = pd.Series(positions)
            avg_position = position_series.mean()
            max_position_used = position_series.max()
            position_volatility = position_series.std()

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'avg_position': avg_position,
                'max_position_used': max_position_used,
                'position_volatility': position_volatility,
                'num_trades': len([p for p in positions if p > 0.01])
            }

        except Exception as e:
            self.logger.error(f"Kelly策略模擬失敗: {e}")
            return {
                'total_return': -999,
                'sharpe_ratio': -999,
                'max_drawdown': 999,
                'volatility': 999,
                'avg_position': 0,
                'max_position_used': 0,
                'position_volatility': 0,
                'num_trades': 0
            }

    def calculate_max_drawdown(self, values: pd.Series) -> float:
        """計算最大回撤"""
        rolling_max = values.expanding().max()
        drawdowns = (values - rolling_max) / rolling_max
        return abs(drawdowns.min())

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna優化目標函數"""

        # Kelly參數空間
        params = {
            'kelly_fraction': trial.suggest_float('kelly_fraction', 0.1, 0.8),
            'kelly_lookback': trial.suggest_int('kelly_lookback', 50, 300),
            'max_kelly_position': trial.suggest_float('max_kelly_position', 0.1, 0.5),
            'vol_scaling_min': trial.suggest_float('vol_scaling_min', 0.3, 0.8),
            'vol_scaling_max': trial.suggest_float('vol_scaling_max', 1.2, 3.0),
            'target_volatility': trial.suggest_float('target_volatility', 0.10, 0.25),
            'confidence_power': trial.suggest_float('confidence_power', 0.5, 2.0),
            'max_drawdown_limit': trial.suggest_float('max_drawdown_limit', 0.15, 0.35)
        }

        try:
            # 加載收益數據
            data_file = self.data_path / "returns_data.csv"
            if not data_file.exists():
                # 嘗試其他可能的數據文件
                alternative_files = [
                    self.data_path / "BTCUSDT_15m_returns.csv",
                    self.data_path / "backtest_returns.csv",
                    self.data_path / "strategy_returns.csv"
                ]

                data_file = None
                for alt_file in alternative_files:
                    if alt_file.exists():
                        data_file = alt_file
                        break

                if data_file is None:
                    self.logger.warning("無法找到收益數據文件")
                    return -999.0

            # 讀取數據
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)

            # 確定收益列
            returns_col = None
            for col in ['returns', 'strategy_returns', 'pnl_pct', 'return']:
                if col in df.columns:
                    returns_col = col
                    break

            if returns_col is None:
                self.logger.warning("無法找到收益數據列")
                return -999.0

            returns = df[returns_col].dropna()

            if len(returns) < 100:
                return -999.0

            # 運行Kelly策略模擬
            results = self.simulate_kelly_strategy(returns, params)

            # 風險約束
            if results['max_drawdown'] > params['max_drawdown_limit']:
                return -999.0

            if results['num_trades'] < 10:  # 至少要有一定的交易次數
                return -999.0

            # 多目標優化得分
            sharpe_ratio = results['sharpe_ratio']
            total_return = results['total_return']
            max_dd = results['max_drawdown']

            # 風險調整後收益
            risk_adjusted_return = total_return / max(max_dd, 0.01)

            # 綜合得分
            score = sharpe_ratio * 0.6 + risk_adjusted_return * 0.3 + min(results['num_trades']/100, 1.0) * 0.1

            return score

        except Exception as e:
            self.logger.error(f"Kelly優化過程出錯: {e}")
            return -999.0

    def optimize(self, n_trials: int = 100) -> Dict:
        """執行Kelly公式參數優化"""
        self.logger.info("開始Kelly公式資金管理參數優化...")

        # 創建研究
        study = optuna.create_study(
            direction='maximize',
            study_name='kelly_optimization'
        )

        # 執行優化
        study.optimize(self.objective, n_trials=n_trials)

        # 獲取最優參數
        best_params = study.best_params
        best_score = study.best_value

        self.logger.info(f"Kelly優化完成! 最佳得分: {best_score:.4f}")
        self.logger.info(f"最優參數: {best_params}")

        # 使用最優參數重新評估
        try:
            data_file = self.data_path / "returns_data.csv"
            if data_file.exists():
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                returns_col = 'returns' if 'returns' in df.columns else df.columns[0]
                returns = df[returns_col].dropna()
                detailed_results = self.simulate_kelly_strategy(returns, best_params)
            else:
                detailed_results = {}
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
        output_file = self.config_path / "kelly_params.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info(f"結果已保存至: {output_file}")

        return result


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='Kelly公式資金管理參數優化')
    parser.add_argument('--data_path', type=str, default='data/', help='數據路徑')
    parser.add_argument('--config_path', type=str, default='configs/', help='配置保存路徑')
    parser.add_argument('--n_trials', type=int, default=100, help='優化試驗次數')

    args = parser.parse_args()

    # 創建優化器
    optimizer = KellyOptimizer(
        data_path=args.data_path,
        config_path=args.config_path
    )

    # 執行優化
    result = optimizer.optimize(n_trials=args.n_trials)

    print("Kelly公式資金管理參數優化完成!")
    print(f"最優參數已保存至: {optimizer.config_path}/kelly_params.json")


if __name__ == "__main__":
    main()