"""
Performance Analyzer Module
Comprehensive performance analysis for trading strategies
Includes risk metrics, drawdown analysis, and benchmark comparisons
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import warnings

warnings.filterwarnings('ignore')

# Analysis libraries
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator

logger = setup_logger(__name__)


class PerformanceAnalyzer:
    """Comprehensive trading performance analyzer"""

    def __init__(self):
        self.benchmarks = {
            'BTC': None,  # Will be loaded dynamically
            'ETH': None,
            'SPY': None
        }

        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    @timing_decorator
    def analyze_performance(self, returns: pd.Series,
                          portfolio_values: pd.Series = None,
                          benchmark_returns: pd.Series = None,
                          trades: pd.DataFrame = None,
                          metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive performance analysis

        Args:
            returns: Series of returns (daily/period returns)
            portfolio_values: Series of portfolio values over time
            benchmark_returns: Benchmark returns for comparison
            trades: DataFrame with trade information
            metadata: Additional metadata (symbol, strategy, etc.)

        Returns:
            Dictionary with comprehensive performance metrics
        """
        try:
            logger.info("Starting comprehensive performance analysis")

            if returns.empty:
                logger.warning("Empty returns series provided")
                return {}

            # Ensure datetime index
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.to_datetime(returns.index)

            # Remove NaN values
            returns = returns.dropna()

            if len(returns) < 10:
                logger.warning("Insufficient data for analysis")
                return {}

            # Initialize results
            results = {
                'analysis_date': datetime.now().isoformat(),
                'period_start': returns.index[0].strftime('%Y-%m-%d'),
                'period_end': returns.index[-1].strftime('%Y-%m-%d'),
                'data_points': len(returns)
            }

            # Add metadata
            if metadata:
                results.update(metadata)

            # Core performance metrics
            results['return_metrics'] = self._calculate_return_metrics(returns)
            results['risk_metrics'] = self._calculate_risk_metrics(returns)
            results['drawdown_analysis'] = self._analyze_drawdowns(returns, portfolio_values)
            results['statistical_metrics'] = self._calculate_statistical_metrics(returns)

            # Benchmark comparison
            if benchmark_returns is not None:
                results['benchmark_comparison'] = self._compare_to_benchmark(
                    returns, benchmark_returns
                )

            # Trade analysis
            if trades is not None and not trades.empty:
                results['trade_analysis'] = self._analyze_trades(trades)

            # Time-based analysis
            results['time_analysis'] = self._analyze_time_patterns(returns)

            # Advanced metrics
            results['advanced_metrics'] = self._calculate_advanced_metrics(returns)

            # Risk-adjusted performance
            results['risk_adjusted'] = self._calculate_risk_adjusted_metrics(returns)

            # Performance summary
            results['summary'] = self._generate_performance_summary(results)

            logger.info("Performance analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}


    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based performance metrics"""
        try:
            # Basic return statistics
            total_return = (1 + returns).prod() - 1
            mean_return = returns.mean()

            # Annualized metrics
            periods_per_year = self._get_periods_per_year(returns)
            annual_return = (1 + mean_return) ** periods_per_year - 1

            # Cumulative returns
            cumulative_returns = (1 + returns).cumprod() - 1

            # Win/Loss statistics
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

            # Best and worst periods
            best_return = returns.max()
            worst_return = returns.min()

            # Consecutive statistics
            consecutive_wins = self._calculate_consecutive_periods(returns > 0)
            consecutive_losses = self._calculate_consecutive_periods(returns < 0)

            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'mean_return': float(mean_return),
                'win_rate': float(win_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'best_return': float(best_return),
                'worst_return': float(worst_return),
                'max_consecutive_wins': int(consecutive_wins['max']),
                'max_consecutive_losses': int(consecutive_losses['max']),
                'current_streak': int(consecutive_wins['current'] if returns.iloc[-1] > 0 else consecutive_losses['current'])
            }

        except Exception as e:
            logger.error(f"Return metrics calculation failed: {e}")
            return {}


    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based performance metrics"""
        try:
            # Volatility metrics
            periods_per_year = self._get_periods_per_year(returns)
            volatility = returns.std()
            annual_volatility = volatility * np.sqrt(periods_per_year)

            # Downside risk
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            annual_downside_risk = downside_std * np.sqrt(periods_per_year)

            # Value at Risk (VaR)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0

            # Skewness and Kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)

            # Tail ratio
            tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0

            return {
                'volatility': float(volatility),
                'annual_volatility': float(annual_volatility),
                'downside_risk': float(downside_std),
                'annual_downside_risk': float(annual_downside_risk),
                'var_95': float(var_95),
                'var_99': float(var_99),
                'cvar_95': float(cvar_95),
                'cvar_99': float(cvar_99),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'tail_ratio': float(tail_ratio)
            }

        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}


    def _analyze_drawdowns(self, returns: pd.Series,
                          portfolio_values: pd.Series = None) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        try:
            # Calculate cumulative returns for drawdown
            if portfolio_values is not None:
                wealth_index = portfolio_values / portfolio_values.iloc[0]
            else:
                wealth_index = (1 + returns).cumprod()

            # Calculate running maximum (peak)
            previous_peaks = wealth_index.expanding(min_periods=1).max()

            # Calculate drawdown
            drawdown = (wealth_index - previous_peaks) / previous_peaks

            # Drawdown statistics
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin()

            # Find the peak before max drawdown
            max_dd_peak_date = wealth_index[:max_drawdown_date].idxmax()

            # Recovery analysis
            recovery_dates = []
            current_dd_start = None

            for i, dd in enumerate(drawdown):
                if dd == 0 and current_dd_start is not None:
                    # Recovery completed
                    recovery_dates.append({
                        'start': current_dd_start,
                        'bottom': drawdown[current_dd_start:].idxmin(),
                        'recovery': drawdown.index[i],
                        'duration': (drawdown.index[i] - current_dd_start).days,
                        'depth': drawdown[current_dd_start:].min()
                    })
                    current_dd_start = None
                elif dd < 0 and current_dd_start is None:
                    current_dd_start = drawdown.index[i]

            # Current drawdown
            current_drawdown = drawdown.iloc[-1]
            is_in_drawdown = current_drawdown < -0.001  # Less than -0.1%

            # Drawdown duration
            max_dd_duration = (max_drawdown_date - max_dd_peak_date).days if max_dd_peak_date != max_drawdown_date else 0

            # Average drawdown
            negative_drawdowns = drawdown[drawdown < 0]
            avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0

            # Drawdown frequency
            drawdown_periods = len([dd for dd in recovery_dates if dd['depth'] < -0.05])  # >5% drawdowns

            return {
                'max_drawdown': float(max_drawdown),
                'max_drawdown_date': max_drawdown_date.strftime('%Y-%m-%d'),
                'max_drawdown_duration_days': int(max_dd_duration),
                'current_drawdown': float(current_drawdown),
                'is_in_drawdown': bool(is_in_drawdown),
                'avg_drawdown': float(avg_drawdown),
                'drawdown_periods': int(drawdown_periods),
                'recovery_periods': len(recovery_dates),
                'drawdown_series': drawdown.to_dict()
            }

        except Exception as e:
            logger.error(f"Drawdown analysis failed: {e}")
            return {}


    def _calculate_statistical_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate statistical performance metrics"""
        try:
            # Basic statistics
            mean_return = returns.mean()
            median_return = returns.median()
            std_return = returns.std()

            # Quantiles
            quantiles = returns.quantile([0.05, 0.25, 0.75, 0.95])

            # Normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(returns[:5000])  # Limit for shapiro test
                jarque_bera_stat, jarque_bera_p = stats.jarque_bera(returns)
            except:
                shapiro_stat = shapiro_p = 0
                jarque_bera_stat = jarque_bera_p = 0

            # Stability metrics
            rolling_mean = returns.rolling(window=min(30, len(returns)//4)).mean()
            stability = 1 - rolling_mean.std() / abs(mean_return) if mean_return != 0 else 0

            return {
                'mean': float(mean_return),
                'median': float(median_return),
                'std': float(std_return),
                'q5': float(quantiles.iloc[0]),
                'q25': float(quantiles.iloc[1]),
                'q75': float(quantiles.iloc[2]),
                'q95': float(quantiles.iloc[3]),
                'shapiro_stat': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'jarque_bera_stat': float(jarque_bera_stat),
                'jarque_bera_p_value': float(jarque_bera_p),
                'stability': float(stability)
            }

        except Exception as e:
            logger.error(f"Statistical metrics calculation failed: {e}")
            return {}


    def _compare_to_benchmark(self, returns: pd.Series,
                            benchmark_returns: pd.Series) -> Dict[str, float]:
        """Compare performance to benchmark"""
        try:
            # Align dates
            aligned_data = pd.DataFrame({
                'strategy': returns,
                'benchmark': benchmark_returns
            }).dropna()

            if aligned_data.empty:
                return {}

            strategy_returns = aligned_data['strategy']
            benchmark_returns = aligned_data['benchmark']

            # Relative performance
            excess_returns = strategy_returns - benchmark_returns

            # Beta calculation
            covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1

            # Alpha calculation (CAPM)
            periods_per_year = self._get_periods_per_year(returns)
            strategy_annual = (1 + strategy_returns.mean()) ** periods_per_year - 1
            benchmark_annual = (1 + benchmark_returns.mean()) ** periods_per_year - 1
            alpha = strategy_annual - (self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate))

            # Correlation
            correlation = strategy_returns.corr(benchmark_returns)

            # Tracking error
            tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

            # Information ratio
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
            information_ratio_annual = information_ratio * np.sqrt(periods_per_year)

            # Up/Down capture ratios
            up_periods = benchmark_returns > 0
            down_periods = benchmark_returns < 0

            up_capture = (strategy_returns[up_periods].mean() /
                         benchmark_returns[up_periods].mean()) if benchmark_returns[up_periods].mean() != 0 else 0

            down_capture = (strategy_returns[down_periods].mean() /
                           benchmark_returns[down_periods].mean()) if benchmark_returns[down_periods].mean() != 0 else 0

            return {
                'alpha': float(alpha),
                'beta': float(beta),
                'correlation': float(correlation),
                'tracking_error': float(tracking_error),
                'information_ratio': float(information_ratio_annual),
                'up_capture': float(up_capture),
                'down_capture': float(down_capture),
                'excess_return_mean': float(excess_returns.mean()),
                'excess_return_std': float(excess_returns.std())
            }

        except Exception as e:
            logger.error(f"Benchmark comparison failed: {e}")
            return {}


    def _analyze_trades(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual trades"""
        try:
            if trades.empty:
                return {}

            # Ensure required columns
            required_cols = ['pnl', 'entry_date', 'exit_date']
            available_cols = [col for col in required_cols if col in trades.columns]

            if 'pnl' not in available_cols:
                return {}

            # Basic trade statistics
            total_trades = len(trades)
            winning_trades = len(trades[trades['pnl'] > 0])
            losing_trades = len(trades[trades['pnl'] < 0])

            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # P&L statistics
            total_pnl = trades['pnl'].sum()
            avg_pnl = trades['pnl'].mean()

            winning_pnl = trades[trades['pnl'] > 0]['pnl']
            losing_pnl = trades[trades['pnl'] < 0]['pnl']

            avg_win = winning_pnl.mean() if len(winning_pnl) > 0 else 0
            avg_loss = losing_pnl.mean() if len(losing_pnl) > 0 else 0

            # Best and worst trades
            best_trade = trades['pnl'].max()
            worst_trade = trades['pnl'].min()

            # Profit factor
            gross_profit = winning_pnl.sum() if len(winning_pnl) > 0 else 0
            gross_loss = abs(losing_pnl.sum()) if len(losing_pnl) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Trade duration analysis (if dates available)
            duration_stats = {}
            if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
                trades['duration'] = pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])
                duration_days = trades['duration'].dt.days

                duration_stats = {
                    'avg_duration_days': float(duration_days.mean()),
                    'median_duration_days': float(duration_days.median()),
                    'max_duration_days': int(duration_days.max()),
                    'min_duration_days': int(duration_days.min())
                }

            # Consecutive trade analysis
            trade_results = (trades['pnl'] > 0).astype(int)
            consecutive_wins = self._calculate_consecutive_periods(trade_results == 1)
            consecutive_losses = self._calculate_consecutive_periods(trade_results == 0)

            result = {
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': float(win_rate),
                'total_pnl': float(total_pnl),
                'avg_pnl_per_trade': float(avg_pnl),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'best_trade': float(best_trade),
                'worst_trade': float(worst_trade),
                'profit_factor': float(profit_factor),
                'max_consecutive_wins': int(consecutive_wins['max']),
                'max_consecutive_losses': int(consecutive_losses['max']),
                **duration_stats
            }

            return result

        except Exception as e:
            logger.error(f"Trade analysis failed: {e}")
            return {}


    def _analyze_time_patterns(self, returns: pd.Series) -> Dict[str, Any]:
        """Analyze time-based performance patterns"""
        try:
            # Monthly analysis
            monthly_returns = returns.groupby(returns.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)

            # Weekday analysis
            weekday_returns = returns.groupby(returns.index.dayofweek).mean()
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_performance = dict(zip(weekday_names[:len(weekday_returns)], weekday_returns.values))

            # Monthly performance
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            if len(monthly_returns) > 0:
                monthly_avg = monthly_returns.groupby(monthly_returns.index.month).mean()
                monthly_performance = dict(zip([month_names[i-1] for i in monthly_avg.index], monthly_avg.values))
            else:
                monthly_performance = {}

            # Yearly analysis
            if len(returns) > 365:
                yearly_returns = returns.groupby(returns.index.year).apply(lambda x: (1 + x).prod() - 1)
                yearly_performance = yearly_returns.to_dict()
            else:
                yearly_performance = {}

            return {
                'monthly_performance': monthly_performance,
                'weekday_performance': weekday_performance,
                'yearly_performance': yearly_performance,
                'best_month': max(monthly_performance.items(), key=lambda x: x[1])[0] if monthly_performance else None,
                'worst_month': min(monthly_performance.items(), key=lambda x: x[1])[0] if monthly_performance else None,
                'best_weekday': max(weekday_performance.items(), key=lambda x: x[1])[0] if weekday_performance else None,
                'worst_weekday': min(weekday_performance.items(), key=lambda x: x[1])[0] if weekday_performance else None
            }

        except Exception as e:
            logger.error(f"Time pattern analysis failed: {e}")
            return {}


    def _calculate_advanced_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced performance metrics"""
        try:
            periods_per_year = self._get_periods_per_year(returns)

            # Calmar Ratio
            annual_return = (1 + returns.mean()) ** periods_per_year - 1
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding(min_periods=1).max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())

            calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0

            # Sterling Ratio
            avg_drawdown = abs(drawdown[drawdown < 0].mean()) if len(drawdown[drawdown < 0]) > 0 else max_drawdown
            sterling_ratio = annual_return / avg_drawdown if avg_drawdown != 0 else 0

            # Omega Ratio
            threshold = 0  # Return threshold
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns < threshold]

            omega_ratio = gains.sum() / losses.sum() if losses.sum() != 0 else float('inf')

            # Kappa 3 (similar to omega but with cubic weighting)
            if len(losses) > 0:
                kappa_3 = gains.sum() / (losses ** 3).sum() ** (1/3) if (losses ** 3).sum() > 0 else float('inf')
            else:
                kappa_3 = float('inf')

            # Gain-to-Pain Ratio
            pain = abs(returns[returns < 0].sum())
            gain = returns[returns > 0].sum()
            gain_to_pain = gain / pain if pain != 0 else float('inf')

            # Lake Ratio (time spent underwater)
            underwater_periods = len(drawdown[drawdown < 0])
            total_periods = len(drawdown)
            lake_ratio = underwater_periods / total_periods if total_periods > 0 else 0

            return {
                'calmar_ratio': float(calmar_ratio),
                'sterling_ratio': float(sterling_ratio),
                'omega_ratio': float(omega_ratio),
                'kappa_3': float(kappa_3),
                'gain_to_pain_ratio': float(gain_to_pain),
                'lake_ratio': float(lake_ratio)
            }

        except Exception as e:
            logger.error(f"Advanced metrics calculation failed: {e}")
            return {}


    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        try:
            periods_per_year = self._get_periods_per_year(returns)

            # Sharpe Ratio
            excess_returns = returns - (self.risk_free_rate / periods_per_year)
            sharpe_ratio = (excess_returns.mean() / excess_returns.std() *
                          np.sqrt(periods_per_year)) if excess_returns.std() != 0 else 0

            # Sortino Ratio
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else excess_returns.std()
            sortino_ratio = (excess_returns.mean() / downside_std *
                           np.sqrt(periods_per_year)) if downside_std != 0 else 0

            # Treynor Ratio (requires benchmark for beta)
            # Using volatility as proxy for systematic risk
            treynor_ratio = (excess_returns.mean() * periods_per_year) / returns.std() if returns.std() != 0 else 0

            # Modified Sharpe (adjust for skewness and kurtosis)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)

            modified_sharpe = sharpe_ratio * (1 + (skewness / 6) * sharpe_ratio -
                                            ((kurtosis - 3) / 24) * sharpe_ratio ** 2)

            return {
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'treynor_ratio': float(treynor_ratio),
                'modified_sharpe': float(modified_sharpe)
            }

        except Exception as e:
            logger.error(f"Risk-adjusted metrics calculation failed: {e}")
            return {}


    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary with overall ratings"""
        try:
            # Extract key metrics
            return_metrics = results.get('return_metrics', {})
            risk_metrics = results.get('risk_metrics', {})
            risk_adjusted = results.get('risk_adjusted', {})
            drawdown_analysis = results.get('drawdown_analysis', {})

            # Performance grades (A+ to F)
            grades = {}

            # Return grade
            annual_return = return_metrics.get('annual_return', 0)
            if annual_return > 0.3:
                grades['returns'] = 'A+'
            elif annual_return > 0.2:
                grades['returns'] = 'A'
            elif annual_return > 0.15:
                grades['returns'] = 'B+'
            elif annual_return > 0.1:
                grades['returns'] = 'B'
            elif annual_return > 0.05:
                grades['returns'] = 'C'
            else:
                grades['returns'] = 'D'

            # Risk grade (based on max drawdown)
            max_drawdown = abs(drawdown_analysis.get('max_drawdown', 0))
            if max_drawdown < 0.05:
                grades['risk'] = 'A+'
            elif max_drawdown < 0.1:
                grades['risk'] = 'A'
            elif max_drawdown < 0.15:
                grades['risk'] = 'B+'
            elif max_drawdown < 0.2:
                grades['risk'] = 'B'
            elif max_drawdown < 0.3:
                grades['risk'] = 'C'
            else:
                grades['risk'] = 'D'

            # Consistency grade (based on Sharpe ratio)
            sharpe_ratio = risk_adjusted.get('sharpe_ratio', 0)
            if sharpe_ratio > 2.0:
                grades['consistency'] = 'A+'
            elif sharpe_ratio > 1.5:
                grades['consistency'] = 'A'
            elif sharpe_ratio > 1.0:
                grades['consistency'] = 'B+'
            elif sharpe_ratio > 0.5:
                grades['consistency'] = 'B'
            elif sharpe_ratio > 0:
                grades['consistency'] = 'C'
            else:
                grades['consistency'] = 'D'

            # Overall grade (weighted average)
            grade_values = {'A+': 4.3, 'A': 4.0, 'B+': 3.3, 'B': 3.0, 'C': 2.0, 'D': 1.0}
            overall_score = (grade_values[grades['returns']] * 0.4 +
                           grade_values[grades['risk']] * 0.3 +
                           grade_values[grades['consistency']] * 0.3)

            overall_grade = 'D'
            for grade, value in sorted(grade_values.items(), key=lambda x: x[1], reverse=True):
                if overall_score >= value:
                    overall_grade = grade
                    break

            # Key highlights
            highlights = []

            if annual_return > 0.15:
                highlights.append(f"Strong annual return: {annual_return:.1%}")

            if max_drawdown < 0.1:
                highlights.append(f"Low maximum drawdown: {max_drawdown:.1%}")

            if sharpe_ratio > 1.0:
                highlights.append(f"Good risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")

            win_rate = return_metrics.get('win_rate', 0)
            if win_rate > 0.6:
                highlights.append(f"High win rate: {win_rate:.1%}")

            # Risk warnings
            warnings = []

            if max_drawdown > 0.2:
                warnings.append(f"High maximum drawdown: {max_drawdown:.1%}")

            if sharpe_ratio < 0.5:
                warnings.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")

            volatility = risk_metrics.get('annual_volatility', 0)
            if volatility > 0.4:
                warnings.append(f"High volatility: {volatility:.1%}")

            return {
                'overall_grade': overall_grade,
                'grades': grades,
                'highlights': highlights,
                'warnings': warnings,
                'recommendation': self._generate_recommendation(overall_grade, highlights, warnings)
            }

        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {}


    def _generate_recommendation(self, overall_grade: str,
                               highlights: List[str], warnings: List[str]) -> str:
        """Generate trading recommendation based on performance"""
        if overall_grade in ['A+', 'A']:
            return "Excellent performance. Consider increasing position size if risk tolerance allows."
        elif overall_grade in ['B+', 'B']:
            return "Good performance with room for improvement. Monitor risk levels closely."
        elif overall_grade == 'C':
            return "Average performance. Consider strategy optimization or parameter adjustment."
        else:
            return "Below-average performance. Strategy revision or replacement recommended."


    def _get_periods_per_year(self, returns: pd.Series) -> float:
        """Estimate periods per year based on data frequency"""
        if len(returns) < 2:
            return 252  # Default to daily

        # Calculate average time delta
        time_delta = (returns.index[-1] - returns.index[0]) / (len(returns) - 1)

        if time_delta.days >= 1:
            return 252  # Daily
        elif time_delta.seconds >= 3600:
            return 252 * 24  # Hourly
        else:
            return 252 * 24 * 60  # Minute


    def _calculate_consecutive_periods(self, condition_series: pd.Series) -> Dict[str, int]:
        """Calculate consecutive periods statistics"""
        try:
            if condition_series.empty:
                return {'max': 0, 'current': 0}

            # Find consecutive periods
            consecutive_counts = []
            current_count = 0

            for value in condition_series:
                if value:
                    current_count += 1
                else:
                    if current_count > 0:
                        consecutive_counts.append(current_count)
                    current_count = 0

            # Handle case where series ends with consecutive values
            if current_count > 0:
                consecutive_counts.append(current_count)

            # Current streak (from the end)
            current_streak = 0
            for value in reversed(condition_series):
                if value:
                    current_streak += 1
                else:
                    break

            return {
                'max': max(consecutive_counts) if consecutive_counts else 0,
                'current': current_streak
            }

        except Exception as e:
            logger.error(f"Consecutive periods calculation failed: {e}")
            return {'max': 0, 'current': 0}


    def plot_performance(self, returns: pd.Series,
                        portfolio_values: pd.Series = None,
                        benchmark_returns: pd.Series = None,
                        save_path: str = None) -> str:
        """Create comprehensive performance plots"""
        try:
            # Setup plot style
            plt.style.use('seaborn-v0_8-whitegrid')

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 12))

            # 1. Cumulative Returns
            ax1 = plt.subplot(3, 2, 1)
            cumulative_returns = (1 + returns).cumprod()
            ax1.plot(cumulative_returns.index, cumulative_returns, label='Strategy', linewidth=2)

            if benchmark_returns is not None:
                benchmark_cumulative = (1 + benchmark_returns).cumprod()
                ax1.plot(benchmark_cumulative.index, benchmark_cumulative,
                        label='Benchmark', linewidth=2, alpha=0.7)

            ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Drawdown
            ax2 = plt.subplot(3, 2, 2)
            wealth_index = cumulative_returns
            previous_peaks = wealth_index.expanding(min_periods=1).max()
            drawdown = (wealth_index - previous_peaks) / previous_peaks

            ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            ax2.plot(drawdown.index, drawdown, color='red', linewidth=1)
            ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)

            # 3. Rolling Returns
            ax3 = plt.subplot(3, 2, 3)
            rolling_returns = returns.rolling(window=30).mean() * 100
            ax3.plot(rolling_returns.index, rolling_returns, linewidth=2)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('30-Period Rolling Returns', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Return (%)')
            ax3.grid(True, alpha=0.3)

            # 4. Return Distribution
            ax4 = plt.subplot(3, 2, 4)
            ax4.hist(returns * 100, bins=50, alpha=0.7, color='blue', density=True)
            ax4.axvline(returns.mean() * 100, color='red', linestyle='--', label=f'Mean: {returns.mean()*100:.2f}%')
            ax4.set_title('Return Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Return (%)')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 5. Rolling Sharpe Ratio
            ax5 = plt.subplot(3, 2, 5)
            rolling_sharpe = returns.rolling(window=60).mean() / returns.rolling(window=60).std()
            ax5.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax5.set_title('60-Period Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
            ax5.set_ylabel('Sharpe Ratio')
            ax5.grid(True, alpha=0.3)

            # 6. Monthly Returns Heatmap
            ax6 = plt.subplot(3, 2, 6)
            monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum()

            if len(monthly_returns) > 0:
                monthly_df = monthly_returns.unstack(level=1)
                monthly_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                sns.heatmap(monthly_df * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                           center=0, ax=ax6, cbar_kws={'label': 'Return (%)'})
                ax6.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            else:
                ax6.text(0.5, 0.5, 'Insufficient data for heatmap',
                        ha='center', va='center', transform=ax6.transAxes)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Performance plot saved to {save_path}")

            plt.show()
            return save_path or "plot_displayed"

        except Exception as e:
            logger.error(f"Performance plotting failed: {e}")
            return ""


    def save_performance_report(self, symbol: str, timeframe: str, version: str,
                              analysis_results: Dict[str, Any]) -> str:
        """Save comprehensive performance report"""
        try:
            results_path = config.get_processed_path(symbol, timeframe, version)
            results_path.mkdir(parents=True, exist_ok=True)

            # Save JSON report
            json_file = results_path / 'performance_analysis.json'
            with open(json_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)

            # Generate text summary
            summary_file = results_path / 'performance_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Performance Analysis for {symbol}_{timeframe}_{version}\n")
                f.write("=" * 60 + "\n\n")

                # Summary
                summary = analysis_results.get('summary', {})
                f.write(f"Overall Grade: {summary.get('overall_grade', 'N/A')}\n\n")

                # Return metrics
                return_metrics = analysis_results.get('return_metrics', {})
                f.write("Return Metrics:\n")
                f.write(f"  Total Return: {return_metrics.get('total_return', 0):.2%}\n")
                f.write(f"  Annual Return: {return_metrics.get('annual_return', 0):.2%}\n")
                f.write(f"  Win Rate: {return_metrics.get('win_rate', 0):.2%}\n\n")

                # Risk metrics
                risk_metrics = analysis_results.get('risk_metrics', {})
                f.write("Risk Metrics:\n")
                f.write(f"  Annual Volatility: {risk_metrics.get('annual_volatility', 0):.2%}\n")
                f.write(f"  Max Drawdown: {analysis_results.get('drawdown_analysis', {}).get('max_drawdown', 0):.2%}\n")
                f.write(f"  VaR (95%): {risk_metrics.get('var_95', 0):.2%}\n\n")

                # Risk-adjusted metrics
                risk_adjusted = analysis_results.get('risk_adjusted', {})
                f.write("Risk-Adjusted Metrics:\n")
                f.write(f"  Sharpe Ratio: {risk_adjusted.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"  Sortino Ratio: {risk_adjusted.get('sortino_ratio', 0):.3f}\n")
                f.write(f"  Calmar Ratio: {analysis_results.get('advanced_metrics', {}).get('calmar_ratio', 0):.3f}\n\n")

                # Highlights and warnings
                highlights = summary.get('highlights', [])
                if highlights:
                    f.write("Key Highlights:\n")
                    for highlight in highlights:
                        f.write(f"  ✓ {highlight}\n")
                    f.write("\n")

                warnings = summary.get('warnings', [])
                if warnings:
                    f.write("Risk Warnings:\n")
                    for warning in warnings:
                        f.write(f"  ⚠ {warning}\n")
                    f.write("\n")

                # Recommendation
                recommendation = summary.get('recommendation', '')
                if recommendation:
                    f.write(f"Recommendation: {recommendation}\n")

            logger.info(f"Performance report saved to {results_path}")
            return str(results_path)

        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
            return ""

# Convenience functions


def analyze_backtest_performance(returns: pd.Series,
                               portfolio_values: pd.Series = None,
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenient function for backtest performance analysis"""
    analyzer = PerformanceAnalyzer()
    return analyzer.analyze_performance(returns, portfolio_values, metadata=metadata)


def compare_strategies(strategy_returns: Dict[str, pd.Series]) -> Dict[str, Dict[str, Any]]:
    """Compare multiple strategies"""
    analyzer = PerformanceAnalyzer()
    results = {}

    for name, returns in strategy_returns.items():
        results[name] = analyzer.analyze_performance(returns, metadata={'strategy_name': name})

    return results

# Usage example
if __name__ == "__main__":
    # Create sample returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')

    # Generate realistic return series (slightly positive bias)
    returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),  # Daily returns
        index=dates
    )

    # Test performance analysis
    analyzer = PerformanceAnalyzer()
    results = analyzer.analyze_performance(
        returns,
        metadata={'symbol': 'BTCUSDT', 'strategy': 'Test Strategy'}
    )

    print("Performance Analysis Results:")
    print(f"Total Return: {results['return_metrics']['total_return']:.2%}")
    print(f"Annual Return: {results['return_metrics']['annual_return']:.2%}")
    print(f"Sharpe Ratio: {results['risk_adjusted']['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['drawdown_analysis']['max_drawdown']:.2%}")
    print(f"Overall Grade: {results['summary']['overall_grade']}")

    print("Performance analyzer example completed!")
