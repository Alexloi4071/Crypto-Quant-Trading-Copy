# -*- coding: utf-8 -*-
"""
自适应参数优化器
基于历史数据自动优化每个时框的技术指标参数
使用信息系数(IC)和夏普率作为优化目标
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score

warnings.filterwarnings('ignore')


class AdaptiveParameterOptimizer:
    """基于历史数据的参数优化器"""
    
    def __init__(self, cache_dir: str = "config", logger: Optional[logging.Logger] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # 参数搜索范围（基础范围）
        self.parameter_search_space = {
            # RSI参数
            'rsi_periods': {
                'min': 3, 'max': 50, 'step': 1,
                'default': [7, 14, 21, 28]
            },
            # 移动平均参数
            'ma_periods': {
                'min': 3, 'max': 300, 'step': 1,
                'default': [5, 10, 20, 50, 100, 200]
            },
            # MACD参数
            'macd_fast': {
                'min': 5, 'max': 30, 'step': 1,
                'default': [8, 12, 17]
            },
            'macd_slow': {
                'min': 15, 'max': 70, 'step': 1,
                'default': [22, 26, 35, 45]
            },
            'macd_signal': {
                'min': 3, 'max': 20, 'step': 1,
                'default': [5, 9, 13]
            },
            # 布林带参数
            'bb_periods': {
                'min': 5, 'max': 100, 'step': 1,
                'default': [10, 20, 30, 50]
            },
            'bb_std': {
                'min': 1.0, 'max': 3.5, 'step': 0.5,
                'default': [1.5, 2.0, 2.5]
            },
            # ATR参数
            'atr_periods': {
                'min': 3, 'max': 100, 'step': 1,
                'default': [7, 14, 21, 30]
            },
            # Stochastic参数
            'stoch_k': {
                'min': 5, 'max': 50, 'step': 1,
                'default': [9, 14, 21]
            },
            'stoch_d': {
                'min': 2, 'max': 20, 'step': 1,
                'default': [3, 5, 9]
            },
            # ADX参数
            'adx_periods': {
                'min': 7, 'max': 50, 'step': 1,
                'default': [14, 21, 28]
            },
            # CCI参数
            'cci_periods': {
                'min': 7, 'max': 60, 'step': 1,
                'default': [14, 20, 34]
            },
            # 成交量MA参数
            'volume_ma_periods': {
                'min': 5, 'max': 100, 'step': 1,
                'default': [10, 20, 50]
            },
            # Fibonacci参数
            'fibo_periods': {
                'min': 13, 'max': 377, 'step': 1,
                'default': [21, 34, 55, 89, 144]
            }
        }
        
    def load_or_optimize(self, timeframe: str, ohlcv: Optional[pd.DataFrame] = None,
                        force_reoptimize: bool = False) -> Dict:
        """加载缓存参数或重新优化"""
        cache_file = self.cache_dir / f"optimized_windows_{timeframe}.json"
        
        # 如果缓存存在且不强制重新优化，则加载
        if cache_file.exists() and not force_reoptimize:
            self.logger.info(f"Loading cached parameters for {timeframe} from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # 否则需要优化
        if ohlcv is None:
            self.logger.warning(f"No OHLCV data provided for {timeframe}, using default parameters")
            return self._get_default_parameters(timeframe)
        
        self.logger.info(f"Optimizing parameters for {timeframe} using historical data")
        optimized_params = self.optimize_windows_for_timeframe(ohlcv, timeframe)
        
        # 保存缓存
        with open(cache_file, 'w') as f:
            json.dump(optimized_params, f, indent=2)
        self.logger.info(f"Saved optimized parameters to {cache_file}")
        
        return optimized_params
    
    def optimize_windows_for_timeframe(self, ohlcv: pd.DataFrame, timeframe: str) -> Dict:
        """
        优化时框参数
        
        使用滚动窗口分析，基于信息系数(IC)和夏普率选择最佳参数
        
        Args:
            ohlcv: OHLCV数据
            timeframe: 时框 (15m, 1h, 4h, 1D)
            
        Returns:
            优化后的参数字典
        """
        self.logger.info(f"Starting parameter optimization for {timeframe}")
        self.logger.info(f"Data shape: {ohlcv.shape}, Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
        
        # 时框特性
        timeframe_properties = self._get_timeframe_properties(timeframe)
        
        optimized_params = {
            'timeframe': timeframe,
            'optimization_date': pd.Timestamp.now().isoformat(),
            'data_points': len(ohlcv),
            'date_range': [str(ohlcv.index[0]), str(ohlcv.index[-1])]
        }
        
        # 并行优化各参数组
        param_groups = [
            ('rsi_periods', self._optimize_rsi_periods),
            ('ma_periods', self._optimize_ma_periods),
            ('macd', self._optimize_macd_params),
            ('bb_periods', self._optimize_bb_periods),
            ('atr_periods', self._optimize_atr_periods),
            ('stoch', self._optimize_stoch_params),
            ('adx_periods', self._optimize_adx_periods),
            ('cci_periods', self._optimize_cci_periods),
            ('volume_ma_periods', self._optimize_volume_ma_periods),
            ('fibo_periods', self._optimize_fibo_periods)
        ]
        
        # 串行优化（避免内存问题）
        for param_name, optimize_func in param_groups:
            try:
                self.logger.info(f"Optimizing {param_name}...")
                result = optimize_func(ohlcv, timeframe_properties)
                optimized_params[param_name] = result
                self.logger.info(f"✓ {param_name}: {result}")
            except Exception as e:
                self.logger.error(f"✗ Failed to optimize {param_name}: {e}")
                # 使用默认值
                default_config = self.parameter_search_space.get(param_name, {})
                optimized_params[param_name] = default_config.get('default', [])
        
        return optimized_params
    
    def _get_timeframe_properties(self, timeframe: str) -> Dict:
        """获取时框特性"""
        properties = {
            '15m': {
                'samples_per_day': 96,
                'samples_per_week': 672,
                'samples_per_month': 2880,
                'noise_level': 'high',
                'min_window': 3,
                'max_window_ratio': 0.3  # 最大窗口为数据长度的30%
            },
            '1h': {
                'samples_per_day': 24,
                'samples_per_week': 168,
                'samples_per_month': 720,
                'noise_level': 'medium',
                'min_window': 5,
                'max_window_ratio': 0.25
            },
            '4h': {
                'samples_per_day': 6,
                'samples_per_week': 42,
                'samples_per_month': 180,
                'noise_level': 'low',
                'min_window': 7,
                'max_window_ratio': 0.2
            },
            '1D': {
                'samples_per_day': 1,
                'samples_per_week': 7,
                'samples_per_month': 30,
                'noise_level': 'very_low',
                'min_window': 10,
                'max_window_ratio': 0.15
            }
        }
        return properties.get(timeframe, properties['1h'])
    
    def _calculate_ic(self, indicator_values: pd.Series, future_returns: pd.Series) -> float:
        """
        计算信息系数(IC) - 使用Spearman相关系数
        
        IC衡量指标对未来收益的预测能力
        """
        # 移除NaN
        valid_mask = ~(indicator_values.isna() | future_returns.isna())
        if valid_mask.sum() < 30:  # 至少需要30个有效样本
            return 0.0
        
        try:
            ic, p_value = spearmanr(indicator_values[valid_mask], future_returns[valid_mask])
            # 考虑p值，如果不显著则降低IC
            if p_value > 0.05:
                ic *= 0.5
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    def _calculate_sharpe_ratio(self, indicator_values: pd.Series, returns: pd.Series,
                               periods_per_year: int = 252) -> float:
        """
        计算基于指标的策略夏普率
        
        简单策略：指标上涨时做多，下跌时平仓
        """
        # 生成信号
        signal = np.sign(indicator_values.diff())
        signal = signal.shift(1)  # 避免look-ahead bias
        
        # 计算策略收益
        strategy_returns = signal * returns
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) < 30 or strategy_returns.std() == 0:
            return 0.0
        
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(periods_per_year)
        return sharpe if not np.isnan(sharpe) else 0.0
    
    def _optimize_rsi_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化RSI周期参数"""
        close = ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)  # 预测未来5期收益
        
        search_space = self.parameter_search_space['rsi_periods']
        min_period = max(search_space['min'], props['min_window'])
        max_period = min(search_space['max'], int(len(ohlcv) * props['max_window_ratio']))
        
        candidates = range(min_period, max_period + 1, search_space['step'])
        scores = {}
        
        for period in candidates:
            if period < 3:
                continue
            try:
                # 计算RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = -delta.where(delta < 0, 0).rolling(period).mean()
                rs = gain / (loss + 1e-9)
                rsi = 100 - (100 / (1 + rs))
                
                # 计算IC和Sharpe
                ic = self._calculate_ic(rsi, future_returns)
                sharpe = self._calculate_sharpe_ratio(rsi, returns)
                
                # 综合得分
                scores[period] = abs(ic) * 0.6 + max(0, sharpe) * 0.4
            except:
                scores[period] = 0.0
        
        # 选择top 4个参数
        sorted_periods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_periods = [p for p, s in sorted_periods[:4] if s > 0]
        
        # 如果没有找到好的参数，使用默认值
        if not best_periods:
            best_periods = search_space['default']
        
        return sorted(best_periods)
    
    def _optimize_ma_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化移动平均周期参数"""
        close = ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)
        
        search_space = self.parameter_search_space['ma_periods']
        min_period = max(search_space['min'], props['min_window'])
        max_period = min(search_space['max'], int(len(ohlcv) * props['max_window_ratio']))
        
        # 使用更稀疏的采样以加快速度
        candidates = list(range(min_period, 30, 2)) + \
                    list(range(30, 100, 5)) + \
                    list(range(100, max_period + 1, 10))
        
        scores = {}
        
        for period in candidates:
            try:
                # 计算SMA
                sma = close.rolling(period).mean()
                
                # 计算IC和Sharpe
                ic = self._calculate_ic(sma, future_returns)
                sharpe = self._calculate_sharpe_ratio(sma, returns)
                
                scores[period] = abs(ic) * 0.5 + max(0, sharpe) * 0.5
            except:
                scores[period] = 0.0
        
        # 选择top 6个参数（覆盖短中长期）
        sorted_periods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_periods = [p for p, s in sorted_periods[:6] if s > 0]
        
        if not best_periods:
            best_periods = search_space['default']
        
        return sorted(best_periods)
    
    def _optimize_macd_params(self, ohlcv: pd.DataFrame, props: Dict) -> Dict:
        """优化MACD参数组合"""
        close = ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)
        
        # MACD参数组合候选
        fast_candidates = [8, 10, 12, 15, 17]
        slow_candidates = [22, 26, 30, 35, 39]
        signal_candidates = [5, 7, 9, 11, 13]
        
        best_score = -np.inf
        best_combo = {'fast': 12, 'slow': 26, 'signal': 9}
        
        # 评估所有组合（限制数量以加快速度）
        for fast in fast_candidates:
            for slow in slow_candidates:
                if slow <= fast:
                    continue
                for signal in signal_candidates:
                    try:
                        # 计算MACD
                        ema_fast = close.ewm(span=fast).mean()
                        ema_slow = close.ewm(span=slow).mean()
                        macd = ema_fast - ema_slow
                        macd_signal = macd.ewm(span=signal).mean()
                        macd_hist = macd - macd_signal
                        
                        # 计算得分
                        ic = self._calculate_ic(macd_hist, future_returns)
                        sharpe = self._calculate_sharpe_ratio(macd_hist, returns)
                        
                        score = abs(ic) * 0.6 + max(0, sharpe) * 0.4
                        
                        if score > best_score:
                            best_score = score
                            best_combo = {'fast': fast, 'slow': slow, 'signal': signal}
                    except:
                        continue
        
        return best_combo
    
    def _optimize_bb_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化布林带周期参数"""
        close = ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)
        
        search_space = self.parameter_search_space['bb_periods']
        min_period = max(search_space['min'], props['min_window'])
        max_period = min(search_space['max'], int(len(ohlcv) * props['max_window_ratio']))
        
        candidates = range(min_period, max_period + 1, 3)
        scores = {}
        
        for period in candidates:
            try:
                # 计算布林带位置
                sma = close.rolling(period).mean()
                std = close.rolling(period).std()
                bb_position = (close - (sma - 2*std)) / (4*std + 1e-9)
                
                # 计算得分
                ic = self._calculate_ic(bb_position, future_returns)
                sharpe = self._calculate_sharpe_ratio(bb_position, returns)
                
                scores[period] = abs(ic) * 0.6 + max(0, sharpe) * 0.4
            except:
                scores[period] = 0.0
        
        sorted_periods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_periods = [p for p, s in sorted_periods[:4] if s > 0]
        
        if not best_periods:
            best_periods = search_space['default']
        
        return sorted(best_periods)
    
    def _optimize_atr_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化ATR周期参数"""
        high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)
        
        search_space = self.parameter_search_space['atr_periods']
        min_period = max(search_space['min'], props['min_window'])
        max_period = min(search_space['max'], int(len(ohlcv) * props['max_window_ratio']))
        
        candidates = range(min_period, max_period + 1, 3)
        scores = {}
        
        for period in candidates:
            try:
                # 计算ATR
                tr1 = high - low
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(period).mean()
                
                # ATR与波动率预测
                future_vol = returns.rolling(5).std().shift(-5)
                
                ic = self._calculate_ic(atr, future_vol)
                
                scores[period] = abs(ic)
            except:
                scores[period] = 0.0
        
        sorted_periods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_periods = [p for p, s in sorted_periods[:4] if s > 0]
        
        if not best_periods:
            best_periods = search_space['default']
        
        return sorted(best_periods)
    
    def _optimize_stoch_params(self, ohlcv: pd.DataFrame, props: Dict) -> Dict:
        """优化随机指标参数"""
        high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)
        
        k_candidates = [9, 14, 21, 28]
        d_candidates = [3, 5, 9]
        
        best_score = -np.inf
        best_combo = {'k': 14, 'd': 3}
        
        for k in k_candidates:
            for d in d_candidates:
                try:
                    # 计算Stochastic
                    lowest_low = low.rolling(k).min()
                    highest_high = high.rolling(k).max()
                    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
                    stoch_d = stoch_k.rolling(d).mean()
                    
                    ic = self._calculate_ic(stoch_d, future_returns)
                    sharpe = self._calculate_sharpe_ratio(stoch_d, returns)
                    
                    score = abs(ic) * 0.6 + max(0, sharpe) * 0.4
                    
                    if score > best_score:
                        best_score = score
                        best_combo = {'k': k, 'd': d}
                except:
                    continue
        
        return best_combo
    
    def _optimize_adx_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化ADX周期参数（简化版）"""
        # ADX计算复杂，使用默认值并根据时框调整
        search_space = self.parameter_search_space['adx_periods']
        default_periods = search_space['default']
        
        # 根据时框特性调整
        if props['noise_level'] == 'high':
            return [p - 3 for p in default_periods if p > 3]
        elif props['noise_level'] == 'very_low':
            return [p + 5 for p in default_periods]
        else:
            return default_periods
    
    def _optimize_cci_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化CCI周期参数"""
        high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']
        returns = close.pct_change()
        future_returns = returns.shift(-5)
        
        search_space = self.parameter_search_space['cci_periods']
        min_period = max(search_space['min'], props['min_window'])
        max_period = min(search_space['max'], int(len(ohlcv) * props['max_window_ratio']))
        
        candidates = range(min_period, max_period + 1, 3)
        scores = {}
        
        for period in candidates:
            try:
                # 计算CCI
                tp = (high + low + close) / 3
                sma_tp = tp.rolling(period).mean()
                mad = (tp - sma_tp).abs().rolling(period).mean()
                cci = (tp - sma_tp) / (0.015 * mad + 1e-9)
                
                ic = self._calculate_ic(cci, future_returns)
                sharpe = self._calculate_sharpe_ratio(cci, returns)
                
                scores[period] = abs(ic) * 0.6 + max(0, sharpe) * 0.4
            except:
                scores[period] = 0.0
        
        sorted_periods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_periods = [p for p, s in sorted_periods[:3] if s > 0]
        
        if not best_periods:
            best_periods = search_space['default']
        
        return sorted(best_periods)
    
    def _optimize_volume_ma_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化成交量MA周期参数"""
        volume = ohlcv['volume']
        close = ohlcv['close']
        returns = close.pct_change()
        
        search_space = self.parameter_search_space['volume_ma_periods']
        min_period = max(search_space['min'], props['min_window'])
        max_period = min(search_space['max'], int(len(ohlcv) * props['max_window_ratio']))
        
        candidates = range(min_period, max_period + 1, 5)
        scores = {}
        
        for period in candidates:
            try:
                # 计算成交量比率
                vol_ma = volume.rolling(period).mean()
                vol_ratio = volume / (vol_ma + 1e-9)
                
                # 成交量异常与未来收益的关系
                future_abs_returns = returns.abs().shift(-5)
                
                ic = self._calculate_ic(vol_ratio, future_abs_returns)
                
                scores[period] = abs(ic)
            except:
                scores[period] = 0.0
        
        sorted_periods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_periods = [p for p, s in sorted_periods[:3] if s > 0]
        
        if not best_periods:
            best_periods = search_space['default']
        
        return sorted(best_periods)
    
    def _optimize_fibo_periods(self, ohlcv: pd.DataFrame, props: Dict) -> List[int]:
        """优化Fibonacci周期参数"""
        # Fibonacci数列作为候选
        fibo_sequence = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        
        search_space = self.parameter_search_space['fibo_periods']
        max_period = int(len(ohlcv) * props['max_window_ratio'])
        
        # 选择在合理范围内的Fibonacci数
        valid_fibos = [f for f in fibo_sequence 
                       if search_space['min'] <= f <= min(search_space['max'], max_period)]
        
        if len(valid_fibos) < 3:
            return search_space['default']
        
        return valid_fibos[:5]  # 返回前5个
    
    def _get_default_parameters(self, timeframe: str) -> Dict:
        """获取默认参数（当无法优化时使用）"""
        return {
            'timeframe': timeframe,
            'rsi_periods': self.parameter_search_space['rsi_periods']['default'],
            'ma_periods': self.parameter_search_space['ma_periods']['default'],
            'macd': {
                'fast': 12,
                'slow': 26,
                'signal': 9
            },
            'bb_periods': self.parameter_search_space['bb_periods']['default'],
            'atr_periods': self.parameter_search_space['atr_periods']['default'],
            'stoch': {
                'k': 14,
                'd': 3
            },
            'adx_periods': self.parameter_search_space['adx_periods']['default'],
            'cci_periods': self.parameter_search_space['cci_periods']['default'],
            'volume_ma_periods': self.parameter_search_space['volume_ma_periods']['default'],
            'fibo_periods': self.parameter_search_space['fibo_periods']['default']
        }


def main():
    """测试优化器"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # 生成测试数据
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='15min')
    np.random.seed(42)
    
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
    ohlcv = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.002),
        'low': close_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.002),
        'close': close_prices,
        'volume': np.abs(np.random.randn(len(dates))) * 1000000
    }, index=dates)
    
    # 测试优化器
    optimizer = AdaptiveParameterOptimizer()
    
    print("Testing parameter optimization for 15m timeframe...")
    params_15m = optimizer.optimize_windows_for_timeframe(ohlcv, '15m')
    
    print("\nOptimized parameters for 15m:")
    for key, value in params_15m.items():
        if key not in ['timeframe', 'optimization_date', 'data_points', 'date_range']:
            print(f"  {key}: {value}")


if __name__ == '__main__':
    main()

