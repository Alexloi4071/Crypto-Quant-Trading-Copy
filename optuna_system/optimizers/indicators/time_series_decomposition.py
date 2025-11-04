# -*- coding: utf-8 -*-
"""
时间序列分解 - 完整实现
使用statsmodels进行专业级时间序列分解
"""
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy import signal

warnings.filterwarnings('ignore')


class TimeSeriesDecomposition:
    """
    时间序列分解：使用STL和seasonal_decompose进行专业级分解
    
    特性：
    1. 使用statsmodels STL（Seasonal and Trend decomposition using Loess）
    2. 自动周期检测（ACF/PACF）
    3. 多周期分解
    4. 傅里叶分析
    5. 趋势强度和季节性强度指标
    """
    
    def __init__(self):
        self.default_periods = [7, 14, 21, 30]  # 默认周期（天）
    
    def decompose(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """完整的时间序列分解"""
        if ohlcv.empty or len(ohlcv) < 50:
            return pd.DataFrame(index=ohlcv.index)
        
        close = ohlcv['close'].copy()
        features = pd.DataFrame(index=ohlcv.index)
        
        try:
            # 1. 自动检测显著周期
            detected_periods = self.detect_seasonality(close)
            if not detected_periods:
                detected_periods = self.default_periods
            
            # 使用第一个检测到的周期进行主要分解
            primary_period = detected_periods[0] if detected_periods else 14
            
            # 2. STL分解（如果数据足够长）
            if len(close) >= primary_period * 2:
                stl_features = self.stl_decompose(close, period=primary_period)
                for key, value in stl_features.items():
                    features[f'stl_{key}'] = value
            
            # 3. 经典季节性分解（作为补充）
            if len(close) >= primary_period * 2:
                seasonal_features = self.seasonal_decompose_additive(close, period=primary_period)
                for key, value in seasonal_features.items():
                    features[f'seasonal_{key}'] = value
            
            # 4. 多周期分析（最多3个周期）
            for i, period in enumerate(detected_periods[:3], 1):
                if period >= 7 and period <= len(close) // 3:
                    multi_features = self.multi_period_analysis(close, period=period)
                    for key, value in multi_features.items():
                        features[f'period{i}_{key}'] = value
            
            # 5. 傅里叶频谱分析
            fourier_features = self.calculate_fourier_components(close, n_components=5)
            for key, value in fourier_features.items():
                features[f'fourier_{key}'] = value
            
            # 6. 趋势和周期性强度指标
            strength_features = self.calculate_strength_indicators(close)
            for key, value in strength_features.items():
                features[f'strength_{key}'] = value
            
        except Exception as e:
            # 如果出错，返回基础特征
            warnings.warn(f"Time series decomposition failed: {e}, using fallback")
            features = self._fallback_decompose(close)
        
        # 清理无效值
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 使用forward fill填充NaN
        features = features.ffill().fillna(0)
        
        return features
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """calculate_all方法（别名：decompose）"""
        return self.decompose(ohlcv)
    
    def detect_seasonality(self, close: pd.Series, max_lag: int = 100) -> List[int]:
        """
        自动检测季节性周期
        
        使用ACF（自相关函数）找出显著的周期性
        
        Returns:
            List[int]: 检测到的周期列表，按显著性排序
        """
        try:
            # 导入statsmodels
            try:
                from statsmodels.tsa.stattools import acf
            except ImportError:
                warnings.warn("statsmodels not available, using default periods")
                return self.default_periods
            
            # 去除趋势（简单差分）
            close_diff = close.diff().dropna()
            
            if len(close_diff) < 50:
                return self.default_periods
            
            # 计算自相关
            max_lag = min(max_lag, len(close_diff) // 3)
            acf_values = acf(close_diff, nlags=max_lag, fft=True)
            
            # 找出局部最大值（可能的周期）
            peaks = []
            for i in range(2, len(acf_values)-1):
                # 局部最大值且显著（>0.2）
                if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                    if acf_values[i] > 0.2:  # 显著性阈值
                        peaks.append((i, acf_values[i]))
            
            # 按相关性排序
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前5个周期
            detected_periods = [p[0] for p in peaks[:5]]
            
            # 如果没有检测到，返回默认
            if not detected_periods:
                return self.default_periods
            
            return detected_periods
            
        except Exception as e:
            warnings.warn(f"Seasonality detection failed: {e}")
            return self.default_periods
    
    def stl_decompose(self, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        使用STL（Seasonal and Trend decomposition using Loess）进行分解
        
        STL的优势：
        - 更稳健，对异常值不敏感
        - 可以处理变化的季节性
        - Loess平滑更灵活
        """
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            warnings.warn("statsmodels not available for STL, using fallback")
            return self._simple_decompose(close, period)
        
        try:
            # 确保周期合理
            period = max(7, min(period, len(close) // 3))
            
            # STL分解
            # seasonal: 季节性周期，必须是奇数
            seasonal = period if period % 2 == 1 else period + 1
            
            stl = STL(close, seasonal=seasonal, robust=True)
            result = stl.fit()
            
            # 计算去季节性和去趋势的序列
            deseasonalized = close - result.seasonal
            detrended = close - result.trend
            
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid,
                'deseasonalized': deseasonalized,
                'detrended': detrended,
                'trend_slope': result.trend.diff(),
                'seasonal_amplitude': result.seasonal.rolling(period).std()
            }
            
        except Exception as e:
            warnings.warn(f"STL decomposition failed: {e}, using fallback")
            return self._simple_decompose(close, period)
    
    def seasonal_decompose_additive(self, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        经典的加法季节性分解
        
        使用statsmodels的seasonal_decompose
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            return self._simple_decompose(close, period)
        
        try:
            # 确保周期合理
            period = max(7, min(period, len(close) // 3))
            
            # 季节性分解（加法模型）
            result = seasonal_decompose(close, model='additive', period=period, extrapolate_trend='freq')
            
            return {
                'classic_trend': result.trend,
                'classic_seasonal': result.seasonal,
                'classic_residual': result.resid
            }
            
        except Exception as e:
            warnings.warn(f"Seasonal decompose failed: {e}")
            return self._simple_decompose(close, period)
    
    def multi_period_analysis(self, close: pd.Series, period: int) -> Dict[str, pd.Series]:
        """
        多周期分析
        
        针对特定周期计算相关指标
        """
        features = {}
        
        try:
            # 周期性移动平均
            features['ma'] = close.rolling(period).mean()
            
            # 周期性标准差
            features['std'] = close.rolling(period).std()
            
            # 周期性范围
            features['range'] = close.rolling(period).max() - close.rolling(period).min()
            
            # 周期性动量
            features['momentum'] = close - close.shift(period)
            
            # 周期性ROC
            features['roc'] = close.pct_change(period)
            
            # 周期性Z-score
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features['zscore'] = (close - ma) / std
            
        except Exception as e:
            warnings.warn(f"Multi-period analysis failed: {e}")
        
        return features
    
    def calculate_fourier_components(self, close: pd.Series, n_components: int = 5) -> Dict[str, pd.Series]:
        """
        傅里叶变换提取周期成分（改进版）
        
        提取最显著的周期性频率成分
        """
        features = {}
        
        try:
            # 去NaN
            close_clean = close.dropna()
            if len(close_clean) < 50:
                return {f'cycle_{i}': pd.Series(0, index=close.index) for i in range(1, n_components+1)}
            
            # 去趋势（使用差分）
            close_diff = close_clean.diff().dropna()
            
            # FFT
            fft_vals = np.fft.fft(close_diff.values)
            fft_freq = np.fft.fftfreq(len(close_diff))
            
            # 功率谱
            power = np.abs(fft_vals) ** 2
            positive_freq_idx = fft_freq > 0
            
            # 找到最强的n个频率
            positive_power = power[positive_freq_idx]
            positive_freq = fft_freq[positive_freq_idx]
            
            top_idx = np.argsort(positive_power)[-n_components:]
            dominant_freqs = positive_freq[top_idx]
            dominant_periods = 1 / dominant_freqs
            dominant_powers = positive_power[top_idx]
            
            # 生成特征
            for i, (period, power_val) in enumerate(zip(dominant_periods, dominant_powers), 1):
                if 2 < period < len(close_clean) / 2:
                    # 正弦和余弦成分
                    t = np.arange(len(close))
                    features[f'cycle_{i}_sin'] = pd.Series(np.sin(2 * np.pi * t / period), index=close.index)
                    features[f'cycle_{i}_cos'] = pd.Series(np.cos(2 * np.pi * t / period), index=close.index)
                    features[f'cycle_{i}_period'] = period
                    features[f'cycle_{i}_power'] = power_val / len(close_diff)  # 归一化功率
                else:
                    features[f'cycle_{i}_sin'] = pd.Series(0, index=close.index)
                    features[f'cycle_{i}_cos'] = pd.Series(0, index=close.index)
                    features[f'cycle_{i}_period'] = 0
                    features[f'cycle_{i}_power'] = 0
                    
        except Exception as e:
            warnings.warn(f"Fourier analysis failed: {e}")
            for i in range(1, n_components+1):
                features[f'cycle_{i}_sin'] = pd.Series(0, index=close.index)
                features[f'cycle_{i}_cos'] = pd.Series(0, index=close.index)
        
        return features
    
    def calculate_strength_indicators(self, close: pd.Series) -> Dict[str, pd.Series]:
        """
        计算趋势强度和季节性强度指标
        
        Returns:
            Dict包含：
            - trend_strength: 趋势强度 (0-1)
            - seasonal_strength: 季节性强度 (0-1)
            - noise_ratio: 噪声比例 (0-1)
        """
        features = {}
        
        try:
            # 使用简单分解计算强度
            period = 14
            
            # 趋势（移动平均）
            trend = close.rolling(period*2, center=True).mean()
            
            # 去趋势
            detrended = close - trend
            
            # 季节性（周期性平均）
            seasonal = detrended.rolling(period).mean()
            
            # 残差
            residual = close - trend - seasonal
            
            # 趋势强度：1 - Var(residual) / Var(detrended)
            var_detrended = detrended.rolling(50).var()
            var_residual = residual.rolling(50).var()
            trend_strength = (1 - var_residual / var_detrended).fillna(0)
            trend_strength = trend_strength.clip(0, 1)
            
            # 季节性强度：Var(seasonal) / Var(detrended)
            var_seasonal = seasonal.rolling(50).var()
            seasonal_strength = (var_seasonal / var_detrended).fillna(0)
            seasonal_strength = seasonal_strength.clip(0, 1)
            
            # 噪声比例
            var_close = close.rolling(50).var()
            noise_ratio = (var_residual / var_close).fillna(0)
            noise_ratio = noise_ratio.clip(0, 1)
            
            features['trend_strength'] = trend_strength
            features['seasonal_strength'] = seasonal_strength
            features['noise_ratio'] = noise_ratio
            
            # 额外的强度指标
            features['trend_consistency'] = trend.diff().rolling(20).apply(
                lambda x: (np.sign(x) == np.sign(x.iloc[-1])).sum() / len(x)
            ).fillna(0.5)
            
        except Exception as e:
            warnings.warn(f"Strength indicator calculation failed: {e}")
            features['trend_strength'] = pd.Series(0, index=close.index)
            features['seasonal_strength'] = pd.Series(0, index=close.index)
            features['noise_ratio'] = pd.Series(0, index=close.index)
        
        return features
    
    def _simple_decompose(self, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """简化分解（后备方法）"""
        # 趋势（移动平均）
        trend = close.rolling(period, center=True).mean()
        
        # 去趋势
        detrended = close - trend
        
        # 季节性（简化：使用周期性平均）
        seasonal = detrended.rolling(period).mean()
        
        # 残差
        residual = close - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _fallback_decompose(self, close: pd.Series) -> pd.DataFrame:
        """后备分解方法"""
        features = pd.DataFrame(index=close.index)
        
        # 简单趋势
        features['trend'] = close.rolling(20).mean()
        features['seasonal'] = close - features['trend']
        features['residual'] = close - features['trend'] - features['seasonal']
        
        return features


def main():
    """测试时间序列分解"""
    print("="*60)
    print("时间序列分解测试")
    print("="*60)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    
    # 模拟价格序列（带趋势和季节性）
    t = np.arange(len(dates))
    trend = 100 + 0.1 * t  # 上升趋势
    seasonal = 5 * np.sin(2 * np.pi * t / 7)  # 7天周期
    noise = np.random.randn(len(dates)) * 2  # 噪声
    close_prices = trend + seasonal + noise
    
    ohlcv = pd.DataFrame({
        'close': close_prices,
        'open': close_prices * 0.99,
        'high': close_prices * 1.01,
        'low': close_prices * 0.98,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # 执行分解
    tsd = TimeSeriesDecomposition()
    features = tsd.decompose(ohlcv)
    
    print(f"\n✅ 生成 {features.shape[1]} 个时间序列分解特征")
    print(f"数据点: {features.shape[0]}")
    print(f"\n特征列表:")
    for i, col in enumerate(features.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 检查特征质量
    print(f"\n特征质量检查:")
    print(f"  NaN数量: {features.isna().sum().sum()}")
    print(f"  Inf数量: {np.isinf(features).sum().sum()}")
    print(f"  零值特征: {(features == 0).all().sum()}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
