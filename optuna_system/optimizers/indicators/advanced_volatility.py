# -*- coding: utf-8 -*-
"""
高级波动率模型完整实现
包含：GARCH(1,1), EGARCH, GJR-GARCH, Realized Volatility, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang
"""

import warnings
from typing import Dict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class AdvancedVolatility:
    """高级波动率模型完整实现"""
    
    def __init__(self):
        pass
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        open_p, high, low, close = ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close']
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. GARCH(1,1) - 完整实现
        for window in [20, 50]:
            garch_features = self.calculate_garch(close, window=window)
            for key, value in garch_features.items():
                features[f'garch_{window}_{key}'] = value
        
        # 2. EGARCH - 指数GARCH
        for window in [20, 50]:
            egarch_features = self.calculate_egarch(close, window=window)
            for key, value in egarch_features.items():
                features[f'egarch_{window}_{key}'] = value
        
        # 3. GJR-GARCH - 非对称GARCH
        for window in [20, 50]:
            gjr_features = self.calculate_gjr_garch(close, window=window)
            for key, value in gjr_features.items():
                features[f'gjr_{window}_{key}'] = value
        
        # 4. Realized Volatility（已实现波动率）
        for window in [5, 10, 20]:
            features[f'realized_vol_{window}'] = self.calculate_realized_volatility(close, window)
        
        # 5. Parkinson波动率估计
        for window in [10, 20, 50]:
            features[f'parkinson_{window}'] = self.calculate_parkinson(high, low, window)
        
        # 6. Garman-Klass波动率估计
        for window in [10, 20, 50]:
            features[f'garman_klass_{window}'] = self.calculate_garman_klass(open_p, high, low, close, window)
        
        # 7. Rogers-Satchell波动率估计
        for window in [10, 20, 50]:
            features[f'rogers_satchell_{window}'] = self.calculate_rogers_satchell(open_p, high, low, close, window)
        
        # 8. Yang-Zhang波动率估计（最精确）
        for window in [10, 20, 50]:
            features[f'yang_zhang_{window}'] = self.calculate_yang_zhang(open_p, high, low, close, window)
        
        # 9. 波动率聚类（Volatility Clustering）
        features['vol_clustering_20'] = self.calculate_volatility_clustering(close, 20)
        features['vol_clustering_50'] = self.calculate_volatility_clustering(close, 50)
        
        # 10. 波动率偏度（Volatility Skew）
        features['vol_skew_20'] = self.calculate_volatility_skew(close, 20)
        features['vol_skew_50'] = self.calculate_volatility_skew(close, 50)
        
        # 11. 波动率锥（Volatility Cone）
        vol_cone = self.calculate_volatility_cone(high, low, close)
        for key, value in vol_cone.items():
            features[f'vol_cone_{key}'] = value
        
        return features.fillna(method='ffill').fillna(0)
    
    def calculate_garch(self, close: pd.Series, window: int = 50, 
                       omega: float = 0.01, alpha: float = 0.05, beta: float = 0.90) -> Dict:
        """
        GARCH(1,1) 完整实现
        
        σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
        
        Args:
            close: 收盘价
            window: 初始估计窗口
            omega: 常数项
            alpha: ARCH项系数
            beta: GARCH项系数
        """
        returns = close.pct_change().fillna(0)
        
        # 初始化条件方差（使用历史方差）
        conditional_variance = np.zeros(len(returns))
        initial_var = returns.iloc[:window].var() if len(returns) > window else returns.var()
        conditional_variance[:window] = initial_var
        
        # 迭代计算GARCH
        for t in range(window, len(returns)):
            # ε²(t-1): 昨天的残差平方
            epsilon_squared = returns.iloc[t-1] ** 2
            
            # σ²(t-1): 昨天的条件方差
            prev_variance = conditional_variance[t-1]
            
            # GARCH(1,1)方程
            conditional_variance[t] = omega + alpha * epsilon_squared + beta * prev_variance
        
        garch_vol = pd.Series(np.sqrt(conditional_variance), index=close.index)
        
        # 标准化残差（用于检测异常值）
        standardized_residuals = returns / (garch_vol + 1e-9)
        
        return {
            'volatility': garch_vol,
            'variance': pd.Series(conditional_variance, index=close.index),
            'std_residuals': standardized_residuals,
            'volatility_ratio': garch_vol / (garch_vol.rolling(20).mean() + 1e-9),
            'high_vol_regime': (garch_vol > garch_vol.rolling(50).quantile(0.75)).astype(int)
        }
    
    def calculate_egarch(self, close: pd.Series, window: int = 50) -> Dict:
        """
        EGARCH - 指数GARCH（捕捉非对称性）
        
        log(σ²(t)) = ω + α·|z(t-1)| + γ·z(t-1) + β·log(σ²(t-1))
        """
        returns = close.pct_change().fillna(0)
        
        # 简化的EGARCH实现（固定参数）
        omega, alpha, gamma, beta = 0.01, 0.1, -0.05, 0.85
        
        log_variance = np.zeros(len(returns))
        initial_var = returns.iloc[:window].var() if len(returns) > window else returns.var()
        log_variance[:window] = np.log(initial_var + 1e-9)
        
        for t in range(window, len(returns)):
            # 标准化残差
            prev_variance = np.exp(log_variance[t-1])
            z = returns.iloc[t-1] / (np.sqrt(prev_variance) + 1e-9)
            
            # EGARCH方程
            log_variance[t] = omega + alpha * abs(z) + gamma * z + beta * log_variance[t-1]
        
        egarch_vol = pd.Series(np.sqrt(np.exp(log_variance)), index=close.index)
        
        # 杠杆效应（负收益导致更高波动）
        leverage_effect = []
        for t in range(1, len(returns)):
            if returns.iloc[t-1] < 0:
                leverage = egarch_vol.iloc[t] / (egarch_vol.iloc[t-1] + 1e-9) - 1
            else:
                leverage = 0
            leverage_effect.append(leverage)
        leverage_effect = pd.Series([0] + leverage_effect, index=close.index)
        
        return {
            'volatility': egarch_vol,
            'leverage_effect': leverage_effect,
            'asymmetry': leverage_effect.rolling(20).mean()
        }
    
    def calculate_gjr_garch(self, close: pd.Series, window: int = 50) -> Dict:
        """
        GJR-GARCH - 非对称GARCH（Glosten-Jagannathan-Runkle）
        
        σ²(t) = ω + α·ε²(t-1) + γ·I(t-1)·ε²(t-1) + β·σ²(t-1)
        其中 I(t-1) = 1 if ε(t-1) < 0, else 0
        """
        returns = close.pct_change().fillna(0)
        
        # GJR-GARCH参数
        omega, alpha, gamma, beta = 0.01, 0.04, 0.09, 0.85
        
        conditional_variance = np.zeros(len(returns))
        initial_var = returns.iloc[:window].var() if len(returns) > window else returns.var()
        conditional_variance[:window] = initial_var
        
        for t in range(window, len(returns)):
            epsilon_squared = returns.iloc[t-1] ** 2
            prev_variance = conditional_variance[t-1]
            
            # 负收益指示器
            negative_indicator = 1 if returns.iloc[t-1] < 0 else 0
            
            # GJR-GARCH方程
            conditional_variance[t] = (omega + 
                                      alpha * epsilon_squared + 
                                      gamma * negative_indicator * epsilon_squared + 
                                      beta * prev_variance)
        
        gjr_vol = pd.Series(np.sqrt(conditional_variance), index=close.index)
        
        # 计算下行波动率（仅负收益）
        downside_returns = returns.clip(upper=0)
        downside_vol = downside_returns.rolling(window).std()
        
        return {
            'volatility': gjr_vol,
            'downside_vol': downside_vol,
            'upside_downside_ratio': gjr_vol / (downside_vol + 1e-9)
        }
    
    def calculate_realized_volatility(self, close: pd.Series, window: int) -> pd.Series:
        """已实现波动率（基于高频收益率）"""
        returns = close.pct_change().fillna(0)
        realized_vol = np.sqrt((returns ** 2).rolling(window).sum())
        return realized_vol
    
    def calculate_parkinson(self, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """
        Parkinson波动率估计（基于高低价）
        
        σ² = (1 / (4·ln(2))) · Σ(ln(H/L))²
        """
        hl_ratio = np.log(high / (low + 1e-9))
        parkinson_var = (hl_ratio ** 2).rolling(window).sum() / (4 * np.log(2) * window)
        return np.sqrt(parkinson_var)
    
    def calculate_garman_klass(self, open_p: pd.Series, high: pd.Series, 
                              low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Garman-Klass波动率估计（考虑开盘和收盘）
        
        σ² = 0.5·(ln(H/L))² - (2·ln(2)-1)·(ln(C/O))²
        """
        hl = np.log(high / (low + 1e-9)) ** 2
        co = np.log(close / (open_p + 1e-9)) ** 2
        
        gk_var = (0.5 * hl - (2 * np.log(2) - 1) * co).rolling(window).mean()
        return np.sqrt(gk_var.clip(lower=0))
    
    def calculate_rogers_satchell(self, open_p: pd.Series, high: pd.Series,
                                  low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Rogers-Satchell波动率估计（考虑漂移）
        
        σ² = Σ[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]
        """
        hc = np.log(high / (close + 1e-9))
        ho = np.log(high / (open_p + 1e-9))
        lc = np.log(low / (close + 1e-9))
        lo = np.log(low / (open_p + 1e-9))
        
        rs_var = (hc * ho + lc * lo).rolling(window).mean()
        return np.sqrt(rs_var.clip(lower=0))
    
    def calculate_yang_zhang(self, open_p: pd.Series, high: pd.Series,
                            low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Yang-Zhang波动率估计（最精确，考虑隔夜跳空）
        
        综合了开盘跳空、日内波动、收盘波动
        """
        # 隔夜波动率（Overnight Volatility）
        overnight_returns = np.log(open_p / (close.shift(1) + 1e-9))
        overnight_var = (overnight_returns ** 2).rolling(window).mean()
        
        # 开盘到收盘波动率（Open-to-Close Volatility）
        oc_returns = np.log(close / (open_p + 1e-9))
        oc_var = (oc_returns ** 2).rolling(window).mean()
        
        # Rogers-Satchell组件
        rs_vol = self.calculate_rogers_satchell(open_p, high, low, close, window)
        rs_var = rs_vol ** 2
        
        # Yang-Zhang = k·隔夜方差 + 开盘收盘方差 + (1-k)·RS方差
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
        return np.sqrt(yz_var.clip(lower=0))
    
    def calculate_volatility_clustering(self, close: pd.Series, window: int) -> pd.Series:
        """
        波动率聚类（高波动后跟高波动，低波动后跟低波动）
        """
        returns = close.pct_change().fillna(0)
        abs_returns = returns.abs()
        
        # 计算波动率的自相关性
        vol_acf = abs_returns.rolling(window).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        )
        return vol_acf
    
    def calculate_volatility_skew(self, close: pd.Series, window: int) -> pd.Series:
        """
        波动率偏度（收益率分布的非对称性）
        """
        returns = close.pct_change().fillna(0)
        vol_skew = returns.rolling(window).skew()
        return vol_skew
    
    def calculate_volatility_cone(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """
        波动率锥（不同时间尺度的波动率分位数）
        
        用于识别当前波动率是否异常高或低
        """
        returns = close.pct_change().fillna(0)
        
        features = {}
        windows = [5, 10, 20, 50]
        
        for w in windows:
            realized_vol = returns.rolling(w).std() * np.sqrt(252)  # 年化
            
            # 计算历史分位数
            vol_rank = realized_vol.rolling(252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else 0.5, 
                raw=False
            )
            
            features[f'{w}d_vol_percentile'] = vol_rank
            features[f'{w}d_vol'] = realized_vol
        
        # 波动率曲线形状（正常、倒挂、陡峭）
        features['vol_curve_slope'] = (features['50d_vol'] - features['5d_vol']) / (features['5d_vol'] + 1e-9)
        
        return features


def main():
    """测试Advanced Volatility完整实现"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1H')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.005)
    
    ohlcv = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(len(dates)) * 0.001),
        'high': close_prices * 1.01,
        'low': close_prices * 0.99,
        'close': close_prices
    }, index=dates)
    
    av = AdvancedVolatility()
    features = av.calculate_all(ohlcv)
    print(f"Generated {features.shape[1]} Advanced Volatility features")
    print(f"\nFeature categories:")
    print(f"  - GARCH: {len([c for c in features.columns if 'garch' in c])} features")
    print(f"  - EGARCH: {len([c for c in features.columns if 'egarch' in c])} features")
    print(f"  - GJR-GARCH: {len([c for c in features.columns if 'gjr' in c])} features")
    print(f"  - Realized Vol: {len([c for c in features.columns if 'realized' in c])} features")
    print(f"  - Parkinson: {len([c for c in features.columns if 'parkinson' in c])} features")
    print(f"  - Garman-Klass: {len([c for c in features.columns if 'garman' in c])} features")
    print(f"  - Rogers-Satchell: {len([c for c in features.columns if 'rogers' in c])} features")
    print(f"  - Yang-Zhang: {len([c for c in features.columns if 'yang' in c])} features")
    print(f"  - Volatility Cone: {len([c for c in features.columns if 'cone' in c])} features")


if __name__ == '__main__':
    main()
