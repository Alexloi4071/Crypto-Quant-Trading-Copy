# -*- coding: utf-8 -*-
"""订单流代理特征（基于OHLCV）"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class OrderFlowProxy:
    """订单流代理：买卖价差估算、大单检测、交易侵略性、市场深度、流动性区域"""
    
    def calculate_proxy(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high, low, close, volume = ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        features = pd.DataFrame(index=ohlcv.index)
        
        features['spread_estimate'] = self.estimate_bid_ask_spread(high, low)
        features['large_order_signal'] = self.detect_large_orders(volume, close)
        features['trade_aggression'] = self.calculate_trade_aggression(close, volume)
        features['market_depth_est'] = self.estimate_market_depth(volume, close)
        
        liquidity = self.identify_liquidity_zones(volume, close)
        for key, value in liquidity.items():
            features[f'liquidity_{key}'] = value
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """calculate_all方法（别名：calculate_proxy）"""
        return self.calculate_proxy(ohlcv)
    
    def estimate_bid_ask_spread(self, high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
        """使用高低价估算价差"""
        return (high - low) / ((high + low) / 2 + 1e-9)
    
    def detect_large_orders(self, volume: pd.Series, close: pd.Series, threshold: float = 2.5) -> pd.Series:
        """检测大单（成交量异常）"""
        vol_ma = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        z_score = (volume - vol_ma) / (vol_std + 1e-9)
        return (z_score > threshold).astype(int)
    
    def calculate_trade_aggression(self, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
        """交易侵略性（基于价格方向和成交量）"""
        price_change = close.diff()
        buy_pressure = volume.where(price_change > 0, 0)
        sell_pressure = volume.where(price_change < 0, 0)
        
        buy_sum = buy_pressure.rolling(window).sum()
        sell_sum = sell_pressure.rolling(window).sum()
        
        aggression = (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-9)
        return aggression
    
    def estimate_market_depth(self, volume: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """市场深度估算（成交量/价格波动率）"""
        price_vol = close.pct_change().rolling(window).std()
        vol_avg = volume.rolling(window).mean()
        return vol_avg / (price_vol + 1e-9)
    
    def identify_liquidity_zones(self, volume: pd.Series, close: pd.Series, window: int = 50) -> dict:
        """识别流动性区域"""
        vol_profile = volume.rolling(window).sum()
        price_levels = close.rolling(window).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.mean(), raw=False)
        
        high_liquidity = (vol_profile > vol_profile.rolling(window).quantile(0.75)).astype(int)
        low_liquidity = (vol_profile < vol_profile.rolling(window).quantile(0.25)).astype(int)
        
        return {
            'high_zone': high_liquidity,
            'low_zone': low_liquidity,
            'level': price_levels,
            'distance_to_level': (close - price_levels) / (close + 1e-9)
        }


def main():
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'high': close_prices * 1.01, 'low': close_prices * 0.99,
        'close': close_prices, 'volume': np.abs(np.random.randn(len(dates))) * 1e6
    }, index=dates)
    ofp = OrderFlowProxy()
    features = ofp.calculate_proxy(ohlcv)
    print(f"Generated {features.shape[1]} order flow proxy features")


if __name__ == '__main__':
    main()

