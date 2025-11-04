# -*- coding: utf-8 -*-
"""成交量分析"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')


class VolumeAnalysis:
    """成交量分析：OBV, A/D Line, CMF, VWAP, Volume Profile, EOM, PVI, NVI"""
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high, low, close, volume = ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        features = pd.DataFrame(index=ohlcv.index)
        
        # OBV
        obv_features = self.calculate_obv_advanced(close, volume)
        for key, value in obv_features.items():
            features[f'obv_{key}'] = value
        
        # A/D Line
        ad_features = self.calculate_ad_line(high, low, close, volume)
        for key, value in ad_features.items():
            features[f'ad_{key}'] = value
        
        # CMF
        features['cmf'] = self.calculate_cmf(high, low, close, volume)
        
        # VWAP
        vwap_features = self.calculate_vwap_full(high, low, close, volume)
        for key, value in vwap_features.items():
            features[f'vwap_{key}'] = value
        
        # Volume Profile (完整精确实现)
        vp_features = self.calculate_volume_profile_precise(high, low, close, volume, window=50, bins=100)
        for key, value in vp_features.items():
            features[f'vp_{key}'] = value
        
        # Delta Proxy
        features['delta_proxy'] = self.calculate_volume_delta_proxy(close, volume)
        
        # EOM
        features['eom'] = self.calculate_ease_of_movement(high, low, volume)
        
        # PVI/NVI
        pvi, nvi = self.calculate_pvi_nvi(close, volume)
        features['pvi'] = pvi
        features['nvi'] = nvi
        
        # MFI
        features['mfi'] = self.calculate_mfi(high, low, close, volume)
        
        return features.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
    
    def calculate_obv_advanced(self, close: pd.Series, volume: pd.Series) -> dict:
        obv = (np.sign(close.diff()) * volume).cumsum()
        return {
            'value': obv,
            'ema': obv.ewm(span=20).mean(),
            'divergence': (close / close.rolling(20).mean()) - (obv / obv.rolling(20).mean())
        }
    
    def calculate_ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> dict:
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        ad_line = (clv * volume).cumsum()
        return {
            'value': ad_line,
            'oscillator': ad_line - ad_line.ewm(span=3).mean()
        }
    
    def calculate_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
        mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
        mfv = mfm * volume
        return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)
    
    def calculate_vwap_full(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> dict:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / (volume.cumsum() + 1e-9)
        std = ((typical_price - vwap) ** 2 * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
        std = np.sqrt(std)
        
        return {
            'value': vwap,
            'upper_band': vwap + 2 * std,
            'lower_band': vwap - 2 * std,
            'distance': (close - vwap) / (close + 1e-9),
            'position': (close - vwap) / (std + 1e-9)
        }
    
    def calculate_volume_profile_precise(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                        volume: pd.Series, window: int = 50, bins: int = 100) -> dict:
        """完整Volume Profile实现 - 精确POC/VAH/VAL/Value Area计算"""
        
        def calculate_profile_for_window(start_idx: int, end_idx: int) -> tuple:
            """计算单个窗口的Volume Profile"""
            if end_idx - start_idx < 10:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            
            # 获取窗口内数据
            highs = high.iloc[start_idx:end_idx]
            lows = low.iloc[start_idx:end_idx]
            prices = close.iloc[start_idx:end_idx]
            vols = volume.iloc[start_idx:end_idx]
            
            # 确定价格范围
            price_min = lows.min()
            price_max = highs.max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            
            # 创建价格区间（bins）
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 分配成交量到价格区间（考虑每根K线的高低范围）
            volume_profile = np.zeros(bins)
            
            for i in range(len(prices)):
                bar_high = highs.iloc[i]
                bar_low = lows.iloc[i]
                bar_vol = vols.iloc[i]
                
                # 找到该K线覆盖的价格区间
                affected_bins = []
                for j, (edge_low, edge_high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    # 检查K线价格范围是否与该bin重叠
                    if not (bar_high < edge_low or bar_low > edge_high):
                        # 计算重叠比例
                        overlap_low = max(bar_low, edge_low)
                        overlap_high = min(bar_high, edge_high)
                        overlap_ratio = (overlap_high - overlap_low) / (bar_high - bar_low + 1e-9)
                        affected_bins.append((j, overlap_ratio))
                
                # 按比例分配成交量
                total_ratio = sum(r for _, r in affected_bins)
                for bin_idx, ratio in affected_bins:
                    volume_profile[bin_idx] += bar_vol * (ratio / (total_ratio + 1e-9))
            
            # 1. 找到POC (Point of Control) - 成交量最大的价格区间
            poc_idx = volume_profile.argmax()
            poc_price = bin_centers[poc_idx]
            
            # 2. 计算Value Area (70%成交量区域)
            total_volume = volume_profile.sum()
            target_volume = total_volume * 0.70
            
            # 从POC向两侧扩展，直到累计成交量达到70%
            value_area_volume = volume_profile[poc_idx]
            lower_idx = poc_idx
            upper_idx = poc_idx
            
            while value_area_volume < target_volume:
                # 决定向上还是向下扩展（选择成交量较大的方向）
                can_expand_down = lower_idx > 0
                can_expand_up = upper_idx < bins - 1
                
                if not can_expand_down and not can_expand_up:
                    break
                
                vol_below = volume_profile[lower_idx - 1] if can_expand_down else 0
                vol_above = volume_profile[upper_idx + 1] if can_expand_up else 0
                
                if vol_below >= vol_above and can_expand_down:
                    lower_idx -= 1
                    value_area_volume += volume_profile[lower_idx]
                elif can_expand_up:
                    upper_idx += 1
                    value_area_volume += volume_profile[upper_idx]
                else:
                    break
            
            vah_price = bin_centers[upper_idx]  # Value Area High
            val_price = bin_centers[lower_idx]  # Value Area Low
            
            # 3. 识别Single Prints（低成交量区域）
            median_volume = np.median(volume_profile[volume_profile > 0])
            single_print_threshold = median_volume * 0.3
            single_prints = np.sum(volume_profile < single_print_threshold)
            
            # 4. 识别Poor High/Low（价格极值处的低成交量）
            edge_threshold = median_volume * 0.5
            poor_high = 1 if volume_profile[-5:].mean() < edge_threshold else 0
            poor_low = 1 if volume_profile[:5].mean() < edge_threshold else 0
            
            return poc_price, vah_price, val_price, single_prints / bins, poor_high + poor_low
        
        # 滚动窗口计算Volume Profile
        poc_list = []
        vah_list = []
        val_list = []
        single_prints_list = []
        poor_hl_list = []
        
        for i in range(len(close)):
            if i < window:
                poc_list.append(np.nan)
                vah_list.append(np.nan)
                val_list.append(np.nan)
                single_prints_list.append(np.nan)
                poor_hl_list.append(np.nan)
            else:
                poc, vah, val, sp, ph = calculate_profile_for_window(i - window, i)
                poc_list.append(poc)
                vah_list.append(vah)
                val_list.append(val)
                single_prints_list.append(sp)
                poor_hl_list.append(ph)
        
        # 计算价格相对于Volume Profile的位置
        poc_series = pd.Series(poc_list, index=close.index)
        vah_series = pd.Series(vah_list, index=close.index)
        val_series = pd.Series(val_list, index=close.index)
        
        return {
            'poc': poc_series,
            'vah': vah_series,
            'val': val_series,
            'price_to_poc': (close - poc_series) / (close + 1e-9),
            'price_to_vah': (close - vah_series) / (close + 1e-9),
            'price_to_val': (close - val_series) / (close + 1e-9),
            'in_value_area': ((close >= val_series) & (close <= vah_series)).astype(int),
            'value_area_width': (vah_series - val_series) / (close + 1e-9),
            'single_prints_ratio': pd.Series(single_prints_list, index=close.index),
            'poor_high_low': pd.Series(poor_hl_list, index=close.index)
        }
    
    def calculate_volume_delta_proxy(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """使用价格变化代理买卖压力"""
        price_change = close.diff()
        buy_volume = volume.where(price_change > 0, 0)
        sell_volume = volume.where(price_change < 0, 0)
        delta = (buy_volume - sell_volume).rolling(20).sum()
        return delta / (volume.rolling(20).sum() + 1e-9)
    
    def calculate_ease_of_movement(self, high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        distance = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        box_ratio = (volume / 1e8) / (high - low + 1e-9)
        eom = distance / (box_ratio + 1e-9)
        return eom.rolling(period).mean()
    
    def calculate_pvi_nvi(self, close: pd.Series, volume: pd.Series) -> tuple:
        pvi = pd.Series(100.0, index=close.index)
        nvi = pd.Series(100.0, index=close.index)
        
        for i in range(1, len(close)):
            if volume.iloc[i] > volume.iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] + ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * pvi.iloc[i-1]
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
            
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] + ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * nvi.iloc[i-1]
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return pvi, nvi
    
    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-9)))
        return mfi


def main():
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'high': close_prices * 1.01, 'low': close_prices * 0.99, 
        'close': close_prices, 'volume': np.abs(np.random.randn(len(dates))) * 1e6
    }, index=dates)
    va = VolumeAnalysis()
    features = va.calculate_all(ohlcv)
    print(f"Generated {features.shape[1]} volume analysis features")


if __name__ == '__main__':
    main()

