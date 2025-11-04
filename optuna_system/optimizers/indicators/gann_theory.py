# -*- coding: utf-8 -*-
"""
甘氏理论完整实现
包含：甘氏角度线（7条）、甘氏扇形、甘氏九方图、甘氏轮、自然阻力点
"""

import warnings
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class GannTheory:
    """甘氏理论完整实现"""
    
    def __init__(self, window: int = 50):
        """
        Args:
            window: 计算窗口（用于识别关键高低点）
        """
        self.window = window
        
        # 甘氏角度线比率（价格/时间）
        self.gann_angles = {
            '1x8': 1/8,   # 极缓
            '1x4': 1/4,   # 缓
            '1x3': 1/3,   # 缓
            '1x2': 1/2,   # 中缓
            '1x1': 1.0,   # 45度（最重要）
            '2x1': 2.0,   # 中陡
            '3x1': 3.0,   # 陡
            '4x1': 4.0,   # 极陡
            '8x1': 8.0    # 超陡
        }
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. 甘氏角度线（从关键摆动点开始）
        gann_angle_features = self.calculate_gann_angles(high, low, close)
        for key, value in gann_angle_features.items():
            features[f'gann_angle_{key}'] = value
        
        # 2. 甘氏扇形（多条角度线的综合）
        gann_fan_features = self.calculate_gann_fan(high, low, close)
        for key, value in gann_fan_features.items():
            features[f'gann_fan_{key}'] = value
        
        # 3. 甘氏九方图（Square of 9）
        gann_sq9_features = self.calculate_gann_square_of_9(close)
        for key, value in gann_sq9_features.items():
            features[f'gann_sq9_{key}'] = value
        
        # 4. 甘氏轮（Gann Wheel）- 时间-价格关系
        gann_wheel_features = self.calculate_gann_wheel(high, low, close)
        for key, value in gann_wheel_features.items():
            features[f'gann_wheel_{key}'] = value
        
        # 5. 自然阻力点（Natural Resistance Points）
        natural_levels = self.calculate_natural_resistance_levels(close)
        for key, value in natural_levels.items():
            features[f'gann_resistance_{key}'] = value
        
        return features.fillna(method='ffill').fillna(0)
    
    def calculate_gann_angles(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """
        计算甘氏角度线
        
        从关键摆动高点向上画角度线，从摆动低点向下画角度线
        """
        features = {}
        
        # 识别关键摆动点
        swing_highs = self._find_swing_highs(high, self.window)
        swing_lows = self._find_swing_lows(low, self.window)
        
        # 为每条角度线计算价格位置
        for angle_name, angle_ratio in self.gann_angles.items():
            # 从摆动低点向上的角度线
            upward_line = pd.Series(np.nan, index=close.index)
            for swing_idx, swing_price in swing_lows.items():
                if swing_idx >= len(close):
                    continue
                # 从摆动点开始，每根K线上涨angle_ratio
                for i in range(swing_idx, len(close)):
                    time_elapsed = i - swing_idx
                    projected_price = swing_price + (time_elapsed * angle_ratio * swing_price * 0.01)
                    if pd.isna(upward_line.iloc[i]) or projected_price > upward_line.iloc[i]:
                        upward_line.iloc[i] = projected_price
            
            # 从摆动高点向下的角度线
            downward_line = pd.Series(np.nan, index=close.index)
            for swing_idx, swing_price in swing_highs.items():
                if swing_idx >= len(close):
                    continue
                for i in range(swing_idx, len(close)):
                    time_elapsed = i - swing_idx
                    projected_price = swing_price - (time_elapsed * angle_ratio * swing_price * 0.01)
                    if pd.isna(downward_line.iloc[i]) or projected_price < downward_line.iloc[i]:
                        downward_line.iloc[i] = projected_price
            
            # 计算价格相对于角度线的位置
            features[f'{angle_name}_up_distance'] = (close - upward_line) / (close + 1e-9)
            features[f'{angle_name}_down_distance'] = (downward_line - close) / (close + 1e-9)
            
            # 突破信号（价格穿越角度线）
            if angle_name == '1x1':  # 45度线最重要
                features[f'{angle_name}_up_break'] = ((close > upward_line) & (close.shift(1) <= upward_line.shift(1))).astype(int)
                features[f'{angle_name}_down_break'] = ((close < downward_line) & (close.shift(1) >= downward_line.shift(1))).astype(int)
        
        return features
    
    def calculate_gann_fan(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """
        甘氏扇形分析
        
        计算价格相对于整个扇形的位置
        """
        features = {}
        
        # 使用最近的摆动低点作为扇形原点
        swing_lows = self._find_swing_lows(low, self.window)
        
        if len(swing_lows) == 0:
            return {'fan_position': pd.Series(0.5, index=close.index)}
        
        # 取最近的摆动低点
        latest_swing_idx = max(swing_lows.keys())
        latest_swing_price = swing_lows[latest_swing_idx]
        
        # 计算当前价格在扇形中的位置（0=最下方1x8线，1=最上方8x1线）
        fan_position = []
        for i in range(len(close)):
            if i <= latest_swing_idx:
                fan_position.append(0.5)
            else:
                time_elapsed = i - latest_swing_idx
                
                # 最下方线（1x8）
                bottom_line = latest_swing_price + (time_elapsed * self.gann_angles['1x8'] * latest_swing_price * 0.01)
                # 最上方线（8x1）
                top_line = latest_swing_price + (time_elapsed * self.gann_angles['8x1'] * latest_swing_price * 0.01)
                
                # 当前价格在扇形中的位置
                if top_line > bottom_line:
                    position = (close.iloc[i] - bottom_line) / (top_line - bottom_line + 1e-9)
                    position = max(0, min(1, position))
                else:
                    position = 0.5
                
                fan_position.append(position)
        
        features['fan_position'] = pd.Series(fan_position, index=close.index)
        
        # 扇形压缩/扩张（价格波动性指标）
        features['fan_width'] = (close.rolling(20).max() - close.rolling(20).min()) / (close + 1e-9)
        
        return features
    
    def calculate_gann_square_of_9(self, close: pd.Series) -> Dict:
        """
        甘氏九方图（Square of 9）
        
        基于价格的平方根计算支撑/阻力位
        """
        features = {}
        
        # 九方图关键角度（度数）
        key_angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        
        # 计算当前价格的平方根
        sqrt_price = np.sqrt(close)
        
        # 计算下一个重要价格水平（基于360度旋转）
        # Square of 9: 从中心螺旋向外，每旋转360度价格增加2个单位的平方根
        
        # 最近的整数平方根
        floor_sqrt = np.floor(sqrt_price)
        ceil_sqrt = np.ceil(sqrt_price)
        
        # 下一个支撑位（向下的完整平方）
        next_support = (floor_sqrt ** 2)
        
        # 下一个阻力位（向上的完整平方）
        next_resistance = (ceil_sqrt ** 2)
        
        features['sq9_next_support'] = next_support
        features['sq9_next_resistance'] = next_resistance
        features['sq9_support_distance'] = (close - next_support) / (close + 1e-9)
        features['sq9_resistance_distance'] = (next_resistance - close) / (close + 1e-9)
        
        # 价格在平方之间的位置（0-1）
        features['sq9_position'] = (close - next_support) / (next_resistance - next_support + 1e-9)
        
        # 45度角倍数的目标位（重要的甘氏水平）
        # 计算当前是否接近45度角的倍数
        current_angle_in_square = ((sqrt_price - floor_sqrt) * 360) % 360
        
        # 计算到所有关键角度的最小距离（向量化）
        distances_to_key_angles = pd.DataFrame({f'dist_{angle}': abs(current_angle_in_square - angle) for angle in key_angles})
        min_distance_to_key_angle = distances_to_key_angles.min(axis=1)
        features['sq9_near_key_angle'] = (min_distance_to_key_angle < 15).astype(int)  # 15度容差
        
        # Cardinal Cross (0, 90, 180, 270度) - 最重要的支撑/阻力
        cardinal_angles = [0, 90, 180, 270]
        distances_to_cardinal = pd.DataFrame({f'dist_{angle}': abs(current_angle_in_square - angle) for angle in cardinal_angles})
        min_cardinal_distance = distances_to_cardinal.min(axis=1)
        features['sq9_near_cardinal'] = (min_cardinal_distance < 10).astype(int)
        
        return features
    
    def calculate_gann_wheel(self, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
        """
        甘氏轮（时间-价格关系）
        
        基于时间周期计算重要的转折点
        """
        features = {}
        
        # 甘氏重要时间周期（K线数）
        gann_time_cycles = [7, 13, 21, 30, 45, 60, 90, 120, 144, 180, 360]
        
        # 找到摆动点并计算距离
        swing_highs = self._find_swing_highs(high, self.window)
        swing_lows = self._find_swing_lows(low, self.window)
        
        # 距离最近摆动高点的时间
        bars_since_swing_high = []
        bars_since_swing_low = []
        
        for i in range(len(close)):
            # 找到最近的摆动高点
            recent_high_dist = 999
            for swing_idx in swing_highs.keys():
                if swing_idx <= i:
                    recent_high_dist = i - swing_idx
            bars_since_swing_high.append(recent_high_dist)
            
            # 找到最近的摆动低点
            recent_low_dist = 999
            for swing_idx in swing_lows.keys():
                if swing_idx <= i:
                    recent_low_dist = i - swing_idx
            bars_since_swing_low.append(recent_low_dist)
        
        features['wheel_bars_since_high'] = pd.Series(bars_since_swing_high, index=close.index)
        features['wheel_bars_since_low'] = pd.Series(bars_since_swing_low, index=close.index)
        
        # 检测是否接近重要时间周期
        near_time_cycle_high = []
        near_time_cycle_low = []
        
        for i in range(len(close)):
            # 检查距离摆动高点的时间是否接近甘氏周期
            near_cycle_h = any(abs(bars_since_swing_high[i] - cycle) <= 2 for cycle in gann_time_cycles)
            near_time_cycle_high.append(1 if near_cycle_h else 0)
            
            # 检查距离摆动低点的时间是否接近甘氏周期
            near_cycle_l = any(abs(bars_since_swing_low[i] - cycle) <= 2 for cycle in gann_time_cycles)
            near_time_cycle_low.append(1 if near_cycle_l else 0)
        
        features['wheel_near_cycle_high'] = pd.Series(near_time_cycle_high, index=close.index)
        features['wheel_near_cycle_low'] = pd.Series(near_time_cycle_low, index=close.index)
        
        return features
    
    def calculate_natural_resistance_levels(self, close: pd.Series) -> Dict:
        """
        计算自然阻力点
        
        基于整数价格、1/8价格分割
        """
        features = {}
        
        # 最近的整数价格
        floor_price = np.floor(close)
        ceil_price = np.ceil(close)
        
        # 1/8分割点（甘氏重要分割）
        eighth_levels = []
        for i in range(len(close)):
            floor_p = floor_price.iloc[i]
            ceil_p = ceil_price.iloc[i]
            
            # 计算8个1/8分割点
            levels = [floor_p + (ceil_p - floor_p) * (j / 8) for j in range(9)]
            
            # 找到最近的1/8分割点
            current_p = close.iloc[i]
            distances = [abs(current_p - level) for level in levels]
            nearest_level = levels[np.argmin(distances)]
            
            eighth_levels.append((current_p - nearest_level) / (current_p + 1e-9))
        
        features['natural_eighth_distance'] = pd.Series(eighth_levels, index=close.index)
        
        # 距离整数价格的距离
        features['natural_integer_distance'] = (close - close.round()) / (close + 1e-9)
        
        # 是否接近整数或半整数
        features['natural_near_integer'] = (abs(close - close.round()) < close * 0.01).astype(int)
        features['natural_near_half'] = (abs(close - (floor_price + 0.5)) < close * 0.01).astype(int)
        
        return features
    
    def _find_swing_highs(self, high: pd.Series, window: int) -> Dict[int, float]:
        """识别摆动高点"""
        swing_highs = {}
        
        for i in range(window, len(high) - window):
            is_swing = True
            for j in range(1, window + 1):
                if high.iloc[i] <= high.iloc[i-j] or high.iloc[i] <= high.iloc[i+j]:
                    is_swing = False
                    break
            
            if is_swing:
                swing_highs[i] = high.iloc[i]
        
        return swing_highs
    
    def _find_swing_lows(self, low: pd.Series, window: int) -> Dict[int, float]:
        """识别摆动低点"""
        swing_lows = {}
        
        for i in range(window, len(low) - window):
            is_swing = True
            for j in range(1, window + 1):
                if low.iloc[i] >= low.iloc[i-j] or low.iloc[i] >= low.iloc[i+j]:
                    is_swing = False
                    break
            
            if is_swing:
                swing_lows[i] = low.iloc[i]
        
        return swing_lows


def main():
    """测试Gann Theory完整实现"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'high': close_prices * 1.01,
        'low': close_prices * 0.99,
        'close': close_prices
    }, index=dates)
    
    gt = GannTheory(window=10)
    features = gt.calculate_all(ohlcv)
    print(f"Generated {features.shape[1]} Gann Theory features")
    print(f"\nFeature categories:")
    print(f"  - Gann Angles: {len([c for c in features.columns if 'angle' in c])} features")
    print(f"  - Gann Fan: {len([c for c in features.columns if 'fan' in c])} features")
    print(f"  - Square of 9: {len([c for c in features.columns if 'sq9' in c])} features")
    print(f"  - Gann Wheel: {len([c for c in features.columns if 'wheel' in c])} features")
    print(f"  - Natural Resistance: {len([c for c in features.columns if 'natural' in c])} features")


if __name__ == '__main__':
    main()
