# -*- coding: utf-8 -*-
"""
和谐形态识别 - 完整实现
基于Scott Carney的和谐交易方法
包含：Gartley, Butterfly, Bat, Crab, Shark, Cypher, ABCD, Three Drives
"""

import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class SwingPoint:
    """摆动点数据类"""
    def __init__(self, index: int, price: float, is_high: bool):
        self.index = index
        self.price = price
        self.is_high = is_high


class HarmonicPatterns:
    """和谐形态完整实现 - Gartley, Butterfly, Bat, Crab, Shark, Cypher, ABCD, Three Drives
    
    新增功能：
    1. 动态容差调整（基于波动率）
    2. 形态评分系统（多维度）
    3. 形态失败检测
    """
    
    def __init__(self, swing_window: int = 5, base_tolerance: float = 0.05, use_dynamic_tolerance: bool = True):
        """
        Args:
            swing_window: 摆动点识别窗口
            base_tolerance: 基础Fibonacci比率容差（±5%）
            use_dynamic_tolerance: 是否使用动态容差调整
        """
        self.swing_window = swing_window
        self.base_tolerance = base_tolerance
        self.use_dynamic_tolerance = use_dynamic_tolerance
        self.tolerance = base_tolerance  # 当前使用的容差（可能动态调整）
        self.current_volatility = 0.0  # 当前波动率
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high, low = ohlcv['high'], ohlcv['low']
        close = ohlcv['close']
        features = pd.DataFrame(index=ohlcv.index)
        
        # 计算当前波动率并调整容差
        if self.use_dynamic_tolerance:
            self.current_volatility = self._calculate_volatility(close)
            self.tolerance = self._adjust_tolerance_by_volatility(self.current_volatility)
        else:
            self.tolerance = self.base_tolerance
        
        # 1. 识别摆动点
        swing_highs, swing_lows = self._find_swing_points(high, low)
        
        # 2. 检测各种和谐形态（传入ohlcv.index以便创建正确的Series）
        features['gartley_bullish'], features['gartley_bullish_strength'] = self.detect_gartley(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['gartley_bearish'], features['gartley_bearish_strength'] = self.detect_gartley(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['butterfly_bullish'], features['butterfly_bullish_strength'] = self.detect_butterfly(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['butterfly_bearish'], features['butterfly_bearish_strength'] = self.detect_butterfly(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['bat_bullish'], features['bat_bullish_strength'] = self.detect_bat(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['bat_bearish'], features['bat_bearish_strength'] = self.detect_bat(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['crab_bullish'], features['crab_bullish_strength'] = self.detect_crab(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['crab_bearish'], features['crab_bearish_strength'] = self.detect_crab(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['shark_bullish'], features['shark_bullish_strength'] = self.detect_shark(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['shark_bearish'], features['shark_bearish_strength'] = self.detect_shark(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['cypher_bullish'], features['cypher_bullish_strength'] = self.detect_cypher(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['cypher_bearish'], features['cypher_bearish_strength'] = self.detect_cypher(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['abcd_bullish'], features['abcd_bullish_strength'] = self.detect_abcd(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['abcd_bearish'], features['abcd_bearish_strength'] = self.detect_abcd(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        features['three_drives_bullish'], features['three_drives_bullish_strength'] = self.detect_three_drives(swing_lows, swing_highs, 'bullish', ohlcv.index)
        features['three_drives_bearish'], features['three_drives_bearish_strength'] = self.detect_three_drives(swing_highs, swing_lows, 'bearish', ohlcv.index)
        
        return features.shift(1).fillna(0)
    
    def _find_swing_points(self, high: pd.Series, low: pd.Series) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """识别摆动高点和低点"""
        swing_highs = []
        swing_lows = []
        
        for i in range(self.swing_window, len(high) - self.swing_window):
            # 摆动高点：中间K线高点高于左右各swing_window根K线
            is_swing_high = True
            for j in range(1, self.swing_window + 1):
                if high.iloc[i] <= high.iloc[i-j] or high.iloc[i] <= high.iloc[i+j]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(SwingPoint(i, high.iloc[i], True))
            
            # 摆动低点：中间K线低点低于左右各swing_window根K线
            is_swing_low = True
            for j in range(1, self.swing_window + 1):
                if low.iloc[i] >= low.iloc[i-j] or low.iloc[i] >= low.iloc[i+j]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(SwingPoint(i, low.iloc[i], False))
        
        return swing_highs, swing_lows
    
    def _calculate_volatility(self, close: pd.Series, window: int = 20) -> float:
        """
        计算当前波动率（使用ATR的简化版本）
        
        Returns:
            float: 波动率（0-1范围，归一化）
        """
        if len(close) < window:
            return 0.02  # 默认2%波动率
        
        # 计算最近window期的波动率
        returns = close.pct_change().dropna()
        recent_returns = returns.tail(window)
        
        # 使用标准差作为波动率度量
        volatility = recent_returns.std()
        
        # 归一化到合理范围 (0.01-0.10)
        normalized_vol = np.clip(volatility, 0.01, 0.10)
        
        return float(normalized_vol)
    
    def _adjust_tolerance_by_volatility(self, volatility: float) -> float:
        """
        根据市场波动率动态调整Fibonacci比率容差
        
        逻辑：
        - 低波动率（<2%）：使用较小容差（精确匹配）
        - 中波动率（2-5%）：使用中等容差
        - 高波动率（>5%）：使用较大容差（宽松匹配）
        
        Args:
            volatility: 当前波动率
            
        Returns:
            float: 调整后的容差
        """
        if volatility < 0.02:
            # 低波动率：严格匹配
            adjusted_tolerance = self.base_tolerance
        elif volatility < 0.05:
            # 中等波动率：线性增加容差
            # 从base_tolerance到base_tolerance*2
            factor = 1 + (volatility - 0.02) / 0.03
            adjusted_tolerance = self.base_tolerance * factor
        else:
            # 高波动率：更宽松的容差
            # 最多增加到base_tolerance*3
            factor = min(3.0, 2 + (volatility - 0.05) / 0.05)
            adjusted_tolerance = self.base_tolerance * factor
        
        # 限制最大容差不超过15%
        return min(adjusted_tolerance, 0.15)
    
    def _check_fib_ratio(self, actual: float, target: float) -> bool:
        """检查Fibonacci比率是否在容差范围内（动态容差）"""
        if target == 0:
            return False
        lower = target * (1 - self.tolerance)
        upper = target * (1 + self.tolerance)
        return lower <= actual <= upper
    
    def _calculate_ratio_quality(self, actual: float, target: float) -> float:
        """计算比率匹配质量（0-1，越接近1越好）"""
        if target == 0:
            return 0.0
        deviation = abs(actual - target) / target
        quality = max(0, 1 - (deviation / self.tolerance))
        return quality
    
    def detect_gartley(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                      pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Gartley形态检测
        
        比率要求：
        - AB = 0.618 XA
        - BC = 0.382-0.886 AB
        - CD = 1.272-1.618 BC
        - AD = 0.786 XA
        """
        # 如果没有足够的摆动点，返回空Series
        if len(primary_swings) < 3 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        # 使用传入的data_index
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        # 遍历所有可能的XABCD组合
        for i in range(len(primary_swings) - 2):
            for j in range(len(secondary_swings) - 1):
                # 确保顺序正确
                if not (primary_swings[i].index < secondary_swings[j].index < primary_swings[i+1].index < 
                       secondary_swings[j+1].index < primary_swings[i+2].index):
                    continue
                
                X = primary_swings[i]
                A = secondary_swings[j]
                B = primary_swings[i+1]
                C = secondary_swings[j+1]
                D = primary_swings[i+2]
                
                # 计算价格差异
                XA = abs(A.price - X.price)
                AB = abs(B.price - A.price)
                BC = abs(C.price - B.price)
                CD = abs(D.price - C.price)
                AD = abs(D.price - A.price)
                
                if XA == 0 or AB == 0 or BC == 0:
                    continue
                
                # 检查Gartley比率
                ratio_AB_XA = AB / XA
                ratio_BC_AB = BC / AB
                ratio_CD_BC = CD / BC
                ratio_AD_XA = AD / XA
                
                if (self._check_fib_ratio(ratio_AB_XA, 0.618) and
                    0.382 <= ratio_BC_AB <= 0.886 and
                    1.272 <= ratio_CD_BC <= 1.618 and
                    self._check_fib_ratio(ratio_AD_XA, 0.786)):
                    
                    # 计算形态强度（所有比率的平均匹配质量）
                    q1 = self._calculate_ratio_quality(ratio_AB_XA, 0.618)
                    q2 = self._calculate_ratio_quality(ratio_BC_AB, 0.618)  # 中间值
                    q3 = self._calculate_ratio_quality(ratio_CD_BC, 1.414)  # 中间值
                    q4 = self._calculate_ratio_quality(ratio_AD_XA, 0.786)
                    strength = (q1 + q2 + q3 + q4) / 4
                    
                    # 标记D点位置为信号
                    if D.index < len(signals):
                        signals[D.index] = 1
                        strengths[D.index] = max(strengths[D.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_butterfly(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                        pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Butterfly形态检测
        
        比率要求：
        - AB = 0.786 XA
        - BC = 0.382-0.886 AB
        - CD = 1.618-2.24 BC
        - AD = 1.27-1.618 XA
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 3 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        for i in range(len(primary_swings) - 2):
            for j in range(len(secondary_swings) - 1):
                if not (primary_swings[i].index < secondary_swings[j].index < primary_swings[i+1].index < 
                       secondary_swings[j+1].index < primary_swings[i+2].index):
                    continue
                
                X, A, B, C, D = primary_swings[i], secondary_swings[j], primary_swings[i+1], secondary_swings[j+1], primary_swings[i+2]
                
                XA = abs(A.price - X.price)
                AB = abs(B.price - A.price)
                BC = abs(C.price - B.price)
                CD = abs(D.price - C.price)
                AD = abs(D.price - A.price)
                
                if XA == 0 or AB == 0 or BC == 0:
                    continue
                
                ratio_AB_XA = AB / XA
                ratio_BC_AB = BC / AB
                ratio_CD_BC = CD / BC
                ratio_AD_XA = AD / XA
                
                if (self._check_fib_ratio(ratio_AB_XA, 0.786) and
                    0.382 <= ratio_BC_AB <= 0.886 and
                    1.618 <= ratio_CD_BC <= 2.24 and
                    1.27 <= ratio_AD_XA <= 1.618):
                    
                    q1 = self._calculate_ratio_quality(ratio_AB_XA, 0.786)
                    q2 = self._calculate_ratio_quality(ratio_BC_AB, 0.618)
                    q3 = self._calculate_ratio_quality(ratio_CD_BC, 1.90)
                    q4 = self._calculate_ratio_quality(ratio_AD_XA, 1.44)
                    strength = (q1 + q2 + q3 + q4) / 4
                    
                    if D.index < len(signals):
                        signals[D.index] = 1
                        strengths[D.index] = max(strengths[D.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_bat(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                  pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Bat形态检测
        
        比率要求：
        - AB = 0.382-0.5 XA
        - BC = 0.382-0.886 AB
        - CD = 1.618-2.618 BC
        - AD = 0.886 XA
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 3 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        for i in range(len(primary_swings) - 2):
            for j in range(len(secondary_swings) - 1):
                if not (primary_swings[i].index < secondary_swings[j].index < primary_swings[i+1].index < 
                       secondary_swings[j+1].index < primary_swings[i+2].index):
                    continue
                
                X, A, B, C, D = primary_swings[i], secondary_swings[j], primary_swings[i+1], secondary_swings[j+1], primary_swings[i+2]
                
                XA = abs(A.price - X.price)
                AB = abs(B.price - A.price)
                BC = abs(C.price - B.price)
                CD = abs(D.price - C.price)
                AD = abs(D.price - A.price)
                
                if XA == 0 or AB == 0 or BC == 0:
                    continue
                
                ratio_AB_XA = AB / XA
                ratio_BC_AB = BC / AB
                ratio_CD_BC = CD / BC
                ratio_AD_XA = AD / XA
                
                if (0.382 <= ratio_AB_XA <= 0.5 and
                    0.382 <= ratio_BC_AB <= 0.886 and
                    1.618 <= ratio_CD_BC <= 2.618 and
                    self._check_fib_ratio(ratio_AD_XA, 0.886)):
                    
                    q1 = self._calculate_ratio_quality(ratio_AB_XA, 0.441)
                    q2 = self._calculate_ratio_quality(ratio_BC_AB, 0.618)
                    q3 = self._calculate_ratio_quality(ratio_CD_BC, 2.0)
                    q4 = self._calculate_ratio_quality(ratio_AD_XA, 0.886)
                    strength = (q1 + q2 + q3 + q4) / 4
                    
                    if D.index < len(signals):
                        signals[D.index] = 1
                        strengths[D.index] = max(strengths[D.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_crab(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                   pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Crab形态检测
        
        比率要求：
        - AB = 0.382-0.618 XA
        - BC = 0.382-0.886 AB
        - CD = 2.618-3.618 BC
        - AD = 1.618 XA
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 3 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        for i in range(len(primary_swings) - 2):
            for j in range(len(secondary_swings) - 1):
                if not (primary_swings[i].index < secondary_swings[j].index < primary_swings[i+1].index < 
                       secondary_swings[j+1].index < primary_swings[i+2].index):
                    continue
                
                X, A, B, C, D = primary_swings[i], secondary_swings[j], primary_swings[i+1], secondary_swings[j+1], primary_swings[i+2]
                
                XA = abs(A.price - X.price)
                AB = abs(B.price - A.price)
                BC = abs(C.price - B.price)
                CD = abs(D.price - C.price)
                AD = abs(D.price - A.price)
                
                if XA == 0 or AB == 0 or BC == 0:
                    continue
                
                ratio_AB_XA = AB / XA
                ratio_BC_AB = BC / AB
                ratio_CD_BC = CD / BC
                ratio_AD_XA = AD / XA
                
                if (0.382 <= ratio_AB_XA <= 0.618 and
                    0.382 <= ratio_BC_AB <= 0.886 and
                    2.618 <= ratio_CD_BC <= 3.618 and
                    self._check_fib_ratio(ratio_AD_XA, 1.618)):
                    
                    q1 = self._calculate_ratio_quality(ratio_AB_XA, 0.5)
                    q2 = self._calculate_ratio_quality(ratio_BC_AB, 0.618)
                    q3 = self._calculate_ratio_quality(ratio_CD_BC, 3.0)
                    q4 = self._calculate_ratio_quality(ratio_AD_XA, 1.618)
                    strength = (q1 + q2 + q3 + q4) / 4
                    
                    if D.index < len(signals):
                        signals[D.index] = 1
                        strengths[D.index] = max(strengths[D.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_shark(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                    pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Shark形态检测
        
        比率要求：
        - AB = 1.13-1.618 OX (O为起点)
        - BC = 1.618-2.24 AB
        - CD = 0.886-1.13 OX
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 3 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        for i in range(len(primary_swings) - 2):
            for j in range(len(secondary_swings) - 1):
                if not (primary_swings[i].index < secondary_swings[j].index < primary_swings[i+1].index < 
                       secondary_swings[j+1].index < primary_swings[i+2].index):
                    continue
                
                O, X, A, B, C = primary_swings[i], secondary_swings[j], primary_swings[i+1], secondary_swings[j+1], primary_swings[i+2]
                
                OX = abs(X.price - O.price)
                AB = abs(A.price - X.price)
                BC = abs(B.price - A.price)
                CD = abs(C.price - B.price)
                
                if OX == 0 or AB == 0 or BC == 0:
                    continue
                
                ratio_AB_OX = AB / OX
                ratio_BC_AB = BC / AB
                ratio_CD_OX = CD / OX
                
                if (1.13 <= ratio_AB_OX <= 1.618 and
                    1.618 <= ratio_BC_AB <= 2.24 and
                    0.886 <= ratio_CD_OX <= 1.13):
                    
                    q1 = self._calculate_ratio_quality(ratio_AB_OX, 1.37)
                    q2 = self._calculate_ratio_quality(ratio_BC_AB, 1.90)
                    q3 = self._calculate_ratio_quality(ratio_CD_OX, 1.0)
                    strength = (q1 + q2 + q3) / 3
                    
                    if C.index < len(signals):
                        signals[C.index] = 1
                        strengths[C.index] = max(strengths[C.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_cypher(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                     pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Cypher形态检测
        
        比率要求：
        - AB = 0.382-0.618 XA
        - BC = 1.272-1.414 XA
        - CD = 0.786 XC
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 3 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        for i in range(len(primary_swings) - 2):
            for j in range(len(secondary_swings) - 1):
                if not (primary_swings[i].index < secondary_swings[j].index < primary_swings[i+1].index < 
                       secondary_swings[j+1].index < primary_swings[i+2].index):
                    continue
                
                X, A, B, C, D = primary_swings[i], secondary_swings[j], primary_swings[i+1], secondary_swings[j+1], primary_swings[i+2]
                
                XA = abs(A.price - X.price)
                AB = abs(B.price - A.price)
                BC = abs(C.price - B.price)
                XC = abs(C.price - X.price)
                CD = abs(D.price - C.price)
                
                if XA == 0 or XC == 0:
                    continue
                
                ratio_AB_XA = AB / XA
                ratio_BC_XA = BC / XA
                ratio_CD_XC = CD / XC
                
                if (0.382 <= ratio_AB_XA <= 0.618 and
                    1.272 <= ratio_BC_XA <= 1.414 and
                    self._check_fib_ratio(ratio_CD_XC, 0.786)):
                    
                    q1 = self._calculate_ratio_quality(ratio_AB_XA, 0.5)
                    q2 = self._calculate_ratio_quality(ratio_BC_XA, 1.343)
                    q3 = self._calculate_ratio_quality(ratio_CD_XC, 0.786)
                    strength = (q1 + q2 + q3) / 3
                    
                    if D.index < len(signals):
                        signals[D.index] = 1
                        strengths[D.index] = max(strengths[D.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_abcd(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                   pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        ABCD形态检测
        
        比率要求：
        - BC = 0.382-0.886 AB
        - CD = 1.272-1.618 AB
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 2 or len(secondary_swings) < 2:
            return pd.Series(0, index=data_index), pd.Series(0.0, index=data_index)
        
        for i in range(len(secondary_swings) - 1):
            for j in range(len(primary_swings) - 1):
                if not (secondary_swings[i].index < primary_swings[j].index < 
                       secondary_swings[i+1].index < primary_swings[j+1].index):
                    continue
                
                A, B, C, D = secondary_swings[i], primary_swings[j], secondary_swings[i+1], primary_swings[j+1]
                
                AB = abs(B.price - A.price)
                BC = abs(C.price - B.price)
                CD = abs(D.price - C.price)
                
                if AB == 0:
                    continue
                
                ratio_BC_AB = BC / AB
                ratio_CD_AB = CD / AB
                
                if (0.382 <= ratio_BC_AB <= 0.886 and
                    1.272 <= ratio_CD_AB <= 1.618):
                    
                    q1 = self._calculate_ratio_quality(ratio_BC_AB, 0.618)
                    q2 = self._calculate_ratio_quality(ratio_CD_AB, 1.414)
                    strength = (q1 + q2) / 2
                    
                    if D.index < len(signals):
                        signals[D.index] = 1
                        strengths[D.index] = max(strengths[D.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def detect_three_drives(self, primary_swings: List[SwingPoint], secondary_swings: List[SwingPoint], 
                           pattern_type: str, data_index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Three Drives形态检测
        
        比率要求：
        - Drive 2 = 1.272-1.618 Drive 1
        - Drive 3 = 1.272-1.618 Drive 2
        """
        signals = np.zeros(len(data_index))
        strengths = np.zeros(len(data_index))
        
        if len(primary_swings) < 4:
            return pd.Series(signals), pd.Series(strengths)
        
        for i in range(len(primary_swings) - 3):
            D1, D2, D3, D4 = primary_swings[i:i+4]
            
            if not (D1.index < D2.index < D3.index < D4.index):
                continue
            
            drive1 = abs(D2.price - D1.price)
            drive2 = abs(D3.price - D2.price)
            drive3 = abs(D4.price - D3.price)
            
            if drive1 == 0 or drive2 == 0:
                continue
            
            ratio_D2_D1 = drive2 / drive1
            ratio_D3_D2 = drive3 / drive2
            
            if (1.272 <= ratio_D2_D1 <= 1.618 and
                1.272 <= ratio_D3_D2 <= 1.618):
                
                q1 = self._calculate_ratio_quality(ratio_D2_D1, 1.414)
                q2 = self._calculate_ratio_quality(ratio_D3_D2, 1.414)
                strength = (q1 + q2) / 2
                
                if D4.index < len(signals):
                    signals[D4.index] = 1
                    strengths[D4.index] = max(strengths[D4.index], strength)
        
        return pd.Series(signals), pd.Series(strengths)
    
    def calculate_pattern_score(self, pattern_points: Dict, pattern_type: str) -> Dict[str, float]:
        """
        综合形态评分系统
        
        评分维度：
        1. Fibonacci比率匹配度 (40%)
        2. 形态对称性 (20%)
        3. 时间对称性 (20%)
        4. 价格幅度合理性 (10%)
        5. 形态完整性 (10%)
        
        Args:
            pattern_points: 形态关键点字典 {'X': price, 'A': price, ...}
            pattern_type: 形态类型 ('gartley', 'butterfly', 等)
            
        Returns:
            Dict包含各维度分数和总分
        """
        scores = {}
        
        try:
            # 1. Fibonacci比率匹配度（已有基础实现）
            # 这里可以获取各个关键比率的质量分数
            scores['fibonacci_accuracy'] = 0.0  # 占40%
            
            # 2. 形态对称性（检查形态是否对称）
            if 'X' in pattern_points and 'D' in pattern_points:
                # XA和CD的时间跨度应该相近
                # 这里简化为价格幅度的对称性
                XA = abs(pattern_points.get('A', 0) - pattern_points.get('X', 0))
                CD = abs(pattern_points.get('D', 0) - pattern_points.get('C', 0))
                if XA > 0:
                    symmetry = 1 - abs(CD - XA) / XA
                    scores['symmetry'] = max(0, min(1, symmetry))  # 占20%
                else:
                    scores['symmetry'] = 0.0
            else:
                scores['symmetry'] = 0.0
            
            # 3. 时间对称性（暂时简化处理）
            scores['time_symmetry'] = 0.5  # 占20%，默认0.5
            
            # 4. 价格幅度合理性
            # 检查价格变化是否在合理范围内
            if 'X' in pattern_points and 'D' in pattern_points:
                total_range = abs(pattern_points['D'] - pattern_points['X'])
                avg_price = (pattern_points['D'] + pattern_points['X']) / 2
                if avg_price > 0:
                    range_ratio = total_range / avg_price
                    # 合理的价格变化应该在0.5%-20%之间
                    if 0.005 <= range_ratio <= 0.20:
                        scores['price_amplitude'] = 1.0  # 占10%
                    elif range_ratio < 0.005:
                        scores['price_amplitude'] = range_ratio / 0.005
                    else:
                        scores['price_amplitude'] = 0.20 / range_ratio
                else:
                    scores['price_amplitude'] = 0.0
            else:
                scores['price_amplitude'] = 0.0
            
            # 5. 形态完整性（所有关键点都存在）
            required_points = ['X', 'A', 'B', 'C', 'D']
            if pattern_type == 'abcd':
                required_points = ['A', 'B', 'C', 'D']
            
            present_points = sum(1 for p in required_points if p in pattern_points and pattern_points[p] is not None)
            scores['completeness'] = present_points / len(required_points)  # 占10%
            
            # 加权总分
            total_score = (
                scores['fibonacci_accuracy'] * 0.40 +
                scores['symmetry'] * 0.20 +
                scores['time_symmetry'] * 0.20 +
                scores['price_amplitude'] * 0.10 +
                scores['completeness'] * 0.10
            )
            
            scores['total'] = total_score
            
        except Exception as e:
            # 出错时返回低分
            scores = {
                'fibonacci_accuracy': 0.0,
                'symmetry': 0.0,
                'time_symmetry': 0.0,
                'price_amplitude': 0.0,
                'completeness': 0.0,
                'total': 0.0
            }
        
        return scores
    
    def check_pattern_failure(self, pattern_points: Dict, current_price: float, 
                             pattern_type: str = 'gartley') -> Tuple[bool, str]:
        """
        形态失败检测
        
        检测形态是否被突破/失效
        
        失效条件：
        1. 价格突破关键点位（如D点）
        2. 形态结构被破坏
        3. 超出合理时间范围
        
        Args:
            pattern_points: 形态关键点
            current_price: 当前价格
            pattern_type: 形态类型
            
        Returns:
            (is_failed, failure_reason)
        """
        try:
            # 检查D点是否被突破（最关键的失败条件）
            if 'D' in pattern_points:
                D_price = pattern_points['D']
                X_price = pattern_points.get('X', D_price)
                
                # 判断形态方向（看涨或看跌）
                is_bullish = D_price < X_price
                
                if is_bullish:
                    # 看涨形态：当前价格跌破D点视为失败
                    if current_price < D_price * 0.98:  # 2%缓冲区
                        return True, "Price broke below D point (bullish pattern failed)"
                else:
                    # 看跌形态：当前价格突破D点视为失败
                    if current_price > D_price * 1.02:  # 2%缓冲区
                        return True, "Price broke above D point (bearish pattern failed)"
            
            # 检查是否存在结构性失败
            # 例如：B点被重新测试
            if 'B' in pattern_points and 'C' in pattern_points:
                B_price = pattern_points['B']
                C_price = pattern_points['C']
                
                is_bullish = C_price < B_price
                
                if is_bullish:
                    # 看涨形态：价格不应跌破C点
                    if current_price < C_price * 0.98:
                        return True, "Price broke structure at C point"
                else:
                    # 看跌形态：价格不应突破C点
                    if current_price > C_price * 1.02:
                        return True, "Price broke structure at C point"
            
            # 形态仍然有效
            return False, "Pattern is valid"
            
        except Exception as e:
            # 出错时保守地认为形态可能失效
            return True, f"Pattern check failed: {str(e)}"


def main():
    """测试Harmonic Patterns完整实现"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'close': close_prices,
        'high': close_prices * 1.01, 
        'low': close_prices * 0.99,
        'open': close_prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    hp = HarmonicPatterns(swing_window=5, base_tolerance=0.05, use_dynamic_tolerance=True)
    features = hp.calculate_all(ohlcv)
    print(f"Generated {features.shape[1]} harmonic pattern features")
    print(f"Feature columns: {features.columns.tolist()}")
    
    # 统计信号数量
    for col in features.columns:
        if not 'strength' in col:
            count = features[col].sum()
            if count > 0:
                print(f"{col}: {count} signals detected")


if __name__ == '__main__':
    main()
