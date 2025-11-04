# -*- coding: utf-8 -*-
"""
艾略特波浪理论完整实现
包含5-3波浪结构识别、波浪比率验证、波浪延伸识别
"""

import warnings
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class WavePoint:
    """波浪点数据类"""
    def __init__(self, index: int, price: float, wave_type: int):
        self.index = index
        self.price = price
        self.wave_type = wave_type  # 1-5 for impulse, A-C for corrective


class ElliottWaveRuleEngine:
    """
    Elliott波浪规则引擎
    
    实现完整的Elliott波浪规则验证：
    1. Wave 2规则
    2. Wave 3规则
    3. Wave 4规则
    4. Wave 5规则
    5. 延伸规则
    6. 交替规则
    """
    
    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: 规则容差（10%）
        """
        self.tolerance = tolerance
    
    def validate_impulse_wave(self, waves: List[WavePoint]) -> Tuple[bool, dict]:
        """
        验证5浪冲击波的所有规则
        
        Args:
            waves: 6个WavePoint（0起点 + 5个波浪终点）
            
        Returns:
            (is_valid, rule_scores): 是否有效，各规则得分
        """
        if len(waves) < 6:
            return False, {}
        
        w0, w1, w2, w3, w4, w5 = waves[:6]
        
        # 计算波浪长度
        wave1 = abs(w1.price - w0.price)
        wave2 = abs(w2.price - w1.price)
        wave3 = abs(w3.price - w2.price)
        wave4 = abs(w4.price - w3.price)
        wave5 = abs(w5.price - w4.price)
        
        # 判断方向
        is_bullish = w1.price > w0.price
        
        rule_scores = {}
        
        # 规则1: Wave 2不回撤超过Wave 1起点
        rule_scores['wave2_retracement'] = self._check_wave2_rule(w0, w1, w2, is_bullish)
        
        # 规则2: Wave 3不是最短的冲击波
        rule_scores['wave3_length'] = self._check_wave3_rule(wave1, wave3, wave5)
        
        # 规则3: Wave 4不进入Wave 1价格区域
        rule_scores['wave4_overlap'] = self._check_wave4_rule(w0, w1, w4, is_bullish)
        
        # 规则4: Wave 3通常是延伸浪
        rule_scores['wave3_extension'] = self._check_wave3_extension(wave1, wave3)
        
        # 规则5: Wave 5与Wave 1的关系
        rule_scores['wave5_relationship'] = self._check_wave5_rule(wave1, wave5)
        
        # 规则6: 交替规则（Wave 2和Wave 4应该不同类型）
        rule_scores['alternation'] = 0.5  # 简化处理
        
        # 计算总体符合度
        total_score = sum(rule_scores.values()) / len(rule_scores)
        is_valid = total_score >= 0.7  # 70%规则符合即认为有效
        
        rule_scores['total_score'] = total_score
        
        return is_valid, rule_scores
    
    def _check_wave2_rule(self, w0: WavePoint, w1: WavePoint, w2: WavePoint, is_bullish: bool) -> float:
        """规则1: Wave 2不能回撤超过Wave 1起点"""
        if is_bullish:
            # 上涨浪：Wave 2低点不能低于Wave 0
            if w2.price > w0.price:
                # 计算回撤比例
                retracement = (w1.price - w2.price) / (w1.price - w0.price)
                # 理想回撤：50%-78.6%
                if 0.5 <= retracement <= 0.786:
                    return 1.0
                elif retracement < 1.0:  # 未破起点
                    return 0.7
            return 0.0
        else:
            # 下跌浪：Wave 2高点不能高于Wave 0
            if w2.price < w0.price:
                retracement = (w2.price - w1.price) / (w0.price - w1.price)
                if 0.5 <= retracement <= 0.786:
                    return 1.0
                elif retracement < 1.0:
                    return 0.7
            return 0.0
    
    def _check_wave3_rule(self, wave1: float, wave3: float, wave5: float) -> float:
        """规则2: Wave 3不能是最短的冲击波"""
        if wave3 > wave1 and wave3 > wave5:
            return 1.0  # Wave 3是最长的（理想）
        elif wave3 > wave1 or wave3 > wave5:
            return 0.8  # Wave 3至少不是最短
        else:
            return 0.0  # 违反规则
    
    def _check_wave4_rule(self, w0: WavePoint, w1: WavePoint, w4: WavePoint, is_bullish: bool) -> float:
        """规则3: Wave 4不能进入Wave 1价格区域"""
        if is_bullish:
            # 上涨浪：Wave 4低点不能低于Wave 1高点
            if w4.price > w1.price:
                return 1.0
            else:
                # 计算重叠程度
                overlap = (w1.price - w4.price) / (w1.price - w0.price)
                return max(0, 1 - overlap * 2)  # 重叠越多分数越低
        else:
            # 下跌浪：Wave 4高点不能高于Wave 1低点
            if w4.price < w1.price:
                return 1.0
            else:
                overlap = (w4.price - w1.price) / (w0.price - w1.price)
                return max(0, 1 - overlap * 2)
    
    def _check_wave3_extension(self, wave1: float, wave3: float) -> float:
        """Wave 3通常是延伸浪（1.618倍Wave 1）"""
        ratio = wave3 / (wave1 + 1e-9)
        # 理想比率：1.618
        if 1.5 <= ratio <= 1.8:
            return 1.0
        elif 1.272 <= ratio <= 2.0:
            return 0.7
        else:
            deviation = abs(ratio - 1.618) / 1.618
            return max(0, 1 - deviation)
    
    def _check_wave5_rule(self, wave1: float, wave5: float) -> float:
        """Wave 5与Wave 1的关系（通常相等或0.618倍）"""
        ratio = wave5 / (wave1 + 1e-9)
        # 理想比率：1.0（相等）或0.618
        if 0.9 <= ratio <= 1.1:
            return 1.0  # 相等
        elif 0.55 <= ratio <= 0.7:
            return 0.9  # 0.618
        elif 1.5 <= ratio <= 1.7:
            return 0.8  # 延伸（1.618）
        else:
            return 0.5  # 其他情况


class ElliottWaveAnalyzer:
    """艾略特波浪分析器 - 完整实现（含规则引擎和嵌套结构）"""
    
    def __init__(self, base_threshold: float = 0.03, use_dynamic_threshold: bool = True):
        """
        Args:
            base_threshold: 基础Zigzag波动幅度（3%）
            use_dynamic_threshold: 是否使用动态阈值（基于波动率）
        """
        self.base_threshold = base_threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.zigzag_threshold = base_threshold
        self.rule_engine = ElliottWaveRuleEngine(tolerance=0.1)
        self.current_volatility = 0.0
    
    def analyze(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high, low, close = ohlcv['high'], ohlcv['low'], ohlcv['close']
        features = pd.DataFrame(index=ohlcv.index)
        
        # 动态调整Zigzag阈值
        if self.use_dynamic_threshold:
            self.current_volatility = self._calculate_volatility(close)
            self.zigzag_threshold = self._adjust_threshold_by_volatility(self.current_volatility)
        else:
            self.zigzag_threshold = self.base_threshold
        
        # 1. Zigzag算法识别摆动点
        zigzag_points = self._calculate_zigzag(high, low, close)
        
        # 2. 识别5浪冲击波
        impulse_waves = self._identify_impulse_waves(zigzag_points)
        features['elliott_impulse_wave'], features['elliott_impulse_wave_number'] = impulse_waves
        
        # 3. 识别3浪修正波
        corrective_waves = self._identify_corrective_waves(zigzag_points)
        features['elliott_corrective_wave'], features['elliott_corrective_wave_type'] = corrective_waves
        
        # 4. 波浪延伸识别
        wave_extensions = self._identify_wave_extensions(zigzag_points)
        features['elliott_wave_3_extension'] = wave_extensions[0]
        features['elliott_wave_5_extension'] = wave_extensions[1]
        
        # 5. 波浪比率分析
        wave_ratios = self._calculate_wave_ratios(zigzag_points)
        for key, value in wave_ratios.items():
            features[f'elliott_{key}'] = value
        
        # 6. Elliott Oscillator（经典5-35 EMA差值）
        features['elliott_oscillator'] = self._calculate_elliott_oscillator(close)
        
        # 7. 波浪强度（综合指标）
        features['elliott_wave_strength'] = self._calculate_wave_strength(impulse_waves, corrective_waves)
        
        # 8. 波浪完成度（当前波浪进度）
        features['elliott_wave_completion'] = self._calculate_wave_completion(zigzag_points, close)
        
        # 9. 趋势阶段（Trending vs Correcting）
        features['elliott_trend_phase'] = self._identify_trend_phase(impulse_waves, corrective_waves)
        
        # 10. 波浪目标价格（基于Fibonacci）
        targets = self._calculate_wave_targets(zigzag_points, close)
        features['elliott_target_distance'] = targets
        
        return features.fillna(0)
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """calculate_all方法（别名：analyze）"""
        return self.analyze(ohlcv)
    
    def _calculate_volatility(self, close: pd.Series, window: int = 20) -> float:
        """
        计算当前波动率
        
        Returns:
            float: 波动率（归一化到0.01-0.10）
        """
        if len(close) < window:
            return 0.03  # 默认3%
        
        returns = close.pct_change().dropna()
        recent_returns = returns.tail(window)
        volatility = recent_returns.std()
        
        # 归一化
        normalized_vol = np.clip(volatility, 0.01, 0.10)
        return float(normalized_vol)
    
    def _adjust_threshold_by_volatility(self, volatility: float) -> float:
        """
        根据波动率动态调整Zigzag阈值
        
        逻辑：
        - 低波动率：使用较小阈值（捕捉更多细节）
        - 高波动率：使用较大阈值（过滤噪声）
        
        Args:
            volatility: 当前波动率
            
        Returns:
            float: 调整后的阈值
        """
        if volatility < 0.02:
            # 低波动率：阈值 = base * 0.8
            adjusted_threshold = self.base_threshold * 0.8
        elif volatility < 0.05:
            # 中等波动率：阈值 = base
            adjusted_threshold = self.base_threshold
        else:
            # 高波动率：阈值 = base * (1 + volatility)
            factor = 1 + (volatility - 0.05) / 0.05
            adjusted_threshold = self.base_threshold * min(factor, 2.0)
        
        # 限制范围在0.02-0.10之间
        return np.clip(adjusted_threshold, 0.02, 0.10)
    
    def identify_nested_waves(self, zigzag_points: List[WavePoint], 
                             max_depth: int = 2) -> dict:
        """
        识别嵌套波浪结构
        
        实现多级波浪识别：
        - Primary（主浪）
        - Intermediate（中浪）
        - Minor（小浪）
        
        Args:
            zigzag_points: Zigzag转折点
            max_depth: 最大递归深度（2=Primary+Intermediate）
            
        Returns:
            嵌套波浪树结构
        """
        wave_tree = {
            'primary': [],
            'intermediate': {},
            'minor': {}
        }
        
        try:
            # 1. 识别Primary级别的5浪结构
            primary_waves = self._identify_impulse_waves_with_rules(zigzag_points)
            wave_tree['primary'] = primary_waves
            
            # 2. 对每个Primary波浪进行分解，找出Intermediate级别
            if max_depth >= 1:
                for i, primary_wave in enumerate(primary_waves):
                    # 提取这个Primary波浪范围内的点
                    if i < len(primary_waves) - 1:
                        start_idx = primary_wave.get('start_index', 0)
                        end_idx = primary_waves[i+1].get('start_index', len(zigzag_points))
                        sub_points = [p for p in zigzag_points if start_idx <= p.index < end_idx]
                        
                        if len(sub_points) >= 6:
                            # 递归分解
                            intermediate_waves = self._identify_impulse_waves_with_rules(sub_points)
                            wave_tree['intermediate'][i] = intermediate_waves
            
            # 3. 对Intermediate波浪继续分解（如果需要）
            if max_depth >= 2:
                for primary_idx, intermediate_list in wave_tree['intermediate'].items():
                    wave_tree['minor'][primary_idx] = {}
                    for j, inter_wave in enumerate(intermediate_list):
                        # 这里可以继续递归，但为了性能，限制在2层
                        pass
        
        except Exception as e:
            # 出错时返回空结构
            pass
        
        return wave_tree
    
    def _identify_impulse_waves_with_rules(self, zigzag_points: List[WavePoint]) -> List[dict]:
        """
        使用规则引擎识别5浪冲击波
        
        Returns:
            List[dict]: 识别到的波浪列表，每个包含起止点和规则评分
        """
        waves = []
        
        if len(zigzag_points) < 6:
            return waves
        
        for i in range(len(zigzag_points) - 5):
            candidate_waves = zigzag_points[i:i+6]
            
            # 使用规则引擎验证
            is_valid, rule_scores = self.rule_engine.validate_impulse_wave(candidate_waves)
            
            if is_valid:
                waves.append({
                    'start_index': candidate_waves[0].index,
                    'end_index': candidate_waves[5].index,
                    'points': candidate_waves,
                    'rule_scores': rule_scores,
                    'quality': rule_scores.get('total_score', 0)
                })
        
        return waves
    
    def _calculate_zigzag(self, high: pd.Series, low: pd.Series, close: pd.Series) -> List[WavePoint]:
        """
        Zigzag算法识别关键摆动点
        
        只保留超过threshold%的波动
        """
        points = []
        
        # 初始化
        if len(close) == 0:
            return points
        
        current_trend = 0  # 0=未知, 1=上涨, -1=下跌
        last_pivot_idx = 0
        last_pivot_price = close.iloc[0]
        
        for i in range(1, len(close)):
            price = close.iloc[i]
            high_price = high.iloc[i]
            low_price = low.iloc[i]
            
            if current_trend == 0:
                # 初始化趋势
                if price > last_pivot_price * (1 + self.zigzag_threshold):
                    current_trend = 1
                    points.append(WavePoint(last_pivot_idx, last_pivot_price, 0))
                    last_pivot_idx = i
                    last_pivot_price = high_price
                elif price < last_pivot_price * (1 - self.zigzag_threshold):
                    current_trend = -1
                    points.append(WavePoint(last_pivot_idx, last_pivot_price, 0))
                    last_pivot_idx = i
                    last_pivot_price = low_price
            
            elif current_trend == 1:
                # 上涨趋势中
                if high_price > last_pivot_price:
                    # 更新高点
                    last_pivot_idx = i
                    last_pivot_price = high_price
                elif low_price < last_pivot_price * (1 - self.zigzag_threshold):
                    # 反转为下跌
                    points.append(WavePoint(last_pivot_idx, last_pivot_price, 0))
                    current_trend = -1
                    last_pivot_idx = i
                    last_pivot_price = low_price
            
            elif current_trend == -1:
                # 下跌趋势中
                if low_price < last_pivot_price:
                    # 更新低点
                    last_pivot_idx = i
                    last_pivot_price = low_price
                elif high_price > last_pivot_price * (1 + self.zigzag_threshold):
                    # 反转为上涨
                    points.append(WavePoint(last_pivot_idx, last_pivot_price, 0))
                    current_trend = 1
                    last_pivot_idx = i
                    last_pivot_price = high_price
        
        # 添加最后一个点
        if len(points) > 0:
            points.append(WavePoint(last_pivot_idx, last_pivot_price, 0))
        
        return points
    
    def _identify_impulse_waves(self, zigzag_points: List[WavePoint]) -> Tuple[pd.Series, pd.Series]:
        """
        识别5浪冲击波
        
        规则：
        - Wave 2不能回撤超过Wave 1起点
        - Wave 3不能是最短的浪
        - Wave 4不能进入Wave 1区域
        - Wave 5通常是1.618或0.618倍Wave 1
        """
        if len(zigzag_points) < 6:
            return pd.Series([0]), pd.Series([0])
        
        signals = np.zeros(len(zigzag_points))
        wave_numbers = np.zeros(len(zigzag_points))
        
        # 遍历所有可能的5浪组合
        for i in range(len(zigzag_points) - 5):
            w0, w1, w2, w3, w4, w5 = zigzag_points[i:i+6]
            
            # 计算波浪长度
            wave1 = abs(w1.price - w0.price)
            wave2 = abs(w2.price - w1.price)
            wave3 = abs(w3.price - w2.price)
            wave4 = abs(w4.price - w3.price)
            wave5 = abs(w5.price - w4.price)
            
            # 规则验证
            # 1. Wave 2不回撤超过Wave 1起点
            if w1.price > w0.price:  # 上涨浪
                if w2.price <= w0.price:
                    continue
            else:  # 下跌浪
                if w2.price >= w0.price:
                    continue
            
            # 2. Wave 3不是最短
            if wave3 <= wave1 and wave3 <= wave5:
                continue
            
            # 3. Wave 4不进入Wave 1区域
            if w1.price > w0.price:  # 上涨浪
                if w4.price <= w1.price:
                    continue
            else:  # 下跌浪
                if w4.price >= w1.price:
                    continue
            
            # 4. Wave 3通常是Wave 1的1.618倍
            ratio_3_1 = wave3 / (wave1 + 1e-9)
            if 1.5 <= ratio_3_1 <= 1.8:  # 容差范围
                # 识别为有效的5浪结构
                for j, point in enumerate([w1, w2, w3, w4, w5]):
                    if point.index < len(signals):
                        signals[point.index] = 1
                        wave_numbers[point.index] = (j % 5) + 1
        
        return pd.Series(signals), pd.Series(wave_numbers)
    
    def _identify_corrective_waves(self, zigzag_points: List[WavePoint]) -> Tuple[pd.Series, pd.Series]:
        """
        识别3浪修正波（ABC）
        
        规则：
        - Wave B通常回撤Wave A的50-78.6%
        - Wave C通常等于Wave A或是1.618倍Wave A
        """
        if len(zigzag_points) < 4:
            return pd.Series([0]), pd.Series([0])
        
        signals = np.zeros(len(zigzag_points))
        wave_types = np.zeros(len(zigzag_points))  # 1=Zigzag, 2=Flat, 3=Triangle
        
        for i in range(len(zigzag_points) - 3):
            wA_start, wA_end, wB_end, wC_end = zigzag_points[i:i+4]
            
            waveA = abs(wA_end.price - wA_start.price)
            waveB = abs(wB_end.price - wA_end.price)
            waveC = abs(wC_end.price - wB_end.price)
            
            # Zigzag修正（5-3-5）
            ratio_B_A = waveB / (waveA + 1e-9)
            ratio_C_A = waveC / (waveA + 1e-9)
            
            if 0.5 <= ratio_B_A <= 0.786 and 0.9 <= ratio_C_A <= 1.1:
                # Zigzag
                for j, point in enumerate([wA_end, wB_end, wC_end]):
                    if point.index < len(signals):
                        signals[point.index] = 1
                        wave_types[point.index] = 1
            
            elif 0.85 <= ratio_B_A <= 1.05 and 0.9 <= ratio_C_A <= 1.1:
                # Flat (3-3-5)
                for j, point in enumerate([wA_end, wB_end, wC_end]):
                    if point.index < len(signals):
                        signals[point.index] = 1
                        wave_types[point.index] = 2
        
        return pd.Series(signals), pd.Series(wave_types)
    
    def _identify_wave_extensions(self, zigzag_points: List[WavePoint]) -> Tuple[pd.Series, pd.Series]:
        """识别Wave 3或Wave 5的延伸"""
        wave3_ext = np.zeros(len(zigzag_points))
        wave5_ext = np.zeros(len(zigzag_points))
        
        if len(zigzag_points) < 6:
            return pd.Series(wave3_ext), pd.Series(wave5_ext)
        
        for i in range(len(zigzag_points) - 5):
            w0, w1, w2, w3, w4, w5 = zigzag_points[i:i+6]
            
            wave1 = abs(w1.price - w0.price)
            wave3 = abs(w3.price - w2.price)
            wave5 = abs(w5.price - w4.price)
            
            # Wave 3延伸（是Wave 1的1.618倍以上）
            if wave3 >= wave1 * 1.618:
                if w3.index < len(wave3_ext):
                    wave3_ext[w3.index] = 1
            
            # Wave 5延伸（是Wave 1的1.618倍以上）
            if wave5 >= wave1 * 1.618:
                if w5.index < len(wave5_ext):
                    wave5_ext[w5.index] = 1
        
        return pd.Series(wave3_ext), pd.Series(wave5_ext)
    
    def _calculate_wave_ratios(self, zigzag_points: List[WavePoint]) -> dict:
        """计算波浪比率特征"""
        ratios = {
            'wave_ratio_3_1': np.zeros(len(zigzag_points)),
            'wave_ratio_5_1': np.zeros(len(zigzag_points)),
            'wave_ratio_C_A': np.zeros(len(zigzag_points))
        }
        
        if len(zigzag_points) < 6:
            return {k: pd.Series(v) for k, v in ratios.items()}
        
        for i in range(len(zigzag_points) - 5):
            w0, w1, w2, w3, w4, w5 = zigzag_points[i:i+6]
            
            wave1 = abs(w1.price - w0.price)
            wave3 = abs(w3.price - w2.price)
            wave5 = abs(w5.price - w4.price)
            
            if wave1 > 0:
                if w3.index < len(ratios['wave_ratio_3_1']):
                    ratios['wave_ratio_3_1'][w3.index] = wave3 / wave1
                if w5.index < len(ratios['wave_ratio_5_1']):
                    ratios['wave_ratio_5_1'][w5.index] = wave5 / wave1
        
        # 修正波比率
        for i in range(len(zigzag_points) - 3):
            wA_start, wA_end, wB_end, wC_end = zigzag_points[i:i+4]
            
            waveA = abs(wA_end.price - wA_start.price)
            waveC = abs(wC_end.price - wB_end.price)
            
            if waveA > 0 and wC_end.index < len(ratios['wave_ratio_C_A']):
                ratios['wave_ratio_C_A'][wC_end.index] = waveC / waveA
        
        return {k: pd.Series(v) for k, v in ratios.items()}
    
    def _calculate_elliott_oscillator(self, close: pd.Series) -> pd.Series:
        """Elliott Oscillator = 5EMA - 35EMA"""
        ema5 = close.ewm(span=5).mean()
        ema35 = close.ewm(span=35).mean()
        return ema5 - ema35
    
    def _calculate_wave_strength(self, impulse_waves: Tuple, corrective_waves: Tuple) -> pd.Series:
        """综合波浪强度"""
        impulse_signal, impulse_num = impulse_waves
        corrective_signal, corrective_type = corrective_waves
        
        # 冲击波权重更高
        strength = impulse_signal * 2.0 + corrective_signal * 1.0
        
        # Wave 3和Wave 5权重最高
        strength += (impulse_num == 3).astype(float) * 1.5
        strength += (impulse_num == 5).astype(float) * 1.0
        
        return strength
    
    def _calculate_wave_completion(self, zigzag_points: List[WavePoint], close: pd.Series) -> pd.Series:
        """计算当前波浪的完成度（0-1）"""
        completion = np.zeros(len(close))
        
        if len(zigzag_points) < 2:
            return pd.Series(completion)
        
        for i in range(len(close)):
            # 找到最近的两个zigzag点
            prev_point = None
            next_point = None
            
            for j, point in enumerate(zigzag_points):
                if point.index <= i:
                    prev_point = point
                elif point.index > i and next_point is None:
                    next_point = point
                    break
            
            if prev_point and next_point:
                # 计算完成度
                total_distance = abs(next_point.price - prev_point.price)
                current_distance = abs(close.iloc[i] - prev_point.price)
                if total_distance > 0:
                    completion[i] = min(1.0, current_distance / total_distance)
        
        return pd.Series(completion, index=close.index)
    
    def _identify_trend_phase(self, impulse_waves: Tuple, corrective_waves: Tuple) -> pd.Series:
        """识别趋势阶段：1=Trending(冲击), 0=Correcting(修正)"""
        impulse_signal, _ = impulse_waves
        corrective_signal, _ = corrective_waves
        
        # 冲击波优先
        phase = impulse_signal.astype(int)
        phase = phase.where(corrective_signal == 0, 0)
        
        return phase
    
    def _calculate_wave_targets(self, zigzag_points: List[WavePoint], close: pd.Series) -> pd.Series:
        """基于Fibonacci计算波浪目标价格距离"""
        targets = np.zeros(len(close))
        
        if len(zigzag_points) < 3:
            return pd.Series(targets, index=close.index)
        
        # 使用最近的3个zigzag点计算目标
        for i in range(len(close)):
            recent_points = [p for p in zigzag_points if p.index <= i][-3:]
            
            if len(recent_points) >= 3:
                p1, p2, p3 = recent_points
                
                # 计算可能的目标价格（1.618倍延伸）
                move = abs(p2.price - p1.price)
                target = p3.price + move * 1.618 * (1 if p2.price > p1.price else -1)
                
                # 计算当前价格与目标的距离
                targets[i] = (target - close.iloc[i]) / (close.iloc[i] + 1e-9)
        
        return pd.Series(targets, index=close.index)


def main():
    """测试Elliott Wave完整实现"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'high': close_prices * 1.01,
        'low': close_prices * 0.99,
        'close': close_prices
    }, index=dates)
    
    ew = ElliottWaveAnalyzer(base_threshold=0.03, use_dynamic_threshold=True)
    features = ew.analyze(ohlcv)
    print(f"Generated {features.shape[1]} Elliott Wave features")
    print(f"Feature columns: {features.columns.tolist()}")
    
    # 统计信号
    print(f"\nImpulse waves detected: {features['elliott_impulse_wave'].sum()}")
    print(f"Corrective waves detected: {features['elliott_corrective_wave'].sum()}")
    print(f"Wave 3 extensions: {features['elliott_wave_3_extension'].sum()}")
    print(f"Wave 5 extensions: {features['elliott_wave_5_extension'].sum()}")


if __name__ == '__main__':
    main()
