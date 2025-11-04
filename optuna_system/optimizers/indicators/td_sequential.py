# -*- coding: utf-8 -*-
"""
TD Sequential完整实现
作者：Tom DeMark
包含：Setup、Countdown、Combo、Perfection、TDST等全部组件
"""

import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class TDSequential:
    """
    TD Sequential指标完整实现
    
    包含25+个特征：
    - Setup阶段（买入/卖出计数1-9+）
    - Countdown阶段（1-13）
    - Combo系统
    - Perfection信号
    - TDST支撑/阻力
    - TD REI（Range Expansion Index）
    - 信号强度和衰减特征
    """
    
    def __init__(self):
        pass
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有TD Sequential特征
        
        Args:
            ohlcv: OHLCV数据，必须包含open, high, low, close列
            
        Returns:
            包含所有TD特征的DataFrame
        """
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        close = ohlcv['close'].copy()
        high = ohlcv['high'].copy()
        low = ohlcv['low'].copy()
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. Setup阶段
        setup_buy, setup_sell = self.calculate_setup(close)
        features['td_setup_buy'] = setup_buy
        features['td_setup_sell'] = setup_sell
        
        # 2. TD9信号（Setup完成）
        features['td_9_buy_signal'] = (setup_buy == 9).astype(int)
        features['td_9_sell_signal'] = (setup_sell == 9).astype(int)
        
        # 3. TD9 Perfection
        perf_buy, perf_sell = self.detect_perfection(close, high, low, 
                                                     features['td_9_buy_signal'],
                                                     features['td_9_sell_signal'])
        features['td_9_buy_perfection'] = perf_buy
        features['td_9_sell_perfection'] = perf_sell
        
        # 4. Countdown阶段
        countdown_buy, countdown_sell = self.calculate_countdown(close, high, low, 
                                                                 setup_buy, setup_sell)
        features['td_countdown_buy'] = countdown_buy
        features['td_countdown_sell'] = countdown_sell
        
        # 5. TD13信号（Countdown完成）
        features['td_13_buy_signal'] = (countdown_buy == 13).astype(int)
        features['td_13_sell_signal'] = (countdown_sell == 13).astype(int)
        
        # 6. Combo系统
        combo_buy, combo_sell = self.calculate_combo(close)
        features['td_combo_buy'] = combo_buy
        features['td_combo_sell'] = combo_sell
        features['td_combo_13_buy'] = (combo_buy == 13).astype(int)
        features['td_combo_13_sell'] = (combo_sell == 13).astype(int)
        
        # 7. TDST支撑/阻力
        tdst_support, tdst_resistance = self.calculate_tdst(close, high, low, 
                                                            features['td_9_buy_signal'],
                                                            features['td_9_sell_signal'])
        features['tdst_support'] = tdst_support
        features['tdst_resistance'] = tdst_resistance
        features['tdst_support_distance'] = (close - tdst_support) / (close + 1e-9)
        features['tdst_resistance_distance'] = (tdst_resistance - close) / (close + 1e-9)
        
        # 8. TD REI (Range Expansion Index)
        rei = self.calculate_td_rei(close, high, low)
        features['td_rei'] = rei
        
        # 9. TD信号强度（综合指标）
        features['td_buy_strength'] = self._calculate_signal_strength(
            features['td_9_buy_signal'],
            features['td_9_buy_perfection'],
            features['td_13_buy_signal'],
            features['td_combo_13_buy']
        )
        features['td_sell_strength'] = self._calculate_signal_strength(
            features['td_9_sell_signal'],
            features['td_9_sell_perfection'],
            features['td_13_sell_signal'],
            features['td_combo_13_sell']
        )
        
        # 10. 距离上次信号的K线数（衰减特征）
        features['td_bars_since_9_buy'] = self._calculate_bars_since_signal(
            features['td_9_buy_signal']
        )
        features['td_bars_since_9_sell'] = self._calculate_bars_since_signal(
            features['td_9_sell_signal']
        )
        features['td_bars_since_13_buy'] = self._calculate_bars_since_signal(
            features['td_13_buy_signal']
        )
        features['td_bars_since_13_sell'] = self._calculate_bars_since_signal(
            features['td_13_sell_signal']
        )
        
        # 11. Setup活跃状态（Setup在1-8之间）
        features['td_setup_buy_active'] = ((setup_buy > 0) & (setup_buy < 9)).astype(int)
        features['td_setup_sell_active'] = ((setup_sell > 0) & (setup_sell < 9)).astype(int)
        
        # 12. Countdown活跃状态
        features['td_countdown_buy_active'] = ((countdown_buy > 0) & (countdown_buy < 13)).astype(int)
        features['td_countdown_sell_active'] = ((countdown_sell > 0) & (countdown_sell < 13)).astype(int)
        
        # Shift所有信号1期（防止look-ahead bias）
        signal_cols = [c for c in features.columns if 'signal' in c or 'strength' in c or 'perfection' in c]
        features[signal_cols] = features[signal_cols].shift(1).fillna(0)
        
        # 清理NaN和Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def calculate_setup(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算TD Setup
        
        买入Setup：当前收盘 < 4根前收盘，计数+1（最多到9+）
        卖出Setup：当前收盘 > 4根前收盘，计数+1（最多到9+）
        
        Args:
            close: 收盘价序列
            
        Returns:
            (setup_buy, setup_sell): 买入和卖出Setup计数
        """
        setup_buy = []
        setup_sell = []
        buy_count = 0
        sell_count = 0
        
        for i in range(len(close)):
            if i < 4:
                buy_count = 0
                sell_count = 0
            else:
                # 买入Setup：当前收盘 < 4根前收盘
                if close.iloc[i] < close.iloc[i-4]:
                    buy_count = (buy_count + 1) if buy_count > 0 else 1
                    sell_count = 0
                # 卖出Setup：当前收盘 > 4根前收盘
                elif close.iloc[i] > close.iloc[i-4]:
                    sell_count = (sell_count + 1) if sell_count > 0 else 1
                    buy_count = 0
                else:
                    # 收盘价相等，计数中断
                    buy_count = 0
                    sell_count = 0
            
            setup_buy.append(buy_count)
            setup_sell.append(sell_count)
        
        return pd.Series(setup_buy, index=close.index), pd.Series(setup_sell, index=close.index)
    
    def detect_perfection(self, close: pd.Series, high: pd.Series, low: pd.Series,
                         td9_buy_signal: pd.Series, td9_sell_signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        检测TD9 Perfection信号
        
        买入Perfection：TD9时的低点 <= 第6或第7根K线的低点
        卖出Perfection：TD9时的高点 >= 第6或第7根K线的高点
        
        Args:
            close: 收盘价
            high: 最高价
            low: 最低价
            td9_buy_signal: TD9买入信号
            td9_sell_signal: TD9卖出信号
            
        Returns:
            (perfection_buy, perfection_sell): Perfection信号
        """
        perfection_buy = np.zeros(len(close), dtype=int)
        perfection_sell = np.zeros(len(close), dtype=int)
        
        # 记录Setup序列的低点和高点
        setup_lows = []
        setup_highs = []
        
        for i in range(len(close)):
            # 买入Perfection检测
            if td9_buy_signal.iloc[i] == 1:
                # 检查第8和第9根（索引6和7在Setup序列中）的低点
                if len(setup_lows) >= 9:
                    low_8 = setup_lows[-3] if len(setup_lows) >= 3 else low.iloc[i]
                    low_7 = setup_lows[-4] if len(setup_lows) >= 4 else low.iloc[i]
                    
                    if low.iloc[i] <= min(low_8, low_7):
                        perfection_buy[i] = 1
                
                setup_lows = []
            
            # 卖出Perfection检测
            if td9_sell_signal.iloc[i] == 1:
                if len(setup_highs) >= 9:
                    high_8 = setup_highs[-3] if len(setup_highs) >= 3 else high.iloc[i]
                    high_7 = setup_highs[-4] if len(setup_highs) >= 4 else high.iloc[i]
                    
                    if high.iloc[i] >= max(high_8, high_7):
                        perfection_sell[i] = 1
                
                setup_highs = []
            
            # 记录当前K线的高低点
            setup_lows.append(low.iloc[i])
            setup_highs.append(high.iloc[i])
            
            # 限制列表长度
            if len(setup_lows) > 10:
                setup_lows.pop(0)
            if len(setup_highs) > 10:
                setup_highs.pop(0)
        
        return pd.Series(perfection_buy, index=close.index), pd.Series(perfection_sell, index=close.index)
    
    def calculate_countdown(self, close: pd.Series, high: pd.Series, low: pd.Series,
                           setup_buy: pd.Series, setup_sell: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算TD Countdown
        
        Countdown在Setup完成（达到9）后开始
        买入Countdown：收盘 <= 2根前的低点
        卖出Countdown：收盘 >= 2根前的高点
        
        Args:
            close: 收盘价
            high: 最高价
            low: 最低价
            setup_buy: 买入Setup计数
            setup_sell: 卖出Setup计数
            
        Returns:
            (countdown_buy, countdown_sell): 买入和卖出Countdown计数
        """
        countdown_buy = []
        countdown_sell = []
        buy_cd = 0
        sell_cd = 0
        setup_completed_buy = False
        setup_completed_sell = False
        
        for i in range(len(close)):
            # 检查Setup是否完成（达到9）
            if setup_buy.iloc[i] >= 9:
                setup_completed_buy = True
            if setup_sell.iloc[i] >= 9:
                setup_completed_sell = True
            
            # 买入Countdown
            if setup_completed_buy and i >= 2:
                if close.iloc[i] <= low.iloc[i-2]:
                    buy_cd += 1
                    if buy_cd >= 13:  # Countdown完成，重置
                        setup_completed_buy = False
                        buy_cd = 0
            else:
                if not setup_completed_buy:
                    buy_cd = 0
            
            # 卖出Countdown
            if setup_completed_sell and i >= 2:
                if close.iloc[i] >= high.iloc[i-2]:
                    sell_cd += 1
                    if sell_cd >= 13:
                        setup_completed_sell = False
                        sell_cd = 0
            else:
                if not setup_completed_sell:
                    sell_cd = 0
            
            countdown_buy.append(buy_cd)
            countdown_sell.append(sell_cd)
        
        return pd.Series(countdown_buy, index=close.index), pd.Series(countdown_sell, index=close.index)
    
    def calculate_combo(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算TD Combo
        
        Combo是另一套独立的计数系统
        买入Combo：收盘 < 2根前收盘
        卖出Combo：收盘 > 2根前收盘
        
        Args:
            close: 收盘价
            
        Returns:
            (combo_buy, combo_sell): 买入和卖出Combo计数
        """
        combo_buy = []
        combo_sell = []
        buy_count = 0
        sell_count = 0
        
        for i in range(len(close)):
            if i < 2:
                buy_count = 0
                sell_count = 0
            else:
                # 买入Combo：收盘 < 2根前收盘
                if close.iloc[i] < close.iloc[i-2]:
                    buy_count = (buy_count + 1) if buy_count > 0 else 1
                    sell_count = 0
                # 卖出Combo：收盘 > 2根前收盘
                elif close.iloc[i] > close.iloc[i-2]:
                    sell_count = (sell_count + 1) if sell_count > 0 else 1
                    buy_count = 0
                else:
                    buy_count = 0
                    sell_count = 0
            
            combo_buy.append(buy_count)
            combo_sell.append(sell_count)
        
        return pd.Series(combo_buy, index=close.index), pd.Series(combo_sell, index=close.index)
    
    def calculate_tdst(self, close: pd.Series, high: pd.Series, low: pd.Series,
                      td9_buy_signal: pd.Series, td9_sell_signal: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算TDST支撑和阻力线
        
        TDST是TD9完成后的支撑/阻力价位
        买入Setup完成后：支撑线 = Setup序列中的最低点
        卖出Setup完成后：阻力线 = Setup序列中的最高点
        
        Args:
            close: 收盘价
            high: 最高价
            low: 最低价
            td9_buy_signal: TD9买入信号
            td9_sell_signal: TD9卖出信号
            
        Returns:
            (tdst_support, tdst_resistance): TDST支撑和阻力
        """
        tdst_support = np.full(len(close), np.nan)
        tdst_resistance = np.full(len(close), np.nan)
        
        current_support = np.nan
        current_resistance = np.nan
        
        setup_low_sequence = []
        setup_high_sequence = []
        
        for i in range(len(close)):
            # 更新当前支撑/阻力
            if not np.isnan(current_support):
                tdst_support[i] = current_support
            if not np.isnan(current_resistance):
                tdst_resistance[i] = current_resistance
            
            # 检测TD9买入信号，设置支撑线
            if td9_buy_signal.iloc[i] == 1:
                if len(setup_low_sequence) >= 9:
                    current_support = min(setup_low_sequence[-9:])
                else:
                    current_support = low.iloc[max(0, i-8):i+1].min()
                
                setup_low_sequence = []
            
            # 检测TD9卖出信号，设置阻力线
            if td9_sell_signal.iloc[i] == 1:
                if len(setup_high_sequence) >= 9:
                    current_resistance = max(setup_high_sequence[-9:])
                else:
                    current_resistance = high.iloc[max(0, i-8):i+1].max()
                
                setup_high_sequence = []
            
            # 记录当前K线的高低点
            setup_low_sequence.append(low.iloc[i])
            setup_high_sequence.append(high.iloc[i])
            
            # 限制序列长度
            if len(setup_low_sequence) > 10:
                setup_low_sequence.pop(0)
            if len(setup_high_sequence) > 10:
                setup_high_sequence.pop(0)
        
        # 填充NaN（使用前向填充）
        tdst_support = pd.Series(tdst_support, index=close.index).fillna(method='ffill')
        tdst_resistance = pd.Series(tdst_resistance, index=close.index).fillna(method='ffill')
        
        return tdst_support, tdst_resistance
    
    def calculate_td_rei(self, close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        计算TD Range Expansion Index (REI)
        
        REI衡量价格区间的扩展程度
        
        Args:
            close: 收盘价
            high: 最高价
            low: 最低价
            
        Returns:
            rei: TD REI值
        """
        # 计算真实波幅
        tr = pd.Series(index=close.index, dtype=float)
        for i in range(len(close)):
            if i == 0:
                tr.iloc[i] = high.iloc[i] - low.iloc[i]
            else:
                tr.iloc[i] = max(
                    high.iloc[i] - low.iloc[i],
                    abs(high.iloc[i] - close.iloc[i-1]),
                    abs(low.iloc[i] - close.iloc[i-1])
                )
        
        # REI = 当前TR / 5期TR平均
        tr_ma = tr.rolling(5).mean()
        rei = tr / (tr_ma + 1e-9)
        
        return rei
    
    def _calculate_signal_strength(self, td9_signal: pd.Series, perfection: pd.Series,
                                  td13_signal: pd.Series, combo13_signal: pd.Series) -> pd.Series:
        """
        计算TD信号强度
        
        权重：
        - TD9: 1.0
        - TD9 Perfection: 2.0
        - TD13: 3.0
        - Combo 13: 2.5
        
        Args:
            td9_signal: TD9信号
            perfection: Perfection信号
            td13_signal: TD13信号
            combo13_signal: Combo 13信号
            
        Returns:
            strength: 信号强度（0-8.5）
        """
        strength = (
            td9_signal * 1.0 +
            perfection * 2.0 +
            td13_signal * 3.0 +
            combo13_signal * 2.5
        )
        
        return strength
    
    def _calculate_bars_since_signal(self, signal: pd.Series, max_bars: int = 999) -> pd.Series:
        """
        计算距离上次信号的K线数
        
        Args:
            signal: 信号序列（0/1）
            max_bars: 最大K线数（超过则截断）
            
        Returns:
            bars_since: 距离上次信号的K线数
        """
        bars_since = np.full(len(signal), max_bars)
        last_signal_idx = -max_bars
        
        for i in range(len(signal)):
            if signal.iloc[i] == 1:
                last_signal_idx = i
            
            bars_since[i] = min(i - last_signal_idx, max_bars)
        
        return pd.Series(bars_since, index=signal.index)


def main():
    """测试TD Sequential实现"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    # 生成测试数据
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='1D')
    np.random.seed(42)
    
    close_prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    ohlcv = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        'low': close_prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        'close': close_prices,
        'volume': np.abs(np.random.randn(len(dates))) * 1000000
    }, index=dates)
    
    # 测试TD Sequential
    td = TDSequential()
    features = td.calculate_all(ohlcv)
    
    print("TD Sequential Features Generated:")
    print(f"Total features: {features.shape[1]}")
    print(f"\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")
    
    # 显示一些信号
    td9_buy_count = features['td_9_buy_signal'].sum()
    td9_sell_count = features['td_9_sell_signal'].sum()
    td13_buy_count = features['td_13_buy_signal'].sum()
    td13_sell_count = features['td_13_sell_signal'].sum()
    
    print(f"\nSignal Statistics:")
    print(f"  TD9 Buy signals: {td9_buy_count}")
    print(f"  TD9 Sell signals: {td9_sell_count}")
    print(f"  TD13 Buy signals: {td13_buy_count}")
    print(f"  TD13 Sell signals: {td13_sell_count}")
    
    print(f"\nSample data (last 10 rows):")
    print(features[['td_setup_buy', 'td_setup_sell', 'td_9_buy_signal', 
                   'td_9_sell_signal', 'td_buy_strength']].tail(10))


if __name__ == '__main__':
    main()

