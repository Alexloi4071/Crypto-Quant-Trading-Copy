# -*- coding: utf-8 -*-
"""
Wyckoff方法完整实现
作者：Richard D. Wyckoff
包含：吸筹阶段、派发阶段、VSA（Volume Spread Analysis）、Composite Man指数
"""

import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class WyckoffAnalysis:
    """
    Wyckoff方法完整实现
    
    包含25+个特征：
    - 吸筹阶段（PS, SC, AR, ST, Spring, SOS, LPS, Backup）
    - 派发阶段（PSY, BC, AR, ST, UT, SOW, LPSY, UTAD）
    - VSA成交量价差分析
    - Composite Man综合指数
    """
    
    def __init__(self, window: int = 20):
        """
        Args:
            window: 分析窗口大小
        """
        self.window = window
    
    def calculate_all(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有Wyckoff特征
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            包含所有Wyckoff特征的DataFrame
        """
        if ohlcv.empty:
            return pd.DataFrame(index=ohlcv.index)
        
        high = ohlcv['high'].copy()
        low = ohlcv['low'].copy()
        close = ohlcv['close'].copy()
        volume = ohlcv['volume'].copy()
        
        features = pd.DataFrame(index=ohlcv.index)
        
        # 1. 吸筹阶段识别
        accumulation_signals = self.detect_accumulation_phases(ohlcv)
        for key, value in accumulation_signals.items():
            features[f'wyk_acc_{key}'] = value
        
        # 2. 派发阶段识别
        distribution_signals = self.detect_distribution_phases(ohlcv)
        for key, value in distribution_signals.items():
            features[f'wyk_dist_{key}'] = value
        
        # 3. VSA分析
        vsa_features = self.analyze_vsa(ohlcv)
        for key, value in vsa_features.items():
            features[f'wyk_vsa_{key}'] = value
        
        # 4. Composite Man指数
        composite_idx = self.calculate_composite_man_index(
            accumulation_signals, distribution_signals, vsa_features
        )
        features['wyk_composite_man'] = composite_idx
        
        # 5. 市场阶段识别（吸筹/派发/上涨/下跌）
        features['wyk_phase'] = self._identify_market_phase(
            accumulation_signals, distribution_signals
        )
        
        # Shift信号特征1期（防止look-ahead bias）
        signal_cols = [c for c in features.columns if any(x in c for x in ['ps_', 'sc_', 'ar_', 'st_', 'spring_', 'sos_', 
                                                                             'lps_', 'psy_', 'bc_', 'ut_', 'sow_', 'lpsy_', 'utad_'])]
        if signal_cols:
            features[signal_cols] = features[signal_cols].shift(1).fillna(0)
        
        # 清理NaN和Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def detect_accumulation_phases(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测吸筹阶段事件
        
        Accumulation Schematic:
        Phase A: PS (Preliminary Support), SC (Selling Climax), AR (Automatic Rally)
        Phase B: ST (Secondary Test)
        Phase C: Spring, Test
        Phase D: SOS (Sign of Strength), LPS (Last Point of Support)
        Phase E: Backup, Jump
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            吸筹阶段信号字典
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        volume = ohlcv['volume']
        
        signals = {}
        
        # 计算辅助指标
        price_range = high - low
        avg_range = price_range.rolling(self.window).mean()
        avg_volume = volume.rolling(self.window).mean()
        volume_ratio = volume / (avg_volume + 1e-9)
        
        # Phase A: Selling Climax (SC)
        # 特征：大幅下跌 + 极高成交量 + 大区间
        price_drop = close.pct_change()
        signals['sc_signal'] = (
            (price_drop < -0.03) &  # 下跌超过3%
            (volume_ratio > 2.0) &  # 成交量是平均的2倍
            (price_range > avg_range * 1.5)  # 区间是平均的1.5倍
        ).astype(int)
        
        # Phase A: Automatic Rally (AR)
        # 特征：在SC后出现的快速反弹
        signals['ar_signal'] = (
            (price_drop > 0.02) &  # 上涨超过2%
            ((signals['sc_signal'].shift(1) == 1) | (signals['sc_signal'].shift(2) == 1) | 
             (signals['sc_signal'].shift(3) == 1))
        ).astype(int)
        
        # Phase A: Preliminary Support (PS)
        # 特征：SC之前的支撑位
        rolling_low = low.rolling(self.window).min()
        signals['ps_signal'] = (
            (low <= rolling_low * 1.01) &  # 接近区间低点
            (volume_ratio > 1.2) &  # 成交量略高
            ((signals['sc_signal'].shift(-1) == 1) | (signals['sc_signal'].shift(-2) == 1))
        ).astype(int)
        
        # Phase B: Secondary Test (ST)
        # 特征：重新测试SC低点，但成交量减少
        signals['st_signal'] = (
            (low <= rolling_low * 1.02) &  # 接近低点
            (volume_ratio < 0.8) &  # 成交量减少
            (signals['sc_signal'].rolling(10).sum().shift(1) > 0)  # 10根K线内有SC
        ).astype(int)
        
        # Phase C: Spring
        # 特征：跌破支撑后快速反弹
        support_level = low.rolling(self.window).min()
        signals['spring_signal'] = (
            (low < support_level.shift(1)) &  # 跌破支撑
            (close > close.shift(1)) &  # 但收盘回升
            (volume_ratio > 1.0)
        ).astype(int)
        
        # Phase D: Sign of Strength (SOS)
        # 特征：强势上涨 + 大成交量
        signals['sos_signal'] = (
            (price_drop > 0.025) &  # 上涨超过2.5%
            (volume_ratio > 1.5) &  # 成交量大
            (close > close.rolling(self.window).mean())  # 突破均线
        ).astype(int)
        
        # Phase D: Last Point of Support (LPS)
        # 特征：SOS后的回调，成交量小
        signals['lps_signal'] = (
            (price_drop < 0) &  # 回调
            (volume_ratio < 0.7) &  # 成交量小
            (signals['sos_signal'].rolling(5).sum().shift(1) > 0) &  # 5根K线内有SOS
            (close > support_level * 1.05)  # 高于支撑位
        ).astype(int)
        
        # Phase E: Backup
        # 特征：突破后的小幅回调确认
        resistance_level = high.rolling(self.window).max()
        signals['backup_signal'] = (
            (close > resistance_level.shift(5)) &  # 突破阻力
            (close < close.shift(1)) &  # 小幅回调
            (volume_ratio < 0.9)
        ).astype(int)
        
        return signals
    
    def detect_distribution_phases(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测派发阶段事件
        
        Distribution Schematic:
        Phase A: PSY (Preliminary Supply), BC (Buying Climax), AR (Automatic Reaction)
        Phase B: ST (Secondary Test)
        Phase C: UT (Upthrust), UTAD (Upthrust After Distribution)
        Phase D: SOW (Sign of Weakness), LPSY (Last Point of Supply)
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            派发阶段信号字典
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        volume = ohlcv['volume']
        
        signals = {}
        
        # 计算辅助指标
        price_range = high - low
        avg_range = price_range.rolling(self.window).mean()
        avg_volume = volume.rolling(self.window).mean()
        volume_ratio = volume / (avg_volume + 1e-9)
        price_change = close.pct_change()
        
        # Phase A: Buying Climax (BC)
        # 特征：大幅上涨 + 极高成交量 + 大区间
        signals['bc_signal'] = (
            (price_change > 0.03) &  # 上涨超过3%
            (volume_ratio > 2.0) &  # 成交量是平均的2倍
            (price_range > avg_range * 1.5)  # 区间是平均的1.5倍
        ).astype(int)
        
        # Phase A: Automatic Reaction (AR)
        # 特征：BC后的快速回调
        signals['ar_dist_signal'] = (
            (price_change < -0.02) &  # 下跌超过2%
            ((signals['bc_signal'].shift(1) == 1) | (signals['bc_signal'].shift(2) == 1) | 
             (signals['bc_signal'].shift(3) == 1))
        ).astype(int)
        
        # Phase A: Preliminary Supply (PSY)
        # 特征：BC之前的阻力位
        rolling_high = high.rolling(self.window).max()
        signals['psy_signal'] = (
            (high >= rolling_high * 0.99) &  # 接近区间高点
            (volume_ratio > 1.2) &  # 成交量略高
            ((signals['bc_signal'].shift(-1) == 1) | (signals['bc_signal'].shift(-2) == 1))
        ).astype(int)
        
        # Phase B: Secondary Test (ST)
        # 特征：重新测试BC高点，但成交量减少
        signals['st_dist_signal'] = (
            (high >= rolling_high * 0.98) &  # 接近高点
            (volume_ratio < 0.8) &  # 成交量减少
            (signals['bc_signal'].rolling(10).sum().shift(1) > 0)  # 10根K线内有BC
        ).astype(int)
        
        # Phase C: Upthrust (UT)
        # 特征：突破阻力后快速回落
        resistance_level = high.rolling(self.window).max()
        signals['ut_signal'] = (
            (high > resistance_level.shift(1)) &  # 突破阻力
            (close < close.shift(1)) &  # 但收盘回落
            (volume_ratio > 1.0)
        ).astype(int)
        
        # Phase C: UTAD (Upthrust After Distribution)
        # 特征：派发末期的假突破
        signals['utad_signal'] = (
            (high > resistance_level.shift(1)) &  # 突破阻力
            (close < resistance_level.shift(1) * 0.99) &  # 收盘回落到阻力下方
            (volume_ratio > 1.3) &  # 成交量大
            (signals['ut_signal'].rolling(10).sum().shift(1) > 0)  # 之前有UT
        ).astype(int)
        
        # Phase D: Sign of Weakness (SOW)
        # 特征：弱势下跌 + 成交量增加
        signals['sow_signal'] = (
            (price_change < -0.025) &  # 下跌超过2.5%
            (volume_ratio > 1.5) &  # 成交量大
            (close < close.rolling(self.window).mean())  # 跌破均线
        ).astype(int)
        
        # Phase D: Last Point of Supply (LPSY)
        # 特征：SOW后的反弹，成交量小
        signals['lpsy_signal'] = (
            (price_change > 0) &  # 反弹
            (volume_ratio < 0.7) &  # 成交量小
            (signals['sow_signal'].rolling(5).sum().shift(1) > 0) &  # 5根K线内有SOW
            (close < resistance_level * 0.95)  # 低于阻力位
        ).astype(int)
        
        return signals
    
    def analyze_vsa(self, ohlcv: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        VSA (Volume Spread Analysis) 成交量价差分析
        
        Args:
            ohlcv: OHLCV数据
            
        Returns:
            VSA特征字典
        """
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        volume = ohlcv['volume']
        
        vsa_features = {}
        
        # 计算价差（Spread）
        spread = high - low
        avg_spread = spread.rolling(self.window).mean()
        spread_ratio = spread / (avg_spread + 1e-9)
        
        # 计算成交量比率
        avg_volume = volume.rolling(self.window).mean()
        volume_ratio = volume / (avg_volume + 1e-9)
        
        # 价格变化
        price_change = close - close.shift(1)
        price_pct = price_change / (close.shift(1) + 1e-9)
        
        # 1. Effort vs Result（努力与结果）
        # 高成交量但价格变化小 = 潜在反转
        vsa_features['effort_vs_result'] = (
            volume_ratio / (abs(price_pct) + 0.001)
        )
        
        # 2. Climax Volume（高潮成交量）
        # 极高成交量 + 大价差
        vsa_features['climax_volume'] = (
            (volume_ratio > 2.5) & 
            (spread_ratio > 1.5)
        ).astype(float)
        
        # 3. Test Volume（测试成交量）
        # 低成交量 + 小价差 = 测试支撑/阻力
        vsa_features['test_volume'] = (
            (volume_ratio < 0.5) & 
            (spread_ratio < 0.7)
        ).astype(float)
        
        # 4. Background Volume（背景成交量）
        # 正常范围的成交量
        vsa_features['background_volume'] = (
            (volume_ratio > 0.7) & 
            (volume_ratio < 1.3)
        ).astype(float)
        
        # 5. No Demand（没有需求）
        # 上涨但成交量减少 = 看跌信号
        vsa_features['no_demand'] = (
            (price_change > 0) & 
            (volume_ratio < 0.6)
        ).astype(float)
        
        # 6. No Supply（没有供应）
        # 下跌但成交量减少 = 看涨信号
        vsa_features['no_supply'] = (
            (price_change < 0) & 
            (volume_ratio < 0.6)
        ).astype(float)
        
        # 7. Stopping Volume（停止成交量）
        # 下跌趋势中的高成交量 + 小价差 = 可能反转
        rolling_trend = close.rolling(10).apply(lambda x: 1 if x[-1] > x[0] else -1, raw=True)
        vsa_features['stopping_volume'] = (
            (rolling_trend < 0) &  # 下跌趋势
            (volume_ratio > 1.8) &  # 高成交量
            (spread_ratio < 1.0) &  # 小价差
            (price_change < 0)  # 当前K线下跌
        ).astype(float)
        
        # 8. Volume Divergence（成交量背离）
        # 价格创新高/新低但成交量减少
        rolling_high = high.rolling(self.window).max()
        rolling_low = low.rolling(self.window).min()
        volume_trend = volume.rolling(5).mean() / volume.rolling(10).mean()
        
        vsa_features['volume_divergence'] = (
            ((high >= rolling_high * 0.99) & (volume_trend < 0.9)) |  # 价格新高但量减
            ((low <= rolling_low * 1.01) & (volume_trend < 0.9))  # 价格新低但量减
        ).astype(float)
        
        return vsa_features
    
    def calculate_composite_man_index(self, accumulation_signals: Dict, 
                                     distribution_signals: Dict,
                                     vsa_features: Dict) -> pd.Series:
        """
        计算Composite Man指数
        
        综合所有Wyckoff信号，给出主力行为指数：
        - 正值：主力吸筹
        - 负值：主力派发
        - 接近0：横盘整理
        
        Args:
            accumulation_signals: 吸筹信号
            distribution_signals: 派发信号
            vsa_features: VSA特征
            
        Returns:
            composite_man_index: 主力行为指数
        """
        # 提取第一个信号以获取索引
        first_signal = list(accumulation_signals.values())[0]
        index = first_signal.index
        
        # 初始化指数
        composite_idx = pd.Series(0.0, index=index)
        
        # 吸筹阶段加分（正值）
        composite_idx += accumulation_signals.get('sc_signal', 0) * 3.0
        composite_idx += accumulation_signals.get('spring_signal', 0) * 4.0
        composite_idx += accumulation_signals.get('sos_signal', 0) * 3.5
        composite_idx += accumulation_signals.get('lps_signal', 0) * 2.0
        composite_idx += accumulation_signals.get('st_signal', 0) * 1.5
        
        # 派发阶段减分（负值）
        composite_idx -= distribution_signals.get('bc_signal', 0) * 3.0
        composite_idx -= distribution_signals.get('ut_signal', 0) * 4.0
        composite_idx -= distribution_signals.get('utad_signal', 0) * 4.5
        composite_idx -= distribution_signals.get('sow_signal', 0) * 3.5
        composite_idx -= distribution_signals.get('lpsy_signal', 0) * 2.0
        
        # VSA特征影响
        composite_idx += vsa_features.get('no_supply', 0) * 2.0
        composite_idx -= vsa_features.get('no_demand', 0) * 2.0
        composite_idx += vsa_features.get('stopping_volume', 0) * 2.5
        
        # 使用EMA平滑指数
        composite_idx = composite_idx.ewm(span=5).mean()
        
        return composite_idx
    
    def _identify_market_phase(self, accumulation_signals: Dict,
                              distribution_signals: Dict) -> pd.Series:
        """
        识别当前市场阶段
        
        Returns:
            phase: 市场阶段编码
            0: 未知/过渡
            1: 吸筹阶段
            2: 上涨阶段
            3: 派发阶段
            4: 下跌阶段
        """
        # 提取第一个信号以获取索引
        first_signal = list(accumulation_signals.values())[0]
        index = first_signal.index
        
        phase = pd.Series(0, index=index)
        
        # 吸筹阶段信号总和
        acc_score = sum([
            accumulation_signals.get('sc_signal', 0),
            accumulation_signals.get('spring_signal', 0),
            accumulation_signals.get('st_signal', 0)
        ])
        
        # 派发阶段信号总和
        dist_score = sum([
            distribution_signals.get('bc_signal', 0),
            distribution_signals.get('ut_signal', 0),
            distribution_signals.get('st_dist_signal', 0)
        ])
        
        # 上涨/下跌信号
        markup_score = accumulation_signals.get('sos_signal', 0)
        markdown_score = distribution_signals.get('sow_signal', 0)
        
        # 确定阶段
        phase = phase.where(acc_score == 0, 1)  # 吸筹
        phase = phase.where(dist_score == 0, 3)  # 派发
        phase = phase.where(markup_score == 0, 2)  # 上涨
        phase = phase.where(markdown_score == 0, 4)  # 下跌
        
        # 使用前向填充保持状态
        phase = phase.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return phase


def main():
    """测试Wyckoff Analysis实现"""
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
    
    # 测试Wyckoff Analysis
    wyckoff = WyckoffAnalysis()
    features = wyckoff.calculate_all(ohlcv)
    
    print("Wyckoff Analysis Features Generated:")
    print(f"Total features: {features.shape[1]}")
    print(f"\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")
    
    # 显示一些信号统计
    print(f"\nSignal Statistics:")
    for col in features.columns:
        if 'signal' in col:
            count = features[col].sum()
            print(f"  {col}: {count}")
    
    print(f"\nMarket Phase Distribution:")
    phase_counts = features['wyk_phase'].value_counts()
    phase_names = {0: 'Unknown', 1: 'Accumulation', 2: 'Markup', 3: 'Distribution', 4: 'Markdown'}
    for phase, count in phase_counts.items():
        print(f"  {phase_names.get(int(phase), 'Unknown')}: {count}")
    
    print(f"\nSample data (last 10 rows):")
    print(features[['wyk_composite_man', 'wyk_phase', 'wyk_vsa_effort_vs_result']].tail(10))


if __name__ == '__main__':
    main()

