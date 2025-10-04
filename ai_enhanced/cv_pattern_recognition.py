"""
Computer Vision Pattern Recognition
基于计算机视觉的图表模式识别系统
自动识别K线图中的技术分析模式，包括形态、趋势线、支撑阻力等
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
from scipy import signal, ndimage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)


class PatternType(Enum):
    """图表模式类型"""
    # 反转形态
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"

    # 持续形态
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG = "flag"
    PENNANT = "pennant"

    # 单根K线形态
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"

    # 趋势线
    SUPPORT_LINE = "support_line"
    RESISTANCE_LINE = "resistance_line"
    TREND_LINE_UP = "trend_line_up"
    TREND_LINE_DOWN = "trend_line_down"

    # 价格通道
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    CHANNEL_HORIZONTAL = "channel_horizontal"


class SignalStrength(Enum):
    """信号强度"""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"

@dataclass
class PatternPoint:
    """模式关键点"""
    x: int  # 时间索引
    y: float  # 价格
    point_type: str  # 点类型：peak, trough, breakout等

    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'point_type': self.point_type
        }

@dataclass
class RecognizedPattern:
    """识别的图表模式"""
    pattern_type: PatternType
    confidence: float  # 0-1之间的置信度
    signal_strength: SignalStrength

    # 几何信息
    start_index: int
    end_index: int
    key_points: List[PatternPoint]

    # 价格信息
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    # 模式特征
    duration_bars: int = 0
    height_ratio: float = 0.0  # 模式高度与价格的比例
    volume_confirmation: bool = False

    # 识别时间
    recognized_at: datetime = field(default_factory=datetime.now)

    # 附加信息
    description: str = ""
    trading_suggestion: str = ""


    def to_dict(self) -> dict:
        return {
            'pattern_type': self.pattern_type.value,
            'confidence': self.confidence,
            'signal_strength': self.signal_strength.value,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'key_points': [p.to_dict() for p in self.key_points],
            'entry_price': self.entry_price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'duration_bars': self.duration_bars,
            'height_ratio': self.height_ratio,
            'volume_confirmation': self.volume_confirmation,
            'recognized_at': self.recognized_at.isoformat(),
            'description': self.description,
            'trading_suggestion': self.trading_suggestion
        }


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        self.image_cache = {}

    def ohlc_to_image(self, ohlc_data: pd.DataFrame,
                     width: int = 800, height: int = 400) -> np.ndarray:
        """将OHLC数据转换为图像"""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

        # 绘制K线图
        self._plot_candlestick(ax, ohlc_data)

        # 转换为图像数组
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return image


    def _plot_candlestick(self, ax, ohlc_data: pd.DataFrame):
        """绘制K线图"""
        for i, (idx, row) in enumerate(ohlc_data.iterrows()):
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # 确定颜色
            color = 'green' if close_price >= open_price else 'red'

            # 绘制影线
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)

            # 绘制实体
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)

            rect = Rectangle((i-0.4, body_bottom), 0.8, body_height,
                           facecolor=color, alpha=0.8)
            ax.add_patch(rect)

        ax.set_xlim(-0.5, len(ohlc_data)-0.5)
        ax.set_ylim(ohlc_data['low'].min() * 0.99, ohlc_data['high'].max() * 1.01)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')


    def extract_price_line(self, image: np.ndarray) -> np.ndarray:
        """从图像中提取价格线"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return edges


    def find_trend_lines(self, image: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """查找趋势线"""
        edges = self.extract_price_line(image)

        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=50, maxLineGap=10)

        trend_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # 过滤水平线和垂直线
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 5:
                    trend_lines.append(((x1, y1), (x2, y2)))

        return trend_lines


class PeakTroughDetector:
    """峰谷检测器"""

    def __init__(self, window_size: int = 5, prominence: float = 0.01):
        self.window_size = window_size
        self.prominence = prominence

    def find_peaks_and_troughs(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """查找峰值和谷值"""
        # 标准化价格以计算相对prominence
        price_std = prices.std()
        min_prominence = price_std * self.prominence

        # 查找峰值
        peaks, _ = signal.find_peaks(prices, prominence=min_prominence,
                                   distance=self.window_size)

        # 查找谷值（反转序列找峰值）
        troughs, _ = signal.find_peaks(-prices, prominence=min_prominence,
                                     distance=self.window_size)

        return peaks, troughs


    def classify_peaks_troughs(self, prices: pd.Series, peaks: np.ndarray,
                              troughs: np.ndarray) -> List[PatternPoint]:
        """分类峰谷点"""
        pattern_points = []

        for peak in peaks:
            point = PatternPoint(
                x=int(peak),
                y=float(prices.iloc[peak]),
                point_type="peak"
            )
            pattern_points.append(point)

        for trough in troughs:
            point = PatternPoint(
                x=int(trough),
                y=float(prices.iloc[trough]),
                point_type="trough"
            )
            pattern_points.append(point)

        # 按时间排序
        pattern_points.sort(key=lambda p: p.x)

        return pattern_points


class CandlestickPatternDetector:
    """蜡烛图形态检测器"""

    def __init__(self):
        self.patterns = {}

    def detect_patterns(self, ohlc_data: pd.DataFrame) -> List[RecognizedPattern]:
        """检测蜡烛图形态"""
        patterns = []

        # 计算蜡烛图特征
        ohlc_data = self._calculate_candle_features(ohlc_data)

        # 检测各种形态
        patterns.extend(self._detect_doji(ohlc_data))
        patterns.extend(self._detect_hammer(ohlc_data))
        patterns.extend(self._detect_shooting_star(ohlc_data))
        patterns.extend(self._detect_engulfing(ohlc_data))

        return patterns


    def _calculate_candle_features(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """计算蜡烛图特征"""
        data = ohlc_data.copy()

        # 实体大小
        data['body'] = abs(data['close'] - data['open'])
        data['body_ratio'] = data['body'] / (data['high'] - data['low'])

        # 上下影线
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']

        # 影线比例
        data['upper_shadow_ratio'] = data['upper_shadow'] / (data['high'] - data['low'])
        data['lower_shadow_ratio'] = data['lower_shadow'] / (data['high'] - data['low'])

        # 蜡烛类型
        data['is_bullish'] = data['close'] > data['open']

        return data


    def _detect_doji(self, data: pd.DataFrame) -> List[RecognizedPattern]:
        """检测十字星形态"""
        patterns = []

        # 十字星条件：实体很小
        doji_mask = data['body_ratio'] < 0.1

        for idx in data.index[doji_mask]:
            if idx < len(data) - 1:  # 确保不是最后一根K线
                confidence = 1.0 - data.loc[idx, 'body_ratio'] * 10

                pattern = RecognizedPattern(
                    pattern_type=PatternType.DOJI,
                    confidence=confidence,
                    signal_strength=SignalStrength.MODERATE,
                    start_index=idx,
                    end_index=idx,
                    key_points=[PatternPoint(idx, data.loc[idx, 'close'], 'doji')],
                    duration_bars=1,
                    description="十字星形态，显示市场犹豫不决",
                    trading_suggestion="等待确认信号后再进行交易"
                )
                patterns.append(pattern)

        return patterns


    def _detect_hammer(self, data: pd.DataFrame) -> List[RecognizedPattern]:
        """检测锤子形态"""
        patterns = []

        # 锤子条件：长下影线，短上影线，小实体
        hammer_mask = (
            (data['lower_shadow_ratio'] > 0.6) &
            (data['upper_shadow_ratio'] < 0.1) &
            (data['body_ratio'] < 0.3)
        )

        for idx in data.index[hammer_mask]:
            confidence = (data.loc[idx, 'lower_shadow_ratio'] +
                         (1 - data.loc[idx, 'body_ratio'])) / 2

            pattern = RecognizedPattern(
                pattern_type=PatternType.HAMMER,
                confidence=confidence,
                signal_strength=SignalStrength.STRONG,
                start_index=idx,
                end_index=idx,
                key_points=[PatternPoint(idx, data.loc[idx, 'low'], 'hammer_low')],
                duration_bars=1,
                description="锤子形态，潜在的看涨反转信号",
                trading_suggestion="在确认突破后考虑买入"
            )
            patterns.append(pattern)

        return patterns


    def _detect_shooting_star(self, data: pd.DataFrame) -> List[RecognizedPattern]:
        """检测流星形态"""
        patterns = []

        # 流星条件：长上影线，短下影线，小实体
        star_mask = (
            (data['upper_shadow_ratio'] > 0.6) &
            (data['lower_shadow_ratio'] < 0.1) &
            (data['body_ratio'] < 0.3)
        )

        for idx in data.index[star_mask]:
            confidence = (data.loc[idx, 'upper_shadow_ratio'] +
                         (1 - data.loc[idx, 'body_ratio'])) / 2

            pattern = RecognizedPattern(
                pattern_type=PatternType.SHOOTING_STAR,
                confidence=confidence,
                signal_strength=SignalStrength.STRONG,
                start_index=idx,
                end_index=idx,
                key_points=[PatternPoint(idx, data.loc[idx, 'high'], 'star_high')],
                duration_bars=1,
                description="流星形态，潜在的看跌反转信号",
                trading_suggestion="在确认突破后考虑卖出"
            )
            patterns.append(pattern)

        return patterns


    def _detect_engulfing(self, data: pd.DataFrame) -> List[RecognizedPattern]:
        """检测吞噬形态"""
        patterns = []

        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]

            # 看涨吞噬：前一根是阴线，当前是阳线，且完全吞噬前一根
            if (not previous['is_bullish'] and current['is_bullish'] and
                current['open'] < previous['close'] and current['close'] > previous['open']):

                confidence = min((current['body'] / previous['body']), 1.0) * 0.8

                pattern = RecognizedPattern(
                    pattern_type=PatternType.ENGULFING_BULLISH,
                    confidence=confidence,
                    signal_strength=SignalStrength.STRONG,
                    start_index=i-1,
                    end_index=i,
                    key_points=[
                        PatternPoint(i-1, previous['close'], 'engulf_start'),
                        PatternPoint(i, current['close'], 'engulf_end')
                    ],
                    duration_bars=2,
                    description="看涨吞噬形态，强烈的看涨信号",
                    trading_suggestion="考虑在突破确认后买入"
                )
                patterns.append(pattern)

            # 看跌吞噬：前一根是阳线，当前是阴线，且完全吞噬前一根
            elif (previous['is_bullish'] and not current['is_bullish'] and
                  current['open'] > previous['close'] and current['close'] < previous['open']):

                confidence = min((current['body'] / previous['body']), 1.0) * 0.8

                pattern = RecognizedPattern(
                    pattern_type=PatternType.ENGULFING_BEARISH,
                    confidence=confidence,
                    signal_strength=SignalStrength.STRONG,
                    start_index=i-1,
                    end_index=i,
                    key_points=[
                        PatternPoint(i-1, previous['close'], 'engulf_start'),
                        PatternPoint(i, current['close'], 'engulf_end')
                    ],
                    duration_bars=2,
                    description="看跌吞噬形态，强烈的看跌信号",
                    trading_suggestion="考虑在突破确认后卖出"
                )
                patterns.append(pattern)

        return patterns


class ChartPatternDetector:
    """图表形态检测器"""

    def __init__(self):
        self.peak_trough_detector = PeakTroughDetector()

    def detect_patterns(self, ohlc_data: pd.DataFrame) -> List[RecognizedPattern]:
        """检测图表形态"""
        patterns = []

        # 使用收盘价进行分析
        close_prices = ohlc_data['close']

        # 查找峰谷
        peaks, troughs = self.peak_trough_detector.find_peaks_and_troughs(close_prices)
        pattern_points = self.peak_trough_detector.classify_peaks_troughs(close_prices, peaks, troughs)

        # 检测各种形态
        patterns.extend(self._detect_double_top_bottom(pattern_points, close_prices))
        patterns.extend(self._detect_head_and_shoulders(pattern_points, close_prices))
        patterns.extend(self._detect_triangles(pattern_points, close_prices))

        return patterns


    def _detect_double_top_bottom(self, pattern_points: List[PatternPoint],
                                 prices: pd.Series) -> List[RecognizedPattern]:
        """检测双顶双底形态"""
        patterns = []

        # 查找连续的峰值对或谷值对
        peaks = [p for p in pattern_points if p.point_type == "peak"]
        troughs = [p for p in pattern_points if p.point_type == "trough"]

        # 检测双顶
        for i in range(len(peaks) - 1):
            peak1, peak2 = peaks[i], peaks[i + 1]

            # 价格相近（差异小于5%）
            price_diff = abs(peak1.y - peak2.y) / peak1.y
            if price_diff < 0.05:
                # 中间需要有一个明显的谷
                between_troughs = [t for t in troughs if peak1.x < t.x < peak2.x]

                if between_troughs:
                    lowest_trough = min(between_troughs, key=lambda t: t.y)

                    # 谷值足够深（至少比峰值低3%）
                    depth_ratio = (peak1.y - lowest_trough.y) / peak1.y
                    if depth_ratio > 0.03:
                        confidence = (1 - price_diff) * min(depth_ratio * 10, 1.0)

                        pattern = RecognizedPattern(
                            pattern_type=PatternType.DOUBLE_TOP,
                            confidence=confidence,
                            signal_strength=SignalStrength.STRONG,
                            start_index=peak1.x,
                            end_index=peak2.x,
                            key_points=[peak1, lowest_trough, peak2],
                            target_price=lowest_trough.y,
                            stop_loss=max(peak1.y, peak2.y) * 1.02,
                            duration_bars=peak2.x - peak1.x,
                            height_ratio=depth_ratio,
                            description="双顶形态，看跌反转信号",
                            trading_suggestion="跌破颈线位时卖出"
                        )
                        patterns.append(pattern)

        # 检测双底
        for i in range(len(troughs) - 1):
            trough1, trough2 = troughs[i], troughs[i + 1]

            # 价格相近（差异小于5%）
            price_diff = abs(trough1.y - trough2.y) / trough1.y
            if price_diff < 0.05:
                # 中间需要有一个明显的峰
                between_peaks = [p for p in peaks if trough1.x < p.x < trough2.x]

                if between_peaks:
                    highest_peak = max(between_peaks, key=lambda p: p.y)

                    # 峰值足够高（至少比谷值高3%）
                    height_ratio = (highest_peak.y - trough1.y) / trough1.y
                    if height_ratio > 0.03:
                        confidence = (1 - price_diff) * min(height_ratio * 10, 1.0)

                        pattern = RecognizedPattern(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            confidence=confidence,
                            signal_strength=SignalStrength.STRONG,
                            start_index=trough1.x,
                            end_index=trough2.x,
                            key_points=[trough1, highest_peak, trough2],
                            target_price=highest_peak.y,
                            stop_loss=min(trough1.y, trough2.y) * 0.98,
                            duration_bars=trough2.x - trough1.x,
                            height_ratio=height_ratio,
                            description="双底形态，看涨反转信号",
                            trading_suggestion="突破颈线位时买入"
                        )
                        patterns.append(pattern)

        return patterns


    def _detect_head_and_shoulders(self, pattern_points: List[PatternPoint],
                                  prices: pd.Series) -> List[RecognizedPattern]:
        """检测头肩形态"""
        patterns = []

        peaks = [p for p in pattern_points if p.point_type == "peak"]
        troughs = [p for p in pattern_points if p.point_type == "trough"]

        # 需要至少3个峰值
        if len(peaks) < 3:
            return patterns

        # 检测头肩顶
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]

            # 头部应该是最高点
            if head.y > left_shoulder.y and head.y > right_shoulder.y:
                # 左右肩高度相近（差异小于10%）
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y) / left_shoulder.y

                if shoulder_diff < 0.1:
                    # 查找颈线（两个肩膀间的低点）
                    neck_troughs = [t for t in troughs
                                   if left_shoulder.x < t.x < right_shoulder.x]

                    if len(neck_troughs) >= 2:
                        neck_line = sum(t.y for t in neck_troughs) / len(neck_troughs)

                        # 头部至少比颈线高5%
                        head_height = (head.y - neck_line) / neck_line
                        if head_height > 0.05:
                            confidence = (1 - shoulder_diff) * min(head_height * 5, 1.0)

                            pattern = RecognizedPattern(
                                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                                confidence=confidence,
                                signal_strength=SignalStrength.VERY_STRONG,
                                start_index=left_shoulder.x,
                                end_index=right_shoulder.x,
                                key_points=[left_shoulder, head, right_shoulder] + neck_troughs,
                                target_price=neck_line - (head.y - neck_line),
                                stop_loss=head.y * 1.02,
                                duration_bars=right_shoulder.x - left_shoulder.x,
                                height_ratio=head_height,
                                description="头肩顶形态，强烈看跌反转信号",
                                trading_suggestion="跌破颈线时卖出"
                            )
                            patterns.append(pattern)

        return patterns


    def _detect_triangles(self, pattern_points: List[PatternPoint],
                         prices: pd.Series) -> List[RecognizedPattern]:
        """检测三角形形态"""
        patterns = []

        if len(pattern_points) < 6:  # 需要至少6个点形成三角形
            return patterns

        # 分离峰值和谷值
        peaks = [p for p in pattern_points if p.point_type == "peak"]
        troughs = [p for p in pattern_points if p.point_type == "trough"]

        if len(peaks) < 3 or len(troughs) < 3:
            return patterns

        # 分析最近的几个峰谷
        recent_peaks = sorted(peaks, key=lambda p: p.x)[-3:]
        recent_troughs = sorted(troughs, key=lambda t: t.x)[-3:]

        # 检测上升三角形
        peak_trend = self._calculate_trend(recent_peaks)
        trough_trend = self._calculate_trend(recent_troughs)

        if abs(peak_trend) < 0.001 and trough_trend > 0.002:  # 阻力线平坦，支撑线上升
            confidence = min(trough_trend * 100, 1.0)

            pattern = RecognizedPattern(
                pattern_type=PatternType.TRIANGLE_ASCENDING,
                confidence=confidence,
                signal_strength=SignalStrength.STRONG,
                start_index=min(recent_peaks[0].x, recent_troughs[0].x),
                end_index=max(recent_peaks[-1].x, recent_troughs[-1].x),
                key_points=recent_peaks + recent_troughs,
                description="上升三角形，看涨持续形态",
                trading_suggestion="向上突破时买入"
            )
            patterns.append(pattern)

        # 检测下降三角形
        elif abs(trough_trend) < 0.001 and peak_trend < -0.002:  # 支撑线平坦，阻力线下降
            confidence = min(abs(peak_trend) * 100, 1.0)

            pattern = RecognizedPattern(
                pattern_type=PatternType.TRIANGLE_DESCENDING,
                confidence=confidence,
                signal_strength=SignalStrength.STRONG,
                start_index=min(recent_peaks[0].x, recent_troughs[0].x),
                end_index=max(recent_peaks[-1].x, recent_troughs[-1].x),
                key_points=recent_peaks + recent_troughs,
                description="下降三角形，看跌持续形态",
                trading_suggestion="向下突破时卖出"
            )
            patterns.append(pattern)

        # 检测对称三角形
        elif peak_trend < -0.001 and trough_trend > 0.001:  # 峰值下降，谷值上升
            convergence = abs(peak_trend) + abs(trough_trend)
            confidence = min(convergence * 50, 1.0)

            pattern = RecognizedPattern(
                pattern_type=PatternType.TRIANGLE_SYMMETRICAL,
                confidence=confidence,
                signal_strength=SignalStrength.MODERATE,
                start_index=min(recent_peaks[0].x, recent_troughs[0].x),
                end_index=max(recent_peaks[-1].x, recent_troughs[-1].x),
                key_points=recent_peaks + recent_troughs,
                description="对称三角形，方向待定的整理形态",
                trading_suggestion="等待突破方向确认"
            )
            patterns.append(pattern)

        return patterns


    def _calculate_trend(self, points: List[PatternPoint]) -> float:
        """计算点列的趋势斜率"""
        if len(points) < 2:
            return 0.0

        x_values = [p.x for p in points]
        y_values = [p.y for p in points]

        # 简单线性回归
        n = len(points)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # 标准化斜率（相对于价格）
        return slope / y_mean


class CVPatternRecognition:
    """计算机视觉图表模式识别主类"""

    def __init__(self):
        self.image_processor = ImageProcessor()
        self.candlestick_detector = CandlestickPatternDetector()
        self.chart_pattern_detector = ChartPatternDetector()

        # 识别历史
        self.recognition_history = []
        self.max_history_size = 1000

        # 统计信息
        self.stats = {
            'total_recognitions': 0,
            'patterns_by_type': {},
            'average_confidence': 0.0,
            'successful_patterns': 0,
            'failed_patterns': 0
        }

        logger.info("计算机视觉图表模式识别器初始化完成")


    def analyze_chart(self, ohlc_data: pd.DataFrame,
                     include_candlestick: bool = True,
                     include_chart_patterns: bool = True,
                     include_image_analysis: bool = False) -> List[RecognizedPattern]:
        """分析图表并识别模式"""

        all_patterns = []

        try:
            # 蜡烛图形态检测
            if include_candlestick:
                candlestick_patterns = self.candlestick_detector.detect_patterns(ohlc_data)
                all_patterns.extend(candlestick_patterns)
                logger.debug(f"检测到 {len(candlestick_patterns)} 个蜡烛图形态")

            # 图表形态检测
            if include_chart_patterns:
                chart_patterns = self.chart_pattern_detector.detect_patterns(ohlc_data)
                all_patterns.extend(chart_patterns)
                logger.debug(f"检测到 {len(chart_patterns)} 个图表形态")

            # 图像分析（可选）
            if include_image_analysis:
                image_patterns = self._analyze_chart_image(ohlc_data)
                all_patterns.extend(image_patterns)
                logger.debug(f"检测到 {len(image_patterns)} 个图像形态")

            # 过滤和排序
            all_patterns = self._filter_and_rank_patterns(all_patterns)

            # 更新统计
            self._update_stats(all_patterns)

            # 记录识别历史
            self._record_recognition(ohlc_data, all_patterns)

            logger.info(f"图表分析完成，识别到 {len(all_patterns)} 个模式")

        except Exception as e:
            logger.error(f"图表分析失败: {e}")

        return all_patterns


    def _analyze_chart_image(self, ohlc_data: pd.DataFrame) -> List[RecognizedPattern]:
        """基于图像的图表分析"""
        patterns = []

        try:
            # 转换为图像
            chart_image = self.image_processor.ohlc_to_image(ohlc_data)

            # 提取趋势线
            trend_lines = self.image_processor.find_trend_lines(chart_image)

            # 转换趋势线为模式
            for i, (start, end) in enumerate(trend_lines):
                # 简单的趋势线分类
                slope = (end[1] - start[1]) / max(end[0] - start[0], 1)

                if slope > 0.1:
                    pattern_type = PatternType.TREND_LINE_UP
                    description = "上升趋势线"
                    suggestion = "趋势线支撑有效时买入"
                elif slope < -0.1:
                    pattern_type = PatternType.TREND_LINE_DOWN
                    description = "下降趋势线"
                    suggestion = "趋势线阻力有效时卖出"
                else:
                    pattern_type = PatternType.SUPPORT_LINE if start[1] > chart_image.shape[0] / 2 else PatternType.RESISTANCE_LINE
                    description = "水平支撑/阻力线"
                    suggestion = "关注突破情况"

                pattern = RecognizedPattern(
                    pattern_type=pattern_type,
                    confidence=0.6,  # 图像识别的置信度相对较低
                    signal_strength=SignalStrength.MODERATE,
                    start_index=int(start[0] * len(ohlc_data) / chart_image.shape[1]),
                    end_index=int(end[0] * len(ohlc_data) / chart_image.shape[1]),
                    key_points=[
                        PatternPoint(int(start[0] * len(ohlc_data) / chart_image.shape[1]),
                                   start[1], 'line_start'),
                        PatternPoint(int(end[0] * len(ohlc_data) / chart_image.shape[1]),
                                   end[1], 'line_end')
                    ],
                    description=description,
                    trading_suggestion=suggestion
                )
                patterns.append(pattern)

        except Exception as e:
            logger.error(f"图像分析失败: {e}")

        return patterns


    def _filter_and_rank_patterns(self, patterns: List[RecognizedPattern]) -> List[RecognizedPattern]:
        """过滤和排序模式"""
        if not patterns:
            return patterns

        # 过滤低置信度的模式
        filtered_patterns = [p for p in patterns if p.confidence > 0.3]

        # 去除重复或重叠的模式
        unique_patterns = self._remove_duplicate_patterns(filtered_patterns)

        # 按置信度和信号强度排序
        strength_order = {
            SignalStrength.VERY_STRONG: 4,
            SignalStrength.STRONG: 3,
            SignalStrength.MODERATE: 2,
            SignalStrength.WEAK: 1
        }

        unique_patterns.sort(
            key=lambda p: (strength_order[p.signal_strength], p.confidence),
            reverse=True
        )

        return unique_patterns


    def _remove_duplicate_patterns(self, patterns: List[RecognizedPattern]) -> List[RecognizedPattern]:
        """移除重复模式"""
        unique_patterns = []

        for pattern in patterns:
            is_duplicate = False

            for existing in unique_patterns:
                # 检查时间重叠和模式类似性
                time_overlap = (
                    max(pattern.start_index, existing.start_index) <
                    min(pattern.end_index, existing.end_index)
                )

                pattern_similar = (
                    pattern.pattern_type == existing.pattern_type or
                    self._are_patterns_similar(pattern.pattern_type, existing.pattern_type)
                )

                if time_overlap and pattern_similar:
                    # 保留置信度更高的模式
                    if pattern.confidence > existing.confidence:
                        unique_patterns.remove(existing)
                        unique_patterns.append(pattern)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_patterns.append(pattern)

        return unique_patterns


    def _are_patterns_similar(self, type1: PatternType, type2: PatternType) -> bool:
        """判断两种模式类型是否相似"""
        similar_groups = [
            {PatternType.DOUBLE_TOP, PatternType.TRIPLE_TOP, PatternType.HEAD_AND_SHOULDERS},
            {PatternType.DOUBLE_BOTTOM, PatternType.TRIPLE_BOTTOM, PatternType.INVERSE_HEAD_AND_SHOULDERS},
            {PatternType.TRIANGLE_ASCENDING, PatternType.TRIANGLE_DESCENDING, PatternType.TRIANGLE_SYMMETRICAL},
            {PatternType.HAMMER, PatternType.HANGING_MAN},
            {PatternType.SUPPORT_LINE, PatternType.RESISTANCE_LINE}
        ]

        for group in similar_groups:
            if type1 in group and type2 in group:
                return True

        return False


    def _update_stats(self, patterns: List[RecognizedPattern]):
        """更新统计信息"""
        self.stats['total_recognitions'] += len(patterns)

        # 按类型统计
        for pattern in patterns:
            pattern_type = pattern.pattern_type.value
            self.stats['patterns_by_type'][pattern_type] = self.stats['patterns_by_type'].get(pattern_type, 0) + 1

        # 更新平均置信度
        if patterns:
            total_confidence = sum(p.confidence for p in patterns)
            current_avg = self.stats['average_confidence']
            current_count = self.stats['total_recognitions'] - len(patterns)

            if current_count > 0:
                self.stats['average_confidence'] = (
                    (current_avg * current_count + total_confidence) /
                    self.stats['total_recognitions']
                )
            else:
                self.stats['average_confidence'] = total_confidence / len(patterns)


    def _record_recognition(self, ohlc_data: pd.DataFrame, patterns: List[RecognizedPattern]):
        """记录识别结果"""
        record = {
            'timestamp': datetime.now(),
            'data_points': len(ohlc_data),
            'patterns_found': len(patterns),
            'pattern_types': [p.pattern_type.value for p in patterns],
            'average_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0.0,
            'high_confidence_count': sum(1 for p in patterns if p.confidence > 0.7)
        }

        self.recognition_history.append(record)

        # 限制历史记录大小
        if len(self.recognition_history) > self.max_history_size:
            self.recognition_history = self.recognition_history[-self.max_history_size:]


    def get_pattern_summary(self, patterns: List[RecognizedPattern]) -> Dict[str, Any]:
        """获取模式摘要"""
        if not patterns:
            return {'message': '未识别到任何模式'}

        # 分类统计
        by_type = defaultdict(int)
        by_strength = defaultdict(int)
        by_signal_direction = defaultdict(int)

        for pattern in patterns:
            by_type[pattern.pattern_type.value] += 1
            by_strength[pattern.signal_strength.value] += 1

            # 简单的信号方向判断
            if pattern.pattern_type.value in ['double_bottom', 'hammer', 'engulfing_bullish',
                                            'inverse_head_and_shoulders', 'triangle_ascending']:
                by_signal_direction['bullish'] += 1
            elif pattern.pattern_type.value in ['double_top', 'shooting_star', 'engulfing_bearish',
                                              'head_and_shoulders', 'triangle_descending']:
                by_signal_direction['bearish'] += 1
            else:
                by_signal_direction['neutral'] += 1

        # 最强信号
        strongest_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)[:3]

        return {
            'total_patterns': len(patterns),
            'average_confidence': np.mean([p.confidence for p in patterns]),
            'patterns_by_type': dict(by_type),
            'patterns_by_strength': dict(by_strength),
            'signal_direction_bias': dict(by_signal_direction),
            'strongest_signals': [
                {
                    'pattern_type': p.pattern_type.value,
                    'confidence': p.confidence,
                    'signal_strength': p.signal_strength.value,
                    'description': p.description,
                    'trading_suggestion': p.trading_suggestion
                }
                for p in strongest_patterns
            ]
        }


    def get_recognition_stats(self) -> Dict[str, Any]:
        """获取识别统计"""
        return {
            'stats': self.stats,
            'history_size': len(self.recognition_history),
            'recent_recognitions': self.recognition_history[-10:] if self.recognition_history else []
        }

# 全局实例
_cv_pattern_recognition_instance = None


def get_cv_pattern_recognition() -> CVPatternRecognition:
    """获取计算机视觉模式识别实例"""
    global _cv_pattern_recognition_instance
    if _cv_pattern_recognition_instance is None:
        _cv_pattern_recognition_instance = CVPatternRecognition()
    return _cv_pattern_recognition_instance
