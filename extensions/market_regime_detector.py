"""
Market Regime Detector
市场状态检测器，基于多种统计和机器学习方法识别市场状态
支持牛市、熊市、震荡市等多种市场状态的实时检测和预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class MarketRegime(Enum):
    """市场状态类型"""
    BULL_MARKET = "bull_market"          # 牛市
    BEAR_MARKET = "bear_market"          # 熊市
    SIDEWAYS = "sideways"                # 震荡市
    HIGH_VOLATILITY = "high_volatility"  # 高波动
    LOW_VOLATILITY = "low_volatility"    # 低波动
    CRISIS = "crisis"                    # 危机
    RECOVERY = "recovery"                # 复苏
    EXPANSION = "expansion"              # 扩张
    CONTRACTION = "contraction"          # 收缩

class DetectionMethod(Enum):
    """检测方法"""
    GAUSSIAN_MIXTURE = "gaussian_mixture"      # 高斯混合模型
    HIDDEN_MARKOV = "hidden_markov"           # 隐马尔可夫模型
    THRESHOLD_MODEL = "threshold_model"       # 阈值模型
    REGIME_SWITCHING = "regime_switching"     # 状态转换模型
    MACHINE_LEARNING = "machine_learning"     # 机器学习
    TECHNICAL_ANALYSIS = "technical_analysis" # 技术分析

@dataclass

class RegimeFeatures:
    """状态特征"""
    # 收益特征
    mean_return: float = 0.0
    volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # 趋势特征
    trend_strength: float = 0.0
    trend_direction: int = 0  # 1: 上涨, -1: 下跌, 0: 无趋势

    # 波动特征
    realized_volatility: float = 0.0
    volatility_of_volatility: float = 0.0

    # 市场广度
    advance_decline_ratio: float = 0.5
    new_highs_lows_ratio: float = 0.5

    # 技术指标
    rsi: float = 50.0
    macd_signal: float = 0.0
    bollinger_position: float = 0.5

    # 宏观指标
    yield_curve_slope: Optional[float] = None
    credit_spread: Optional[float] = None
    vix_level: Optional[float] = None

    def to_array(self) -> np.ndarray:
        """转换为数组格式"""
        return np.array([
            self.mean_return, self.volatility, self.skewness, self.kurtosis,
            self.trend_strength, self.trend_direction,
            self.realized_volatility, self.volatility_of_volatility,
            self.advance_decline_ratio, self.new_highs_lows_ratio,
            self.rsi, self.macd_signal, self.bollinger_position
        ])

    def to_dict(self) -> dict:
        return {
            'mean_return': self.mean_return,
            'volatility': self.volatility,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'trend_strength': self.trend_strength,
            'trend_direction': self.trend_direction,
            'realized_volatility': self.realized_volatility,
            'volatility_of_volatility': self.volatility_of_volatility,
            'advance_decline_ratio': self.advance_decline_ratio,
            'new_highs_lows_ratio': self.new_highs_lows_ratio,
            'rsi': self.rsi,
            'macd_signal': self.macd_signal,
            'bollinger_position': self.bollinger_position,
            'yield_curve_slope': self.yield_curve_slope,
            'credit_spread': self.credit_spread,
            'vix_level': self.vix_level
        }

@dataclass

class RegimeDetection:
    """状态检测结果"""
    timestamp: datetime
    regime: MarketRegime
    confidence: float
    probability_distribution: Dict[MarketRegime, float]

    # 特征值
    features: RegimeFeatures

    # 检测方法信息
    method: DetectionMethod
    model_parameters: Dict[str, Any] = field(default_factory=dict)

    # 持续性信息
    regime_duration: int = 0  # 当前状态持续天数
    last_regime: Optional[MarketRegime] = None
    transition_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'regime': self.regime.value,
            'confidence': self.confidence,
            'probability_distribution': {r.value: p for r, p in self.probability_distribution.items()},
            'features': self.features.to_dict(),
            'method': self.method.value,
            'model_parameters': self.model_parameters,
            'regime_duration': self.regime_duration,
            'last_regime': self.last_regime.value if self.last_regime else None,
            'transition_probability': self.transition_probability
        }

class FeatureExtractor:
    """特征提取器"""

    def __init__(self, lookback_window: int = 60):
        self.lookback_window = lookback_window

    def extract_features(self, data: pd.DataFrame,
                        market_data: Dict[str, pd.Series] = None) -> RegimeFeatures:
        """提取特征"""

        if len(data) < self.lookback_window:
            logger.warning(f"数据长度不足: {len(data)} < {self.lookback_window}")
            return RegimeFeatures()

        # 计算收益率
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.tail(self.lookback_window)

        features = RegimeFeatures()

        # 收益特征
        features.mean_return = recent_returns.mean() * 252  # 年化
        features.volatility = recent_returns.std() * np.sqrt(252)  # 年化
        features.skewness = recent_returns.skew()
        features.kurtosis = recent_returns.kurtosis()

        # 趋势特征
        features.trend_strength, features.trend_direction = self._calculate_trend(data)

        # 波动特征
        features.realized_volatility = self._calculate_realized_volatility(data)
        features.volatility_of_volatility = self._calculate_vol_of_vol(data)

        # 技术指标
        features.rsi = self._calculate_rsi(data['close'])
        features.macd_signal = self._calculate_macd_signal(data['close'])
        features.bollinger_position = self._calculate_bollinger_position(data['close'])

        # 市场广度（如果有相关数据）
        if market_data:
            features.advance_decline_ratio = self._calculate_advance_decline(market_data)
            features.new_highs_lows_ratio = self._calculate_new_highs_lows(market_data)

            # 宏观指标
            if 'vix' in market_data:
                features.vix_level = market_data['vix'].iloc[-1]
            if 'yield_curve' in market_data:
                features.yield_curve_slope = self._calculate_yield_curve_slope(market_data['yield_curve'])

        return features

    def _calculate_trend(self, data: pd.DataFrame) -> Tuple[float, int]:
        """计算趋势强度和方向"""
        prices = data['close'].tail(self.lookback_window)

        # 线性回归斜率作为趋势强度
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)

        # 趋势强度（R方）
        trend_strength = r_value ** 2

        # 趋势方向
        trend_direction = 1 if slope > 0 else -1 if slope < 0 else 0

        return trend_strength, trend_direction

    def _calculate_realized_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """计算已实现波动率"""
        returns = data['close'].pct_change().dropna()
        realized_vol = returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)
        return realized_vol if not np.isnan(realized_vol) else 0.0

    def _calculate_vol_of_vol(self, data: pd.DataFrame, window: int = 20) -> float:
        """计算波动率的波动率"""
        returns = data['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.pct_change().std()
        return vol_of_vol if not np.isnan(vol_of_vol) else 0.0

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()

        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0

    def _calculate_macd_signal(self, prices: pd.Series) -> float:
        """计算MACD信号"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()

        macd_signal = macd.iloc[-1] - signal.iloc[-1]
        return macd_signal if not np.isnan(macd_signal) else 0.0

    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> float:
        """计算价格在布林带中的位置"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]

        if current_upper == current_lower:
            return 0.5

        position = (current_price - current_lower) / (current_upper - current_lower)
        return np.clip(position, 0, 1)

    def _calculate_advance_decline(self, market_data: Dict[str, pd.Series]) -> float:
        """计算涨跌比"""
        if 'advances' in market_data and 'declines' in market_data:
            advances = market_data['advances'].iloc[-1]
            declines = market_data['declines'].iloc[-1]
            total = advances + declines
            return advances / total if total > 0 else 0.5
        return 0.5

    def _calculate_new_highs_lows(self, market_data: Dict[str, pd.Series]) -> float:
        """计算新高新低比"""
        if 'new_highs' in market_data and 'new_lows' in market_data:
            new_highs = market_data['new_highs'].iloc[-1]
            new_lows = market_data['new_lows'].iloc[-1]
            total = new_highs + new_lows
            return new_highs / total if total > 0 else 0.5
        return 0.5

    def _calculate_yield_curve_slope(self, yield_curve: pd.Series) -> float:
        """计算收益率曲线斜率"""
        # 简化计算：10年期 - 2年期
        if len(yield_curve) >= 2:
            return yield_curve.iloc[-1] - yield_curve.iloc[0]
        return 0.0

class GaussianMixtureDetector:
    """高斯混合模型检测器"""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_history = deque(maxlen=1000)
        self.regime_mapping = {}

    def fit(self, features_list: List[RegimeFeatures]):
        """训练模型"""
        if len(features_list) < 50:
            raise ValueError("训练数据不足，至少需要50个样本")

        # 准备训练数据
        X = np.array([f.to_array() for f in features_list])

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练高斯混合模型
        self.model = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='full'
        )

        self.model.fit(X_scaled)

        # 创建状态映射
        self._create_regime_mapping(X_scaled)

        logger.info(f"高斯混合模型训练完成，{self.n_components}个状态")

    def _create_regime_mapping(self, X_scaled: np.ndarray):
        """创建状态映射"""
        # 获取每个样本的状态标签
        labels = self.model.predict(X_scaled)

        # 计算每个状态的特征均值
        for i in range(self.n_components):
            mask = labels == i
            if np.sum(mask) > 0:
                cluster_features = X_scaled[mask]
                mean_features = np.mean(cluster_features, axis=0)

                # 基于特征均值确定市场状态
                regime = self._interpret_cluster(mean_features)
                self.regime_mapping[i] = regime

    def _interpret_cluster(self, mean_features: np.ndarray) -> MarketRegime:
        """解释聚类结果"""
        # 特征索引
        mean_return_idx = 0
        volatility_idx = 1
        trend_strength_idx = 4
        trend_direction_idx = 5

        mean_return = mean_features[mean_return_idx]
        volatility = mean_features[volatility_idx]
        trend_strength = mean_features[trend_strength_idx]
        trend_direction = mean_features[trend_direction_idx]

        # 基于规则的状态判断
        if volatility > 1.0:  # 高波动
            if mean_return > 0:
                return MarketRegime.CRISIS if trend_direction < 0 else MarketRegime.HIGH_VOLATILITY
            else:
                return MarketRegime.BEAR_MARKET
        elif mean_return > 0.1 and trend_direction > 0:
            return MarketRegime.BULL_MARKET
        elif mean_return < -0.1 and trend_direction < 0:
            return MarketRegime.BEAR_MARKET
        elif abs(mean_return) < 0.05 and volatility < 0.2:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.SIDEWAYS

    def detect_regime(self, features: RegimeFeatures) -> RegimeDetection:
        """检测当前市场状态"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 准备数据
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # 预测状态概率
        probabilities = self.model.predict_proba(X_scaled)[0]
        predicted_label = np.argmax(probabilities)

        # 获取对应的市场状态
        regime = self.regime_mapping.get(predicted_label, MarketRegime.SIDEWAYS)
        confidence = probabilities[predicted_label]

        # 构建概率分布
        prob_distribution = {}
        for label, prob in enumerate(probabilities):
            market_regime = self.regime_mapping.get(label, MarketRegime.SIDEWAYS)
            prob_distribution[market_regime] = prob

        # 添加到历史
        self.feature_history.append(features)

        detection = RegimeDetection(
            timestamp=datetime.now(),
            regime=regime,
            confidence=confidence,
            probability_distribution=prob_distribution,
            features=features,
            method=DetectionMethod.GAUSSIAN_MIXTURE,
            model_parameters={
                'n_components': self.n_components,
                'predicted_label': int(predicted_label)
            }
        )

        return detection

class ThresholdDetector:
    """阈值模型检测器"""

    def __init__(self):
        self.thresholds = {
            'bull_market': {'min_return': 0.15, 'max_volatility': 0.25, 'min_trend': 0.3},
            'bear_market': {'max_return': -0.10, 'min_trend_down': -0.3},
            'high_volatility': {'min_volatility': 0.30},
            'low_volatility': {'max_volatility': 0.12},
            'crisis': {'max_return': -0.20, 'min_volatility': 0.40}
        }

    def detect_regime(self, features: RegimeFeatures) -> RegimeDetection:
        """基于阈值检测市场状态"""

        # 检查各种状态条件
        regime_scores = {}

        # 牛市条件
        bull_score = 0
        if features.mean_return >= self.thresholds['bull_market']['min_return']:
            bull_score += 0.4
        if features.volatility <= self.thresholds['bull_market']['max_volatility']:
            bull_score += 0.3
        if features.trend_strength >= self.thresholds['bull_market']['min_trend'] and features.trend_direction > 0:
            bull_score += 0.3
        regime_scores[MarketRegime.BULL_MARKET] = bull_score

        # 熊市条件
        bear_score = 0
        if features.mean_return <= self.thresholds['bear_market']['max_return']:
            bear_score += 0.5
        if features.trend_strength >= abs(self.thresholds['bear_market']['min_trend_down']) and features.trend_direction < 0:
            bear_score += 0.5
        regime_scores[MarketRegime.BEAR_MARKET] = bear_score

        # 危机条件
        crisis_score = 0
        if features.mean_return <= self.thresholds['crisis']['max_return']:
            crisis_score += 0.5
        if features.volatility >= self.thresholds['crisis']['min_volatility']:
            crisis_score += 0.5
        regime_scores[MarketRegime.CRISIS] = crisis_score

        # 高波动条件
        high_vol_score = 0
        if features.volatility >= self.thresholds['high_volatility']['min_volatility']:
            high_vol_score = 0.8
        regime_scores[MarketRegime.HIGH_VOLATILITY] = high_vol_score

        # 低波动条件
        low_vol_score = 0
        if features.volatility <= self.thresholds['low_volatility']['max_volatility']:
            low_vol_score = 0.7
        regime_scores[MarketRegime.LOW_VOLATILITY] = low_vol_score

        # 震荡市（默认）
        regime_scores[MarketRegime.SIDEWAYS] = 0.2

        # 选择最高得分的状态
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]

        # 归一化概率分布
        total_score = sum(regime_scores.values())
        if total_score > 0:
            prob_distribution = {regime: score / total_score for regime, score in regime_scores.items()}
        else:
            prob_distribution = {regime: 1.0 / len(regime_scores) for regime in regime_scores}

        detection = RegimeDetection(
            timestamp=datetime.now(),
            regime=best_regime,
            confidence=confidence,
            probability_distribution=prob_distribution,
            features=features,
            method=DetectionMethod.THRESHOLD_MODEL,
            model_parameters={'thresholds': self.thresholds}
        )

        return detection

class MarketRegimeDetector:
    """市场状态检测器主类"""

    def __init__(self, detection_method: DetectionMethod = DetectionMethod.GAUSSIAN_MIXTURE):
        self.detection_method = detection_method
        self.feature_extractor = FeatureExtractor()

        # 初始化检测器
        if detection_method == DetectionMethod.GAUSSIAN_MIXTURE:
            self.detector = GaussianMixtureDetector()
        elif detection_method == DetectionMethod.THRESHOLD_MODEL:
            self.detector = ThresholdDetector()
        else:
            raise ValueError(f"不支持的检测方法: {detection_method}")

        # 历史记录
        self.detection_history = deque(maxlen=1000)
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_start_time = datetime.now()

        # 统计信息
        self.stats = {
            'total_detections': 0,
            'regime_changes': 0,
            'regime_durations': defaultdict(list),
            'detection_accuracy': 0.0
        }

        logger.info(f"市场状态检测器初始化完成，方法: {detection_method.value}")

    def train(self, historical_data: pd.DataFrame,
              market_data: Dict[str, pd.Series] = None):
        """训练检测器"""
        if self.detection_method == DetectionMethod.THRESHOLD_MODEL:
            logger.info("阈值模型无需训练")
            return

        logger.info("开始训练市场状态检测器...")

        # 提取历史特征
        features_list = []
        window_size = 60

        for i in range(window_size, len(historical_data), 5):  # 每5天取一个样本
            data_window = historical_data.iloc[i-window_size:i]

            # 构建市场数据窗口
            market_window = {}
            if market_data:
                for key, series in market_data.items():
                    if len(series) > i:
                        market_window[key] = series.iloc[i-window_size:i]

            features = self.feature_extractor.extract_features(data_window, market_window)
            features_list.append(features)

        # 训练模型
        if hasattr(self.detector, 'fit'):
            self.detector.fit(features_list)

        logger.info(f"训练完成，提取了 {len(features_list)} 个特征样本")

    def detect_current_regime(self, data: pd.DataFrame,
                            market_data: Dict[str, pd.Series] = None) -> RegimeDetection:
        """检测当前市场状态"""

        # 提取特征
        features = self.feature_extractor.extract_features(data, market_data)

        # 检测状态
        detection = self.detector.detect_regime(features)

        # 更新持续时间和转换信息
        if self.detection_history:
            last_detection = self.detection_history[-1]
            detection.last_regime = last_detection.regime

            if detection.regime == self.current_regime:
                detection.regime_duration = (datetime.now() - self.regime_start_time).days
            else:
                # 状态改变
                self.stats['regime_changes'] += 1

                # 记录上一个状态的持续时间
                duration = (datetime.now() - self.regime_start_time).days
                self.stats['regime_durations'][self.current_regime.value].append(duration)

                self.current_regime = detection.regime
                self.regime_start_time = datetime.now()
                detection.regime_duration = 0

                # 计算转换概率（简化）
                detection.transition_probability = 0.1  # 可以基于历史数据计算
        else:
            self.current_regime = detection.regime
            self.regime_start_time = datetime.now()

        # 添加到历史
        self.detection_history.append(detection)
        self.stats['total_detections'] += 1

        logger.debug(f"检测到市场状态: {detection.regime.value} (置信度: {detection.confidence:.3f})")

        return detection

    def get_regime_statistics(self, days_back: int = 252) -> Dict[str, Any]:
        """获取状态统计信息"""
        cutoff_time = datetime.now() - timedelta(days=days_back)

        # 过滤历史记录
        recent_detections = [
            d for d in self.detection_history
            if d.timestamp >= cutoff_time
        ]

        if not recent_detections:
            return {}

        # 统计各状态的出现频率
        regime_counts = defaultdict(int)
        for detection in recent_detections:
            regime_counts[detection.regime.value] += 1

        total_detections = len(recent_detections)
        regime_frequencies = {
            regime: count / total_detections
            for regime, count in regime_counts.items()
        }

        # 计算平均置信度
        avg_confidence = np.mean([d.confidence for d in recent_detections])

        # 状态转换统计
        transitions = defaultdict(int)
        for i in range(1, len(recent_detections)):
            prev_regime = recent_detections[i-1].regime.value
            curr_regime = recent_detections[i].regime.value

            if prev_regime != curr_regime:
                transitions[f"{prev_regime} -> {curr_regime}"] += 1

        return {
            'detection_period_days': days_back,
            'total_detections': total_detections,
            'current_regime': self.current_regime.value,
            'regime_frequencies': regime_frequencies,
            'average_confidence': avg_confidence,
            'state_transitions': dict(transitions),
            'regime_durations': {
                regime: {
                    'mean': np.mean(durations) if durations else 0,
                    'std': np.std(durations) if durations else 0,
                    'min': min(durations) if durations else 0,
                    'max': max(durations) if durations else 0
                }
                for regime, durations in self.stats['regime_durations'].items()
            }
        }

    def predict_regime_change(self, forecast_days: int = 5) -> Dict[str, Any]:
        """预测状态变化"""
        if len(self.detection_history) < 10:
            return {'prediction': 'insufficient_data'}

        recent_detections = list(self.detection_history)[-20:]  # 最近20个检测结果

        # 分析趋势
        confidence_trend = [d.confidence for d in recent_detections]
        confidence_slope = np.polyfit(range(len(confidence_trend)), confidence_trend, 1)[0]

        # 分析概率分布的变化
        regime_probabilities = defaultdict(list)
        for detection in recent_detections:
            for regime, prob in detection.probability_distribution.items():
                regime_probabilities[regime].append(prob)

        # 计算各状态概率的趋势
        prob_trends = {}
        for regime, probs in regime_probabilities.items():
            if len(probs) > 5:
                trend = np.polyfit(range(len(probs)), probs, 1)[0]
                prob_trends[regime.value] = trend

        # 预测逻辑（简化）
        current_regime = self.current_regime
        current_duration = (datetime.now() - self.regime_start_time).days

        # 基于历史平均持续时间判断
        avg_duration = np.mean(self.stats['regime_durations'][current_regime.value]) if self.stats['regime_durations'][current_regime.value] else 30

        change_probability = min(current_duration / avg_duration, 1.0) * 0.5

        # 如果置信度在下降，增加变化概率
        if confidence_slope < -0.01:
            change_probability += 0.2

        # 预测最可能的下一个状态
        if current_regime == MarketRegime.BULL_MARKET:
            next_regime_candidates = [MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY]
        elif current_regime == MarketRegime.BEAR_MARKET:
            next_regime_candidates = [MarketRegime.RECOVERY, MarketRegime.SIDEWAYS]
        else:
            next_regime_candidates = [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET]

        return {
            'current_regime': current_regime.value,
            'current_duration_days': current_duration,
            'change_probability': change_probability,
            'confidence_trend': confidence_slope,
            'probability_trends': prob_trends,
            'next_regime_candidates': [r.value for r in next_regime_candidates],
            'forecast_horizon_days': forecast_days
        }

    def get_detection_summary(self) -> Dict[str, Any]:
        """获取检测摘要"""
        return {
            'detection_method': self.detection_method.value,
            'current_regime': self.current_regime.value,
            'regime_start_time': self.regime_start_time.isoformat(),
            'current_duration_days': (datetime.now() - self.regime_start_time).days,
            'total_detections': self.stats['total_detections'],
            'total_regime_changes': self.stats['regime_changes'],
            'detection_history_size': len(self.detection_history),
            'recent_detections': [
                d.to_dict() for d in list(self.detection_history)[-5:]
            ] if self.detection_history else []
        }

# 全局实例
_market_regime_detector_instance = None

def get_market_regime_detector() -> MarketRegimeDetector:
    """获取市场状态检测器实例"""
    global _market_regime_detector_instance
    if _market_regime_detector_instance is None:
        _market_regime_detector_instance = MarketRegimeDetector()
    return _market_regime_detector_instance
