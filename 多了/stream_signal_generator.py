"""
Stream Signal Generator
流式信号生成器，基于实时数据流生成交易信号
集成机器学习模型和技术分析，提供智能信号决策
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.model_manager import ModelManager
from src.realtime.metrics_calculator import get_realtime_metrics_calculator, CalculatedMetric
from src.trading.signal_generator import SignalType, SignalStrength
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class SignalSource(Enum):
    """信号源枚举"""
    TECHNICAL = "technical"
    ML_MODEL = "ml_model"
    SENTIMENT = "sentiment"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    HYBRID = "hybrid"

@dataclass
class StreamSignal:
    """流式信号数据结构"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0-1
    source: SignalSource
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 信号细节
    price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    
    # 支撑数据
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 有效期
    expires_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """检查信号是否仍有效"""
        if self.expires_at is None:
            return True
        return datetime.now() < self.expires_at
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'source': self.source.value,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'expected_return': self.expected_return,
            'risk_score': self.risk_score,
            'indicators': self.indicators,
            'metadata': self.metadata,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class TechnicalSignalGenerator:
    """技术分析信号生成器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 技术指标阈值
        self.thresholds = {
            'rsi_oversold': self.config.get('rsi_oversold', 30),
            'rsi_overbought': self.config.get('rsi_overbought', 70),
            'bb_squeeze': self.config.get('bb_squeeze_ratio', 0.02),
            'volume_surge': self.config.get('volume_surge_multiplier', 2.0),
            'price_breakout': self.config.get('breakout_threshold', 0.02)
        }
        
    def generate_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric]) -> List[StreamSignal]:
        """基于技术指标生成信号"""
        signals = []
        
        try:
            # RSI信号
            rsi_signals = self._check_rsi_signals(symbol, metrics)
            signals.extend(rsi_signals)
            
            # 移动平均信号
            ma_signals = self._check_ma_signals(symbol, metrics)
            signals.extend(ma_signals)
            
            # 布林带信号
            bb_signals = self._check_bollinger_signals(symbol, metrics)
            signals.extend(bb_signals)
            
            # 成交量信号
            volume_signals = self._check_volume_signals(symbol, metrics)
            signals.extend(volume_signals)
            
            # MACD信号
            macd_signals = self._check_macd_signals(symbol, metrics)
            signals.extend(macd_signals)
            
        except Exception as e:
            logger.error(f"生成技术信号失败 {symbol}: {e}")
        
        return signals
    
    def _check_rsi_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric]) -> List[StreamSignal]:
        """检查RSI信号"""
        signals = []
        
        rsi_metric = metrics.get('rsi_14')
        if not rsi_metric:
            return signals
        
        rsi_value = rsi_metric.value
        
        if rsi_value <= self.thresholds['rsi_oversold']:
            # 超卖信号
            signal = StreamSignal(
                signal_id=f"rsi_oversold_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=SignalStrength.MEDIUM if rsi_value > 20 else SignalStrength.STRONG,
                confidence=min(1.0, (self.thresholds['rsi_oversold'] - rsi_value) / 20),
                source=SignalSource.TECHNICAL,
                indicators={'rsi': rsi_value},
                metadata={'reason': 'RSI oversold condition'},
                expires_at=datetime.now() + timedelta(hours=1)
            )
            signals.append(signal)
            
        elif rsi_value >= self.thresholds['rsi_overbought']:
            # 超买信号
            signal = StreamSignal(
                signal_id=f"rsi_overbought_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=SignalStrength.MEDIUM if rsi_value < 80 else SignalStrength.STRONG,
                confidence=min(1.0, (rsi_value - self.thresholds['rsi_overbought']) / 20),
                source=SignalSource.TECHNICAL,
                indicators={'rsi': rsi_value},
                metadata={'reason': 'RSI overbought condition'},
                expires_at=datetime.now() + timedelta(hours=1)
            )
            signals.append(signal)
        
        return signals
    
    def _check_ma_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric]) -> List[StreamSignal]:
        """检查移动平均信号"""
        signals = []
        
        sma_5 = metrics.get('sma_5')
        sma_20 = metrics.get('sma_20')
        current_price = metrics.get('price_mean')
        
        if not all([sma_5, sma_20, current_price]):
            return signals
        
        # 金叉/死叉信号
        if sma_5.value > sma_20.value * 1.01:  # 1%阈值避免噪音
            signal = StreamSignal(
                signal_id=f"ma_golden_cross_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=SignalStrength.MEDIUM,
                confidence=min(1.0, (sma_5.value - sma_20.value) / sma_20.value * 10),
                source=SignalSource.TECHNICAL,
                indicators={'sma_5': sma_5.value, 'sma_20': sma_20.value},
                metadata={'reason': 'Moving average golden cross'},
                expires_at=datetime.now() + timedelta(hours=2)
            )
            signals.append(signal)
            
        elif sma_5.value < sma_20.value * 0.99:  # 1%阈值
            signal = StreamSignal(
                signal_id=f"ma_death_cross_{symbol}_{int(datetime.now().timestamp())}",
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=SignalStrength.MEDIUM,
                confidence=min(1.0, (sma_20.value - sma_5.value) / sma_20.value * 10),
                source=SignalSource.TECHNICAL,
                indicators={'sma_5': sma_5.value, 'sma_20': sma_20.value},
                metadata={'reason': 'Moving average death cross'},
                expires_at=datetime.now() + timedelta(hours=2)
            )
            signals.append(signal)
        
        return signals
    
    def _check_bollinger_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric]) -> List[StreamSignal]:
        """检查布林带信号"""
        signals = []
        # 简化实现，实际需要布林带指标
        return signals
    
    def _check_volume_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric]) -> List[StreamSignal]:
        """检查成交量信号"""
        signals = []
        
        volume_mean = metrics.get('volume_mean')
        if not volume_mean:
            return signals
        
        # 这里需要当前成交量数据进行比较
        # 简化实现
        
        return signals
    
    def _check_macd_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric]) -> List[StreamSignal]:
        """检查MACD信号"""
        signals = []
        # 简化实现，需要MACD指标支持
        return signals

class MLSignalGenerator:
    """机器学习信号生成器"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.loaded_models = {}
        
    def generate_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric], 
                        features: Dict[str, float] = None) -> List[StreamSignal]:
        """基于机器学习模型生成信号"""
        signals = []
        
        try:
            # 获取模型
            model = self._get_model(symbol)
            if not model:
                return signals
            
            # 准备特征数据
            feature_vector = self._prepare_features(symbol, metrics, features)
            if not feature_vector:
                return signals
            
            # 模型预测
            prediction = model.predict([feature_vector])[0]
            confidence = None
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba([feature_vector])[0]
                confidence = max(probabilities)
            
            # 转换预测为信号
            signal = self._prediction_to_signal(symbol, prediction, confidence, features)
            if signal:
                signals.append(signal)
                
        except Exception as e:
            logger.error(f"ML信号生成失败 {symbol}: {e}")
        
        return signals
    
    def _get_model(self, symbol: str):
        """获取交易对对应的模型"""
        if symbol not in self.loaded_models:
            try:
                model = self.model_manager.load_latest_model(symbol)
                if model:
                    self.loaded_models[symbol] = model
            except Exception as e:
                logger.debug(f"加载模型失败 {symbol}: {e}")
                return None
        
        return self.loaded_models.get(symbol)
    
    def _prepare_features(self, symbol: str, metrics: Dict[str, CalculatedMetric], 
                         additional_features: Dict[str, float] = None) -> Optional[List[float]]:
        """准备模型特征"""
        features = []
        
        # 从指标中提取特征
        feature_names = ['rsi_14', 'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 
                        'price_mean', 'price_std', 'volume_mean']
        
        for feature_name in feature_names:
            metric = metrics.get(feature_name)
            if metric:
                features.append(metric.value)
            else:
                features.append(0.0)  # 缺失值用0填充
        
        # 添加额外特征
        if additional_features:
            for value in additional_features.values():
                features.append(value)
        
        return features if len(features) > 0 else None
    
    def _prediction_to_signal(self, symbol: str, prediction: Any, confidence: Optional[float],
                             features: Dict[str, float] = None) -> Optional[StreamSignal]:
        """将预测转换为交易信号"""
        
        # 假设模型输出: 0=持有, 1=买入, 2=卖出
        if prediction == 1:
            signal_type = SignalType.BUY
        elif prediction == 2:
            signal_type = SignalType.SELL
        else:
            return None  # 持有信号不生成
        
        # 确定信号强度
        if confidence is not None:
            if confidence >= 0.8:
                strength = SignalStrength.STRONG
            elif confidence >= 0.6:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.MEDIUM
            confidence = 0.5
        
        signal = StreamSignal(
            signal_id=f"ml_signal_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source=SignalSource.ML_MODEL,
            metadata={
                'model_prediction': int(prediction),
                'feature_count': len(features) if features else 0
            },
            expires_at=datetime.now() + timedelta(minutes=30)
        )
        
        return signal

class HybridSignalGenerator:
    """混合信号生成器"""
    
    def __init__(self, technical_generator: TechnicalSignalGenerator, 
                 ml_generator: MLSignalGenerator):
        self.technical_generator = technical_generator
        self.ml_generator = ml_generator
        
        # 信号权重配置
        self.signal_weights = {
            SignalSource.TECHNICAL: 0.4,
            SignalSource.ML_MODEL: 0.6
        }
    
    def generate_hybrid_signals(self, symbol: str, metrics: Dict[str, CalculatedMetric],
                              features: Dict[str, float] = None) -> List[StreamSignal]:
        """生成混合信号"""
        all_signals = []
        
        # 收集各种信号
        technical_signals = self.technical_generator.generate_signals(symbol, metrics)
        ml_signals = self.ml_generator.generate_signals(symbol, metrics, features)
        
        # 按类型分组信号
        buy_signals = []
        sell_signals = []
        
        for signal in technical_signals + ml_signals:
            if signal.signal_type == SignalType.BUY:
                buy_signals.append(signal)
            elif signal.signal_type == SignalType.SELL:
                sell_signals.append(signal)
        
        # 生成综合信号
        if buy_signals:
            hybrid_buy = self._combine_signals(symbol, SignalType.BUY, buy_signals)
            if hybrid_buy:
                all_signals.append(hybrid_buy)
        
        if sell_signals:
            hybrid_sell = self._combine_signals(symbol, SignalType.SELL, sell_signals)
            if hybrid_sell:
                all_signals.append(hybrid_sell)
        
        return all_signals
    
    def _combine_signals(self, symbol: str, signal_type: SignalType, 
                        signals: List[StreamSignal]) -> Optional[StreamSignal]:
        """组合同类型信号"""
        if not signals:
            return None
        
        # 加权平均置信度
        total_confidence = 0
        total_weight = 0
        
        source_contributions = defaultdict(list)
        
        for signal in signals:
            weight = self.signal_weights.get(signal.source, 0.5)
            total_confidence += signal.confidence * weight
            total_weight += weight
            source_contributions[signal.source].append(signal)
        
        if total_weight == 0:
            return None
        
        avg_confidence = total_confidence / total_weight
        
        # 确定信号强度
        if avg_confidence >= 0.8:
            strength = SignalStrength.STRONG
        elif avg_confidence >= 0.6:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK
        
        # 收集支撑数据
        combined_indicators = {}
        combined_metadata = {
            'signal_count': len(signals),
            'sources': list(source_contributions.keys()),
            'contributing_signals': [s.signal_id for s in signals]
        }
        
        for signal in signals:
            combined_indicators.update(signal.indicators)
            
        hybrid_signal = StreamSignal(
            signal_id=f"hybrid_{signal_type.value}_{symbol}_{int(datetime.now().timestamp())}",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            source=SignalSource.HYBRID,
            indicators=combined_indicators,
            metadata=combined_metadata,
            expires_at=datetime.now() + timedelta(minutes=45)
        )
        
        return hybrid_signal

class StreamSignalGenerator:
    """流式信号生成器主类"""
    
    def __init__(self, model_manager: ModelManager = None):
        self.model_manager = model_manager
        self.metrics_calculator = get_realtime_metrics_calculator()
        
        # 信号生成器
        self.technical_generator = TechnicalSignalGenerator()
        self.ml_generator = MLSignalGenerator(model_manager) if model_manager else None
        self.hybrid_generator = None
        
        if self.ml_generator:
            self.hybrid_generator = HybridSignalGenerator(
                self.technical_generator, 
                self.ml_generator
            )
        
        # 信号历史
        self.signal_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_signals = {}  # signal_id -> StreamSignal
        
        # 统计信息
        self.stats = {
            'total_signals_generated': 0,
            'signals_by_type': defaultdict(int),
            'signals_by_source': defaultdict(int),
            'signal_accuracy': 0.0,  # 需要回测数据
            'start_time': datetime.now()
        }
        
        # 任务管理
        self.generation_task = None
        self.cleanup_task = None
        self.is_running = False
        
        # 回调函数
        self.on_signal_callbacks = []
        
        logger.info("流式信号生成器初始化完成")
    
    async def start(self):
        """启动信号生成器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动生成任务
        self.generation_task = asyncio.create_task(self._signal_generation_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("流式信号生成器启动完成")
    
    async def stop(self):
        """停止信号生成器"""
        if not self.is_running:
            return
        
        logger.info("正在停止流式信号生成器...")
        self.is_running = False
        
        # 取消任务
        if self.generation_task:
            self.generation_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        logger.info("流式信号生成器已停止")
    
    async def _signal_generation_loop(self):
        """信号生成循环"""
        while self.is_running:
            try:
                # 获取所有活跃的计算器
                calculators = self.metrics_calculator.calculators
                
                for symbol, calculator in calculators.items():
                    try:
                        # 获取最新指标
                        current_metrics = calculator.get_all_current_metrics()
                        
                        if not current_metrics:
                            continue
                        
                        # 生成信号
                        await self._generate_signals_for_symbol(symbol, current_metrics)
                        
                    except Exception as e:
                        logger.error(f"为 {symbol} 生成信号失败: {e}")
                
                await asyncio.sleep(5)  # 5秒间隔
                
            except Exception as e:
                logger.error(f"信号生成循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _generate_signals_for_symbol(self, symbol: str, metrics: Dict[str, CalculatedMetric]):
        """为特定交易对生成信号"""
        try:
            generated_signals = []
            
            # 技术分析信号
            technical_signals = self.technical_generator.generate_signals(symbol, metrics)
            generated_signals.extend(technical_signals)
            
            # ML信号
            if self.ml_generator:
                ml_signals = self.ml_generator.generate_signals(symbol, metrics)
                generated_signals.extend(ml_signals)
            
            # 混合信号
            if self.hybrid_generator:
                hybrid_signals = self.hybrid_generator.generate_hybrid_signals(symbol, metrics)
                generated_signals.extend(hybrid_signals)
            
            # 处理生成的信号
            for signal in generated_signals:
                await self._process_new_signal(signal)
                
        except Exception as e:
            logger.error(f"生成信号失败 {symbol}: {e}")
    
    async def _process_new_signal(self, signal: StreamSignal):
        """处理新生成的信号"""
        # 去重检查（避免重复信号）
        similar_signals = [s for s in self.active_signals.values() 
                         if (s.symbol == signal.symbol and 
                             s.signal_type == signal.signal_type and
                             s.source == signal.source and
                             (datetime.now() - s.timestamp).total_seconds() < 300)]  # 5分钟内
        
        if similar_signals:
            logger.debug(f"跳过重复信号: {signal.signal_id}")
            return
        
        # 添加到活跃信号
        self.active_signals[signal.signal_id] = signal
        
        # 添加到历史
        self.signal_history[signal.symbol].append(signal)
        
        # 更新统计
        self.stats['total_signals_generated'] += 1
        self.stats['signals_by_type'][signal.signal_type.value] += 1
        self.stats['signals_by_source'][signal.source.value] += 1
        
        # 调用回调函数
        for callback in self.on_signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"信号回调错误: {e}")
        
        logger.info(f"生成新信号: {signal.signal_type.value} {signal.symbol} "
                   f"({signal.source.value}, 置信度: {signal.confidence:.2f})")
    
    async def _cleanup_loop(self):
        """清理过期信号"""
        while self.is_running:
            try:
                current_time = datetime.now()
                expired_signals = []
                
                for signal_id, signal in self.active_signals.items():
                    if not signal.is_valid():
                        expired_signals.append(signal_id)
                
                for signal_id in expired_signals:
                    del self.active_signals[signal_id]
                
                if expired_signals:
                    logger.debug(f"清理 {len(expired_signals)} 个过期信号")
                
                await asyncio.sleep(60)  # 每分钟清理一次
                
            except Exception as e:
                logger.error(f"信号清理错误: {e}")
                await asyncio.sleep(60)
    
    # 查询接口
    def get_active_signals(self, symbol: str = None) -> List[StreamSignal]:
        """获取活跃信号"""
        if symbol:
            return [s for s in self.active_signals.values() if s.symbol == symbol]
        return list(self.active_signals.values())
    
    def get_signal_history(self, symbol: str, limit: int = 100) -> List[StreamSignal]:
        """获取信号历史"""
        history = self.signal_history.get(symbol, deque())
        return list(history)[-limit:]
    
    def get_latest_signal(self, symbol: str, signal_type: SignalType = None) -> Optional[StreamSignal]:
        """获取最新信号"""
        signals = self.get_active_signals(symbol)
        
        if signal_type:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        if not signals:
            return None
        
        return max(signals, key=lambda s: s.timestamp)
    
    # 回调注册
    def on_signal(self, callback: Callable):
        """注册信号回调"""
        self.on_signal_callbacks.append(callback)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'active_signals_count': len(self.active_signals),
            'stats': dict(self.stats),
            'signal_summary': {
                'by_type': dict(self.stats['signals_by_type']),
                'by_source': dict(self.stats['signals_by_source'])
            },
            'generators_enabled': {
                'technical': True,
                'ml': self.ml_generator is not None,
                'hybrid': self.hybrid_generator is not None
            }
        }

# 全局实例
_stream_signal_generator_instance = None

def get_stream_signal_generator() -> StreamSignalGenerator:
    """获取流式信号生成器实例"""
    global _stream_signal_generator_instance
    if _stream_signal_generator_instance is None:
        try:
            model_manager = ModelManager()
            _stream_signal_generator_instance = StreamSignalGenerator(model_manager)
        except Exception as e:
            logger.warning(f"无法初始化模型管理器: {e}")
            _stream_signal_generator_instance = StreamSignalGenerator()
    return _stream_signal_generator_instance