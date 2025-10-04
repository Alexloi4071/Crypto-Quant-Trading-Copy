"""
Signal Generator Module
Generates trading signals by combining ML models, technical analysis, and market conditions
Supports multiple signal types and confidence scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger, get_trading_logger
from src.utils.helpers import timing_decorator
# DEPRECATED: 舊實時特徵入口，信號計算請直接使用版本化特徵或策略層生成
from src.models.model_manager import ModelOptimizer as ModelManager
from src.backtesting.wyckoff_strategy import WyckoffAnalyzer

logger = setup_logger(__name__)

class SignalType(Enum):
    """Signal types"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2

class SignalSource(Enum):
    """Signal sources"""
    ML_MODEL = "ml_model"
    TECHNICAL = "technical"
    WYCKOFF = "wyckoff"
    SENTIMENT = "sentiment"
    COMPOSITE = "composite"

@dataclass

class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    timeframe: str
    signal_type: SignalType
    confidence: float
    source: SignalSource
    price: float
    timestamp: datetime
    features: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type.value,
            'signal_name': self.signal_type.name,
            'confidence': self.confidence,
            'source': self.source.value,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'features': self.features,
            'metadata': self.metadata
        }

class SignalGenerator:
    """Main signal generation system"""

    def __init__(self):
        self.feature_engine = FeatureEngineering()
        self.model_manager = ModelManager()
        self.wyckoff_analyzer = WyckoffAnalyzer()

        # Signal generation settings
        self.confidence_threshold = config.get('MIN_SIGNAL_CONFIDENCE', 0.6)
        self.signal_weights = {
            SignalSource.ML_MODEL: 0.4,
            SignalSource.TECHNICAL: 0.25,
            SignalSource.WYCKOFF: 0.25,
            SignalSource.SENTIMENT: 0.1
        }

        # Signal history for tracking
        self.signal_history = {}
        self.last_signals = {}

        # Model cache
        self.loaded_models = {}

        logger.info("Signal generator initialized")

    @timing_decorator

    async def generate_signal(self, symbol: str, timeframe: str,
                            df: pd.DataFrame, version: str = None) -> Optional[TradingSignal]:
        """
        Generate comprehensive trading signal

        Args:
            symbol: Trading symbol
            timeframe: Time frame
            df: OHLCV DataFrame
            version: Model version to use

        Returns:
            TradingSignal object or None
        """
        try:
            logger.debug(f"Generating signal for {symbol}_{timeframe}")

            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}_{timeframe}")
                return None

            # Get current price
            current_price = float(df['close'].iloc[-1])
            timestamp = datetime.now()

            # Initialize signal components
            signal_components = {}

            # 1. ML Model Signal
            ml_signal = await self._generate_ml_signal(symbol, timeframe, df, version)
            if ml_signal:
                signal_components[SignalSource.ML_MODEL] = ml_signal

            # 2. Technical Analysis Signal
            technical_signal = self._generate_technical_signal(symbol, timeframe, df)
            if technical_signal:
                signal_components[SignalSource.TECHNICAL] = technical_signal

            # 3. Wyckoff Analysis Signal
            wyckoff_signal = self._generate_wyckoff_signal(symbol, timeframe, df)
            if wyckoff_signal:
                signal_components[SignalSource.WYCKOFF] = wyckoff_signal

            # 4. Sentiment Signal (placeholder for now)
            sentiment_signal = self._generate_sentiment_signal(symbol, timeframe)
            if sentiment_signal:
                signal_components[SignalSource.SENTIMENT] = sentiment_signal

            # Combine signals
            if not signal_components:
                logger.warning(f"No signal components generated for {symbol}_{timeframe}")
                return None

            composite_signal = self._combine_signals(signal_components)

            # Create final signal
            signal = TradingSignal(
                symbol=symbol,
                timeframe=timeframe,
                signal_type=composite_signal['signal_type'],
                confidence=composite_signal['confidence'],
                source=SignalSource.COMPOSITE,
                price=current_price,
                timestamp=timestamp,
                features=composite_signal['features'],
                metadata={
                    'components': {src.value: comp for src, comp in signal_components.items()},
                    'version': version,
                    'data_points': len(df)
                }
            )

            # Store in history
            self._store_signal_history(signal)

            # Log signal
            trading_logger = get_trading_logger(symbol, timeframe, version or 'latest')
            trading_logger.log_signal(signal.to_dict())

            logger.info(f"Generated {signal.signal_type.name} signal for {symbol}_{timeframe} "
                       f"(confidence: {signal.confidence:.2f})")

            return signal

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}_{timeframe}: {e}")
            return None

    async def _generate_ml_signal(self, symbol: str, timeframe: str,
                                df: pd.DataFrame, version: str = None) -> Optional[Dict[str, Any]]:
        """Generate ML model-based signal"""
        try:
            # Load model
            model_key = f"{symbol}_{timeframe}_{version or 'latest'}"

            if model_key not in self.loaded_models:
                model_info = await self.model_manager.load_model(symbol, timeframe, version)
                if not model_info:
                    logger.warning(f"No ML model found for {symbol}_{timeframe}")
                    return None
                self.loaded_models[model_key] = model_info

            model_info = self.loaded_models[model_key]
            model = model_info['model']
            feature_list = model_info.get('feature_list', [])

            # Generate features
            features_df = self.feature_engine.generate_all_features(df)

            if features_df.empty:
                logger.warning("Failed to generate features for ML prediction")
                return None

            # Prepare feature vector
            latest_features = features_df.iloc[-1]

            # Select only features used by the model
            if feature_list:
                available_features = [f for f in feature_list if f in latest_features.index]
                if len(available_features) < len(feature_list) * 0.8:  # At least 80% of features
                    logger.warning(f"Missing too many features: {len(available_features)}/{len(feature_list)}")
                    return None
                feature_vector = latest_features[available_features].values.reshape(1, -1)
            else:
                feature_vector = latest_features.values.reshape(1, -1)

            # Make prediction
            prediction = model.predict(feature_vector)[0]

            # Get prediction probability/confidence
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_vector)[0]
                    confidence = float(max(probabilities))

                    # Adjust confidence based on prediction strength
                    if prediction == 1:  # Long
                        confidence = float(probabilities[2] if len(probabilities) > 2 else probabilities[1])
                    elif prediction == -1:  # Short
                        confidence = float(probabilities[0])
                    else:  # Neutral
                        confidence = float(probabilities[1] if len(probabilities) > 2 else max(probabilities))
                else:
                    confidence = 0.7  # Default confidence
            except Exception as e:
                logger.debug(f"Failed to get prediction probability: {e}")
                confidence = 0.7

            # Convert prediction to signal type
            if prediction == 1:
                signal_type = SignalType.BUY if confidence < 0.8 else SignalType.STRONG_BUY
            elif prediction == -1:
                signal_type = SignalType.SELL if confidence < 0.8 else SignalType.STRONG_SELL
            else:
                signal_type = SignalType.NEUTRAL

            return {
                'signal_type': signal_type,
                'confidence': confidence,
                'raw_prediction': int(prediction),
                'features_used': len(available_features) if feature_list else len(latest_features),
                'model_version': version or 'latest'
            }

        except Exception as e:
            logger.error(f"ML signal generation failed: {e}")
            return None

    def _generate_technical_signal(self, symbol: str, timeframe: str,
                                 df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate technical analysis signal"""
        try:
            # Generate technical features
            technical_df = self.feature_engine.calculate_technical_indicators(df)

            if technical_df.empty:
                return None

            # Get latest values
            latest = technical_df.iloc[-1]

            # Technical signal logic
            signal_score = 0.0
            signal_reasons = []

            # Moving Average signals
            if 'sma_20' in latest.index and 'sma_50' in latest.index:
                if latest['sma_20'] > latest['sma_50']:
                    signal_score += 0.2
                    signal_reasons.append("SMA 20 > SMA 50")
                else:
                    signal_score -= 0.2
                    signal_reasons.append("SMA 20 < SMA 50")

            # RSI signals
            if 'rsi_14' in latest.index:
                rsi = latest['rsi_14']
                if rsi < 30:
                    signal_score += 0.3
                    signal_reasons.append(f"RSI oversold ({rsi:.1f})")
                elif rsi > 70:
                    signal_score -= 0.3
                    signal_reasons.append(f"RSI overbought ({rsi:.1f})")

            # MACD signals
            if 'macd' in latest.index and 'macd_signal' in latest.index:
                if latest['macd'] > latest['macd_signal']:
                    signal_score += 0.15
                    signal_reasons.append("MACD bullish crossover")
                else:
                    signal_score -= 0.15
                    signal_reasons.append("MACD bearish crossover")

            # Bollinger Bands signals
            if all(col in latest.index for col in ['bb_upper', 'bb_lower', 'close']):
                close = df['close'].iloc[-1]
                if close <= latest['bb_lower']:
                    signal_score += 0.25
                    signal_reasons.append("Price at lower Bollinger Band")
                elif close >= latest['bb_upper']:
                    signal_score -= 0.25
                    signal_reasons.append("Price at upper Bollinger Band")

            # Volume signals
            if 'volume_sma_20' in latest.index:
                volume_ratio = df['volume'].iloc[-1] / latest['volume_sma_20']
                if volume_ratio > 1.5:
                    # High volume - amplify existing signal
                    signal_score *= 1.2
                    signal_reasons.append("High volume confirmation")

            # Determine signal type and confidence
            confidence = min(abs(signal_score), 1.0)

            if signal_score > 0.3:
                signal_type = SignalType.STRONG_BUY if signal_score > 0.6 else SignalType.BUY
            elif signal_score < -0.3:
                signal_type = SignalType.STRONG_SELL if signal_score < -0.6 else SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL

            return {
                'signal_type': signal_type,
                'confidence': confidence,
                'score': signal_score,
                'reasons': signal_reasons
            }

        except Exception as e:
            logger.error(f"Technical signal generation failed: {e}")
            return None

    def _generate_wyckoff_signal(self, symbol: str, timeframe: str,
                               df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate Wyckoff analysis signal"""
        try:
            # Perform Wyckoff analysis
            wyckoff_analysis = self.wyckoff_analyzer.analyze_wyckoff_structure(df, symbol)

            if not wyckoff_analysis:
                return None

            # Extract signal from Wyckoff analysis
            trading_signals = wyckoff_analysis.get('trading_signals', {})

            if not trading_signals:
                return None

            wyckoff_signal = trading_signals.get('signal', 'NEUTRAL')
            wyckoff_confidence = trading_signals.get('confidence', 0.5)

            # Convert Wyckoff signal to our signal type
            if wyckoff_signal == 'BULLISH':
                if wyckoff_confidence > 0.8:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
            elif wyckoff_signal == 'BEARISH':
                if wyckoff_confidence > 0.8:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL

            # Get current market phase
            market_phases = wyckoff_analysis.get('market_phases', {})
            current_phase = market_phases.get('current_phase', {})

            return {
                'signal_type': signal_type,
                'confidence': wyckoff_confidence,
                'wyckoff_signal': wyckoff_signal,
                'market_phase': current_phase.get('phase', 'NEUTRAL'),
                'phase_confidence': current_phase.get('confidence', 0.5),
                'components': trading_signals.get('components', [])
            }

        except Exception as e:
            logger.error(f"Wyckoff signal generation failed: {e}")
            return None

    def _generate_sentiment_signal(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Generate sentiment-based signal (placeholder)"""
        try:
            # This is a placeholder for future sentiment analysis integration
            # Could include:
            # - Social media sentiment
            # - News sentiment
            # - Fear & Greed index
            # - On-chain metrics

            # For now, return neutral signal
            return {
                'signal_type': SignalType.NEUTRAL,
                'confidence': 0.5,
                'sentiment_score': 0.0,
                'sources': ['placeholder']
            }

        except Exception as e:
            logger.error(f"Sentiment signal generation failed: {e}")
            return None

    def _combine_signals(self, signal_components: Dict[SignalSource, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple signal sources into composite signal"""
        try:
            # Weighted signal combination
            weighted_score = 0.0
            total_weight = 0.0
            combined_confidence = 0.0
            combined_features = {}

            for source, component in signal_components.items():
                weight = self.signal_weights.get(source, 0.1)
                signal_type = component['signal_type']
                confidence = component['confidence']

                # Convert signal type to numeric score
                signal_score = signal_type.value

                # Weight the signal by confidence and source weight
                effective_weight = weight * confidence
                weighted_score += signal_score * effective_weight
                total_weight += effective_weight
                combined_confidence += confidence * weight

                # Collect features
                combined_features[source.value] = component

            # Normalize
            if total_weight > 0:
                final_score = weighted_score / total_weight
                combined_confidence = combined_confidence / sum(self.signal_weights.values())
            else:
                final_score = 0.0
                combined_confidence = 0.5

            # Determine final signal type
            if final_score >= 1.5:
                signal_type = SignalType.STRONG_BUY
            elif final_score >= 0.5:
                signal_type = SignalType.BUY
            elif final_score <= -1.5:
                signal_type = SignalType.STRONG_SELL
            elif final_score <= -0.5:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.NEUTRAL

            # Adjust confidence based on agreement between signals
            signal_agreement = self._calculate_signal_agreement(signal_components)
            adjusted_confidence = combined_confidence * signal_agreement

            return {
                'signal_type': signal_type,
                'confidence': min(adjusted_confidence, 1.0),
                'raw_score': final_score,
                'features': combined_features,
                'agreement_score': signal_agreement,
                'components_count': len(signal_components)
            }

        except Exception as e:
            logger.error(f"Signal combination failed: {e}")
            return {
                'signal_type': SignalType.NEUTRAL,
                'confidence': 0.5,
                'features': {},
                'error': str(e)
            }

    def _calculate_signal_agreement(self, signal_components: Dict[SignalSource, Dict[str, Any]]) -> float:
        """Calculate agreement between different signal sources"""
        try:
            if len(signal_components) < 2:
                return 1.0

            signal_values = [comp['signal_type'].value for comp in signal_components.values()]

            # Calculate agreement as inverse of variance
            if len(set(signal_values)) == 1:
                # Perfect agreement
                return 1.0
            else:
                # Calculate how close signals are to each other
                mean_signal = np.mean(signal_values)
                variance = np.var(signal_values)

                # Convert variance to agreement score (0-1)
                # Max variance for signals is 4 (strong buy vs strong sell)
                agreement = max(0.0, 1.0 - variance / 4.0)
                return agreement

        except Exception as e:
            logger.error(f"Agreement calculation failed: {e}")
            return 0.5

    def _store_signal_history(self, signal: TradingSignal):
        """Store signal in history for tracking and analysis"""
        try:
            key = f"{signal.symbol}_{signal.timeframe}"

            if key not in self.signal_history:
                self.signal_history[key] = []

            # Store signal data
            signal_data = signal.to_dict()
            self.signal_history[key].append(signal_data)

            # Keep only last 1000 signals per symbol/timeframe
            if len(self.signal_history[key]) > 1000:
                self.signal_history[key] = self.signal_history[key][-1000:]

            # Update last signal
            self.last_signals[key] = signal_data

        except Exception as e:
            logger.error(f"Failed to store signal history: {e}")

    async def generate_batch_signals(self, symbols: List[str], timeframe: str,
                                   data_dict: Dict[str, pd.DataFrame],
                                   version: str = None) -> Dict[str, TradingSignal]:
        """Generate signals for multiple symbols"""
        try:
            signals = {}

            # Process symbols concurrently
            tasks = []
            for symbol in symbols:
                if symbol in data_dict and not data_dict[symbol].empty:
                    task = self.generate_signal(symbol, timeframe, data_dict[symbol], version)
                    tasks.append((symbol, task))

            # Wait for all signals to be generated
            for symbol, task in tasks:
                try:
                    signal = await task
                    if signal:
                        signals[symbol] = signal
                except Exception as e:
                    logger.error(f"Failed to generate signal for {symbol}: {e}")

            logger.info(f"Generated {len(signals)} signals for {timeframe}")
            return signals

        except Exception as e:
            logger.error(f"Batch signal generation failed: {e}")
            return {}

    def get_signal_history(self, symbol: str, timeframe: str,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get signal history for a symbol/timeframe"""
        key = f"{symbol}_{timeframe}"

        if key not in self.signal_history:
            return []

        return self.signal_history[key][-limit:]

    def get_last_signal(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the last signal for a symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        return self.last_signals.get(key)

    def get_signal_statistics(self, symbol: str = None,
                            timeframe: str = None) -> Dict[str, Any]:
        """Get signal generation statistics"""
        try:
            if symbol and timeframe:
                key = f"{symbol}_{timeframe}"
                if key not in self.signal_history:
                    return {}

                signals = self.signal_history[key]
            else:
                # All signals
                signals = []
                for signal_list in self.signal_history.values():
                    signals.extend(signal_list)

            if not signals:
                return {}

            # Calculate statistics
            signal_types = [s['signal_type'] for s in signals]
            confidences = [s['confidence'] for s in signals]
            sources = [s['source'] for s in signals]

            return {
                'total_signals': len(signals),
                'signal_distribution': {
                    'STRONG_BUY': signal_types.count(2),
                    'BUY': signal_types.count(1),
                    'NEUTRAL': signal_types.count(0),
                    'SELL': signal_types.count(-1),
                    'STRONG_SELL': signal_types.count(-2)
                },
                'average_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'source_distribution': dict(pd.Series(sources).value_counts()),
                'last_signal_time': signals[-1]['timestamp'] if signals else None
            }

        except Exception as e:
            logger.error(f"Failed to get signal statistics: {e}")
            return {}

    def clear_signal_history(self, symbol: str = None, timeframe: str = None):
        """Clear signal history"""
        try:
            if symbol and timeframe:
                key = f"{symbol}_{timeframe}"
                if key in self.signal_history:
                    del self.signal_history[key]
                if key in self.last_signals:
                    del self.last_signals[key]
            else:
                self.signal_history.clear()
                self.last_signals.clear()

            logger.info("Signal history cleared")

        except Exception as e:
            logger.error(f"Failed to clear signal history: {e}")

# Convenience functions

async def generate_quick_signal(symbol: str, timeframe: str,
                              df: pd.DataFrame) -> Optional[TradingSignal]:
    """Quick signal generation function"""
    generator = SignalGenerator()
    return await generator.generate_signal(symbol, timeframe, df)

def create_manual_signal(symbol: str, timeframe: str, signal_type: SignalType,
                        confidence: float, price: float,
                        reason: str = "Manual") -> TradingSignal:
    """Create a manual trading signal"""
    return TradingSignal(
        symbol=symbol,
        timeframe=timeframe,
        signal_type=signal_type,
        confidence=confidence,
        source=SignalSource.TECHNICAL,
        price=price,
        timestamp=datetime.now(),
        features={'manual_reason': reason},
        metadata={'manual': True}
    )

# Usage example
if __name__ == "__main__":

    async def test_signal_generator():
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=200, freq='1H')
        np.random.seed(42)

        prices = 45000 + np.cumsum(np.random.randn(200) * 50)

        sample_df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000, 5000, 200)
        }, index=dates)

        # Test signal generation
        generator = SignalGenerator()

        signal = await generator.generate_signal('BTCUSDT', '1h', sample_df)

        if signal:
            print(f"Generated signal: {signal.signal_type.name}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Price: {signal.price:.2f}")
            print(f"Components: {len(signal.metadata.get('components', {}))}")
        else:
            print("No signal generated")

        # Test batch signals
        batch_signals = await generator.generate_batch_signals(
            ['BTCUSDT'], '1h', {'BTCUSDT': sample_df}
        )

        print(f"Batch signals: {len(batch_signals)}")

        # Get statistics
        stats = generator.get_signal_statistics()
        print(f"Signal statistics: {stats}")

        print("Signal generator test completed!")

    # Run test
    asyncio.run(test_signal_generator())
