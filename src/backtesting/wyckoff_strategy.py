"""
Wyckoff Volume-Price Analysis Strategy
Implementation of Wyckoff Method for price action and volume analysis
Includes accumulation/distribution phases and volume-spread analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

warnings.filterwarnings('ignore')

# Technical analysis
import talib

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator

logger = setup_logger(__name__)

class WyckoffAnalyzer:
    """Wyckoff Method analysis for market structure and phases"""

    def __init__(self):
        self.price_levels = {}
        self.volume_analysis = {}
        self.market_phases = {}
        self.wyckoff_signals = {}

        # Wyckoff phase definitions
        self.phases = {
            'ACCUMULATION': 1,
            'MARKUP': 2,
            'DISTRIBUTION': 3,
            'MARKDOWN': 4,
            'NEUTRAL': 0
        }

    @timing_decorator

    def analyze_wyckoff_structure(self, df: pd.DataFrame,
                                 symbol: str = "Unknown") -> Dict[str, Any]:
        """
        Comprehensive Wyckoff analysis

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol

        Returns:
            Dictionary with Wyckoff analysis results
        """
        try:
            logger.info(f"Starting Wyckoff analysis for {symbol}")

            if df.empty or len(df) < 100:
                logger.warning("Insufficient data for Wyckoff analysis")
                return {}

            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")

            df_clean = df[required_cols].dropna()

            results = {
                'symbol': symbol,
                'analysis_date': datetime.now().isoformat(),
                'data_points': len(df_clean)
            }

            # Core Wyckoff components
            results['volume_spread_analysis'] = self._analyze_volume_spread(df_clean)
            results['supply_demand'] = self._analyze_supply_demand(df_clean)
            results['market_phases'] = self._identify_market_phases(df_clean)
            results['wyckoff_events'] = self._identify_wyckoff_events(df_clean)
            results['composite_operator'] = self._analyze_composite_operator(df_clean)
            results['trading_signals'] = self._generate_wyckoff_signals(df_clean)

            # Store analysis for symbol
            self.wyckoff_signals[symbol] = results

            logger.info(f"Wyckoff analysis completed for {symbol}")
            return results

        except Exception as e:
            logger.error(f"Wyckoff analysis failed: {e}")
            return {}

    def _analyze_volume_spread_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Volume Spread Analysis (VSA) - core of Wyckoff method"""
        try:
            # Calculate spreads (price ranges)
            df = df.copy()
            df['spread'] = df['high'] - df['low']
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

            # Relative volume
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma']

            # Price change
            df['price_change'] = df['close'].pct_change()

            # Classify volume and spread
            volume_threshold = df['relative_volume'].quantile(0.7)
            spread_threshold = df['spread'].quantile(0.7)

            df['high_volume'] = df['relative_volume'] > volume_threshold
            df['low_volume'] = df['relative_volume'] < (1 / volume_threshold)
            df['wide_spread'] = df['spread'] > spread_threshold
            df['narrow_spread'] = df['spread'] < (spread_threshold / 2)

            # VSA patterns
            vsa_patterns = []

            for i in range(1, len(df)):
                row = df.iloc[i]
                prev_row = df.iloc[i-1]

                pattern = {
                    'date': df.index[i],
                    'close': row['close'],
                    'volume': row['volume'],
                    'spread': row['spread'],
                    'pattern': 'NEUTRAL',
                    'strength': 0.5
                }

                # High volume + Wide spread + Up close = Professional Buying
                if (row['high_volume'] and row['wide_spread'] and
                    row['close'] > row['open'] and row['close'] > prev_row['close']):
                    pattern['pattern'] = 'BUYING_CLIMAX'
                    pattern['strength'] = 0.8

                # High volume + Wide spread + Down close = Professional Selling
                elif (row['high_volume'] and row['wide_spread'] and
                      row['close'] < row['open'] and row['close'] < prev_row['close']):
                    pattern['pattern'] = 'SELLING_CLIMAX'
                    pattern['strength'] = 0.2

                # High volume + Narrow spread = Absorption
                elif row['high_volume'] and row['narrow_spread']:
                    pattern['pattern'] = 'ABSORPTION'
                    pattern['strength'] = 0.6

                # Low volume + Wide spread + Up close = Weak Rally
                elif (row['low_volume'] and row['wide_spread'] and
                      row['close'] > row['open']):
                    pattern['pattern'] = 'WEAK_RALLY'
                    pattern['strength'] = 0.3

                # Low volume + Wide spread + Down close = Weak Decline
                elif (row['low_volume'] and row['wide_spread'] and
                      row['close'] < row['open']):
                    pattern['pattern'] = 'WEAK_DECLINE'
                    pattern['strength'] = 0.7

                # High volume + Narrow spread + No progress = Testing
                elif (row['high_volume'] and row['narrow_spread'] and
                      abs(row['price_change']) < 0.01):
                    pattern['pattern'] = 'TESTING'
                    pattern['strength'] = 0.5

                vsa_patterns.append(pattern)

            # Statistics
            pattern_counts = {}
            for pattern in vsa_patterns:
                pattern_type = pattern['pattern']
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1

            return {
                'vsa_patterns': vsa_patterns[-50:],  # Last 50 patterns
                'pattern_counts': pattern_counts,
                'avg_relative_volume': df['relative_volume'].mean(),
                'avg_spread': df['spread'].mean(),
                'volume_trend': 'INCREASING' if df['volume'].iloc[-20:].mean() > df['volume'].iloc[-40:-20].mean() else 'DECREASING'
            }

        except Exception as e:
            logger.error(f"VSA analysis failed: {e}")
            return {}

    def _analyze_volume_spread(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Simplified volume-spread analysis wrapper"""
        return self._analyze_volume_spread_analysis(df)

    def _analyze_supply_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze supply and demand dynamics"""
        try:
            df = df.copy()

            # Effort vs Result analysis
            df['price_change'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            df['effort'] = df['volume'] / df['volume_ma']  # Relative volume as effort
            df['result'] = abs(df['price_change'])  # Price change as result

            # Supply/Demand indicators
            df['buying_pressure'] = np.where(
                df['close'] > df['open'],
                df['volume'] * ((df['close'] - df['low']) / (df['high'] - df['low'])),
                0
            )

            df['selling_pressure'] = np.where(
                df['close'] < df['open'],
                df['volume'] * ((df['high'] - df['close']) / (df['high'] - df['low'])),
                0
            )

            # Accumulation/Distribution Line
            df['ad_line'] = ((df['close'] - df['low'] - (df['high'] - df['close'])) /
                           (df['high'] - df['low']) * df['volume']).fillna(0).cumsum()

            # Money Flow Index
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['money_flow'] = df['typical_price'] * df['volume']

            # Positive and negative money flows
            df['positive_mf'] = np.where(df['typical_price'] > df['typical_price'].shift(1),
                                       df['money_flow'], 0)
            df['negative_mf'] = np.where(df['typical_price'] < df['typical_price'].shift(1),
                                       df['money_flow'], 0)

            # Money Flow Index (14 periods)
            positive_mf_sum = df['positive_mf'].rolling(14).sum()
            negative_mf_sum = df['negative_mf'].rolling(14).sum()

            df['mfi'] = 100 - (100 / (1 + positive_mf_sum / negative_mf_sum))

            # Supply/Demand balance
            recent_buying = df['buying_pressure'].iloc[-20:].sum()
            recent_selling = df['selling_pressure'].iloc[-20:].sum()

            if recent_buying + recent_selling > 0:
                demand_ratio = recent_buying / (recent_buying + recent_selling)
            else:
                demand_ratio = 0.5

            # Strength analysis
            current_mfi = df['mfi'].iloc[-1]
            current_ad = df['ad_line'].iloc[-1]
            prev_ad = df['ad_line'].iloc[-20]

            supply_demand_balance = {
                'demand_ratio': demand_ratio,
                'supply_ratio': 1 - demand_ratio,
                'mfi_current': current_mfi,
                'ad_line_trend': 'BULLISH' if current_ad > prev_ad else 'BEARISH',
                'balance': 'DEMAND' if demand_ratio > 0.6 else 'SUPPLY' if demand_ratio < 0.4 else 'NEUTRAL'
            }

            return supply_demand_balance

        except Exception as e:
            logger.error(f"Supply/demand analysis failed: {e}")
            return {}

    def _identify_market_phases(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify Wyckoff market phases"""
        try:
            df = df.copy()

            # Calculate key indicators for phase identification
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['price_change'] = df['close'].pct_change(20)  # 20-period change
            df['volume_change'] = df['volume'].pct_change(20)

            # Identify phases
            phases = []

            for i in range(50, len(df)):  # Need enough data for indicators
                row = df.iloc[i]

                # Phase classification logic
                price_trend = row['sma_20'] - df.iloc[i-20]['sma_20']
                volume_trend = row['volume_ma'] - df.iloc[i-20]['volume_ma']

                phase_data = {
                    'date': df.index[i],
                    'close': row['close'],
                    'phase': 'NEUTRAL',
                    'confidence': 0.5,
                    'characteristics': []
                }

                # Accumulation Phase
                # - Sideways price action
                # - High volume on down days, low volume on up days
                # - Smart money buying
                if (abs(price_trend) < row['close'] * 0.05 and  # Sideways
                    row['volume_ma'] > df['volume_ma'].iloc[i-40:i].mean()):  # Higher volume

                    phase_data['phase'] = 'ACCUMULATION'
                    phase_data['confidence'] = 0.7
                    phase_data['characteristics'] = [
                        'Sideways price action',
                        'Increased volume',
                        'Potential smart money buying'
                    ]

                # Markup Phase
                # - Rising prices
                # - Increasing volume on breakouts
                # - Sustained upward movement
                elif (price_trend > 0 and
                      row['sma_20'] > row['sma_50'] and
                      row['volume_ma'] > df['volume_ma'].iloc[i-20:i].mean()):

                    phase_data['phase'] = 'MARKUP'
                    phase_data['confidence'] = 0.8
                    phase_data['characteristics'] = [
                        'Rising prices',
                        'Increasing volume',
                        'Upward momentum'
                    ]

                # Distribution Phase
                # - Sideways to slightly up price action at high levels
                # - High volume on up days, low volume on down days
                # - Smart money selling
                elif (abs(price_trend) < row['close'] * 0.05 and
                      row['close'] > df['close'].iloc[i-100:i].quantile(0.8) and  # Near highs
                      row['volume_ma'] > df['volume_ma'].iloc[i-40:i].mean()):

                    phase_data['phase'] = 'DISTRIBUTION'
                    phase_data['confidence'] = 0.7
                    phase_data['characteristics'] = [
                        'Sideways action near highs',
                        'High volume',
                        'Potential smart money selling'
                    ]

                # Markdown Phase
                # - Falling prices
                # - Increasing volume on breakdowns
                # - Sustained downward movement
                elif (price_trend < 0 and
                      row['sma_20'] < row['sma_50'] and
                      row['volume_ma'] > df['volume_ma'].iloc[i-20:i].mean()):

                    phase_data['phase'] = 'MARKDOWN'
                    phase_data['confidence'] = 0.8
                    phase_data['characteristics'] = [
                        'Falling prices',
                        'Increasing volume',
                        'Downward momentum'
                    ]

                phases.append(phase_data)

            # Current phase
            current_phase = phases[-1] if phases else {'phase': 'NEUTRAL', 'confidence': 0.5}

            # Phase distribution
            phase_counts = {}
            for phase in phases[-100:]:  # Last 100 periods
                phase_type = phase['phase']
                phase_counts[phase_type] = phase_counts.get(phase_type, 0) + 1

            return {
                'current_phase': current_phase,
                'phase_history': phases[-50:],  # Last 50 phases
                'phase_distribution': phase_counts,
                'trend_strength': self._calculate_trend_strength(df)
            }

        except Exception as e:
            logger.error(f"Phase identification failed: {e}")
            return {}

    def _identify_wyckoff_events(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify specific Wyckoff events and patterns"""
        try:
            df = df.copy()

            # Calculate indicators for event identification
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)

            events = []

            for i in range(20, len(df)):
                row = df.iloc[i]
                prev_rows = df.iloc[i-20:i]

                event_data = {
                    'date': df.index[i],
                    'event': 'NONE',
                    'significance': 0.0,
                    'description': ''
                }

                # Spring (False breakdown in accumulation)
                if (row['low'] < prev_rows['low'].min() and  # New low
                    row['close'] > prev_rows['close'].quantile(0.5) and  # But closed higher
                    row['volume'] > row['volume_ma'] * 1.5):  # High volume

                    event_data.update({
                        'event': 'SPRING',
                        'significance': 0.8,
                        'description': 'False breakdown with high volume - potential accumulation end'
                    })

                # Upthrust (False breakout in distribution)
                elif (row['high'] > prev_rows['high'].max() and  # New high
                      row['close'] < prev_rows['close'].quantile(0.5) and  # But closed lower
                      row['volume'] > row['volume_ma'] * 1.5):  # High volume

                    event_data.update({
                        'event': 'UPTHRUST',
                        'significance': 0.8,
                        'description': 'False breakout with high volume - potential distribution end'
                    })

                # Sign of Strength (SOS)
                elif (row['close'] > prev_rows['high'].max() and  # Breakout
                      row['volume'] > row['volume_ma'] * 1.3 and  # Good volume
                      row['close'] > row['open']):  # Strong close

                    event_data.update({
                        'event': 'SIGN_OF_STRENGTH',
                        'significance': 0.7,
                        'description': 'Breakout with volume - strength confirmation'
                    })

                # Sign of Weakness (SOW)
                elif (row['close'] < prev_rows['low'].min() and  # Breakdown
                      row['volume'] > row['volume_ma'] * 1.3 and  # Good volume
                      row['close'] < row['open']):  # Weak close

                    event_data.update({
                        'event': 'SIGN_OF_WEAKNESS',
                        'significance': 0.7,
                        'description': 'Breakdown with volume - weakness confirmation'
                    })

                # Last Point of Support (LPS)
                elif (row['low'] > prev_rows['low'].min() and  # Higher low
                      row['volume'] < row['volume_ma'] * 0.8 and  # Low volume
                      df.iloc[i-5:i]['close'].min() > prev_rows['close'].quantile(0.3)):  # Above support

                    event_data.update({
                        'event': 'LAST_POINT_OF_SUPPORT',
                        'significance': 0.6,
                        'description': 'Higher low on low volume - potential support'
                    })

                # Last Point of Supply (LPSY)
                elif (row['high'] < prev_rows['high'].max() and  # Lower high
                      row['volume'] < row['volume_ma'] * 0.8 and  # Low volume
                      df.iloc[i-5:i]['close'].max() < prev_rows['close'].quantile(0.7)):  # Below resistance

                    event_data.update({
                        'event': 'LAST_POINT_OF_SUPPLY',
                        'significance': 0.6,
                        'description': 'Lower high on low volume - potential resistance'
                    })

                if event_data['event'] != 'NONE':
                    events.append(event_data)

            # Get recent significant events
            recent_events = [e for e in events[-50:] if e['significance'] > 0.6]

            return {
                'recent_events': recent_events,
                'total_events': len(events),
                'event_summary': self._summarize_events(events)
            }

        except Exception as e:
            logger.error(f"Wyckoff event identification failed: {e}")
            return {}

    def _analyze_composite_operator(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the composite operator (smart money) activity"""
        try:
            df = df.copy()

            # Calculate composite operator indicators
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['price_ma'] = df['close'].rolling(20).mean()

            # Smart money indicators
            smart_money_activity = []

            for i in range(20, len(df)):
                row = df.iloc[i]

                # Large volume with minimal price movement = Absorption
                if (row['volume'] > row['volume_ma'] * 1.5 and
                    abs(row['close'] - row['open']) < (row['high'] - row['low']) * 0.3):

                    smart_money_activity.append({
                        'date': df.index[i],
                        'activity': 'ABSORPTION',
                        'strength': 0.7,
                        'interpretation': 'Smart money absorbing supply/demand'
                    })

                # High volume at key levels
                elif (row['volume'] > row['volume_ma'] * 2.0 and
                      (row['close'] == df.iloc[i-20:i]['high'].max() or
                       row['close'] == df.iloc[i-20:i]['low'].min())):

                    smart_money_activity.append({
                        'date': df.index[i],
                        'activity': 'TESTING',
                        'strength': 0.8,
                        'interpretation': 'Testing of key levels with volume'
                    })

            # Current smart money sentiment
            recent_activity = smart_money_activity[-10:] if smart_money_activity else []

            if recent_activity:
                absorption_count = len([a for a in recent_activity if a['activity'] == 'ABSORPTION'])
                testing_count = len([a for a in recent_activity if a['activity'] == 'TESTING'])

                if absorption_count > testing_count:
                    sentiment = 'ACCUMULATING'
                elif testing_count > absorption_count:
                    sentiment = 'TESTING_LEVELS'
                else:
                    sentiment = 'NEUTRAL'
            else:
                sentiment = 'NEUTRAL'

            return {
                'current_sentiment': sentiment,
                'recent_activity': recent_activity,
                'activity_count': len(smart_money_activity),
                'interpretation': self._interpret_smart_money_activity(sentiment, recent_activity)
            }

        except Exception as e:
            logger.error(f"Composite operator analysis failed: {e}")
            return {}

    def _generate_wyckoff_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on Wyckoff analysis"""
        try:
            # Get all analysis components
            vsa = self._analyze_volume_spread_analysis(df)
            supply_demand = self._analyze_supply_demand(df)
            phases = self._identify_market_phases(df)
            events = self._identify_wyckoff_events(df)

            # Generate composite signal
            signal_strength = 0.0
            signal_direction = 0  # -1: Bearish, 0: Neutral, 1: Bullish
            signal_components = []

            # Phase-based signals
            current_phase = phases.get('current_phase', {})
            phase_type = current_phase.get('phase', 'NEUTRAL')

            if phase_type == 'ACCUMULATION':
                signal_strength += 0.3
                signal_direction += 1
                signal_components.append('Accumulation phase detected')
            elif phase_type == 'DISTRIBUTION':
                signal_strength += 0.3
                signal_direction -= 1
                signal_components.append('Distribution phase detected')
            elif phase_type == 'MARKUP':
                signal_strength += 0.4
                signal_direction += 1
                signal_components.append('Markup phase - bullish trend')
            elif phase_type == 'MARKDOWN':
                signal_strength += 0.4
                signal_direction -= 1
                signal_components.append('Markdown phase - bearish trend')

            # Supply/demand signals
            balance = supply_demand.get('balance', 'NEUTRAL')
            if balance == 'DEMAND':
                signal_strength += 0.2
                signal_direction += 1
                signal_components.append('Demand exceeding supply')
            elif balance == 'SUPPLY':
                signal_strength += 0.2
                signal_direction -= 1
                signal_components.append('Supply exceeding demand')

            # Recent events signals
            recent_events = events.get('recent_events', [])
            for event in recent_events[-3:]:  # Last 3 events
                if event['event'] in ['SPRING', 'SIGN_OF_STRENGTH', 'LAST_POINT_OF_SUPPORT']:
                    signal_strength += 0.15
                    signal_direction += 1
                    signal_components.append(f"Bullish event: {event['event']}")
                elif event['event'] in ['UPTHRUST', 'SIGN_OF_WEAKNESS', 'LAST_POINT_OF_SUPPLY']:
                    signal_strength += 0.15
                    signal_direction -= 1
                    signal_components.append(f"Bearish event: {event['event']}")

            # VSA patterns
            recent_patterns = vsa.get('vsa_patterns', [])
            if recent_patterns:
                last_pattern = recent_patterns[-1]
                pattern_type = last_pattern.get('pattern', 'NEUTRAL')

                if pattern_type in ['BUYING_CLIMAX', 'ABSORPTION']:
                    signal_strength += 0.1
                    signal_direction += 1
                    signal_components.append(f"VSA: {pattern_type}")
                elif pattern_type in ['SELLING_CLIMAX', 'WEAK_RALLY']:
                    signal_strength += 0.1
                    signal_direction -= 1
                    signal_components.append(f"VSA: {pattern_type}")

            # Normalize signal
            signal_strength = min(signal_strength, 1.0)

            # Determine final signal
            if signal_direction > 0 and signal_strength > 0.6:
                final_signal = 'BULLISH'
                confidence = signal_strength
            elif signal_direction < 0 and signal_strength > 0.6:
                final_signal = 'BEARISH'
                confidence = signal_strength
            else:
                final_signal = 'NEUTRAL'
                confidence = 0.5

            return {
                'signal': final_signal,
                'confidence': confidence,
                'strength': signal_strength,
                'components': signal_components,
                'recommendation': self._generate_recommendation(final_signal, confidence),
                'risk_level': 'HIGH' if confidence > 0.8 else 'MEDIUM' if confidence > 0.6 else 'LOW'
            }

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.5}

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength"""
        try:
            if len(df) < 50:
                return 0.5

            recent_closes = df['close'].iloc[-20:]
            trend_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]

            # Normalize trend strength (0 to 1)
            max_price = df['close'].iloc[-50:].max()
            min_price = df['close'].iloc[-50:].min()
            price_range = max_price - min_price

            if price_range > 0:
                normalized_slope = abs(trend_slope) / (price_range / 20)
                return min(normalized_slope, 1.0)
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return 0.5

    def _summarize_events(self, events: List[Dict]) -> Dict[str, int]:
        """Summarize Wyckoff events"""
        summary = {}
        for event in events:
            event_type = event['event']
            summary[event_type] = summary.get(event_type, 0) + 1
        return summary

    def _interpret_smart_money_activity(self, sentiment: str, recent_activity: List[Dict]) -> str:
        """Interpret smart money activity"""
        if sentiment == 'ACCUMULATING':
            return "Smart money appears to be accumulating positions. Look for continuation patterns."
        elif sentiment == 'TESTING_LEVELS':
            return "Smart money is testing key levels. Prepare for potential breakout/breakdown."
        else:
            return "Smart money activity is neutral. Monitor for changes in volume patterns."

    def _generate_recommendation(self, signal: str, confidence: float) -> str:
        """Generate trading recommendation"""
        if signal == 'BULLISH' and confidence > 0.7:
            return "Consider long positions with proper risk management."
        elif signal == 'BEARISH' and confidence > 0.7:
            return "Consider short positions or exit long positions."
        elif confidence > 0.6:
            return f"Moderate {signal.lower()} bias. Wait for confirmation."
        else:
            return "No clear direction. Stay in cash or reduce positions."

    def get_wyckoff_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of Wyckoff analysis for a symbol"""
        if symbol not in self.wyckoff_signals:
            return {}

        analysis = self.wyckoff_signals[symbol]

        return {
            'symbol': symbol,
            'current_phase': analysis.get('market_phases', {}).get('current_phase', {}),
            'signal': analysis.get('trading_signals', {}),
            'recent_events': analysis.get('wyckoff_events', {}).get('recent_events', [])[-5:],
            'supply_demand_balance': analysis.get('supply_demand', {}),
            'analysis_timestamp': analysis.get('analysis_date')
        }

# Convenience functions

def analyze_wyckoff_symbol(df: pd.DataFrame, symbol: str = "Unknown") -> Dict[str, Any]:
    """Convenient function to analyze Wyckoff patterns for a symbol"""
    analyzer = WyckoffAnalyzer()
    return analyzer.analyze_wyckoff_structure(df, symbol)

def get_wyckoff_signal(df: pd.DataFrame) -> Tuple[str, float]:
    """Get simple Wyckoff signal"""
    analyzer = WyckoffAnalyzer()
    analysis = analyzer.analyze_wyckoff_structure(df)

    signals = analysis.get('trading_signals', {})
    return signals.get('signal', 'NEUTRAL'), signals.get('confidence', 0.5)

# Usage example
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='D')

    # Generate price data with accumulation/distribution phases
    base_price = 45000
    prices = [base_price]
    volumes = []

    for i in range(1, 500):
        # Create different phases
        if i < 100:  # Accumulation
            price_change = np.random.randn() * 50
            volume_mult = 1.2 if price_change < 0 else 0.8  # Higher volume on down days
        elif i < 200:  # Markup
            price_change = np.random.randn() * 80 + 30  # Upward bias
            volume_mult = 1.1
        elif i < 350:  # Distribution
            price_change = np.random.randn() * 40
            volume_mult = 1.3 if price_change > 0 else 0.9  # Higher volume on up days
        else:  # Markdown
            price_change = np.random.randn() * 100 - 40  # Downward bias
            volume_mult = 1.2

        new_price = max(prices[-1] + price_change, 1000)  # Prevent negative prices
        prices.append(new_price)

        base_volume = 2000
        volume = int(base_volume * volume_mult * (0.5 + np.random.rand()))
        volumes.append(volume)

    # Create DataFrame
    sample_df = pd.DataFrame({
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.randn()) * 0.02) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.randn()) * 0.02) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    }, index=dates)

    # Test Wyckoff analysis
    analyzer = WyckoffAnalyzer()
    results = analyzer.analyze_wyckoff_structure(sample_df, "BTCUSDT")

    print("Wyckoff Analysis Results:")
    print(f"Current Phase: {results.get('market_phases', {}).get('current_phase', {})}")
    print(f"Trading Signal: {results.get('trading_signals', {})}")
    print(f"Supply/Demand: {results.get('supply_demand', {}).get('balance', 'N/A')}")

    # Test signal generation
    signal, confidence = get_wyckoff_signal(sample_df)
    print(f"Simple Signal: {signal} (Confidence: {confidence:.2f})")

    print("Wyckoff analysis example completed!")
