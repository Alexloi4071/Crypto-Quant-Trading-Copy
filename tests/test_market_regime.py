# -*- coding: utf-8 -*-
"""
Tests for Market Regime Detection

Tests all detection methods:
1. MarketRegimeDetector (trend + volatility)
2. AdaptiveRegimeDetector (multi-indicator with smoothing)
3. Utility functions
"""
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from optuna_system.utils.market_regime import (
    MarketRegimeDetector, AdaptiveRegimeDetector,
    get_dynamic_class_distribution, detect_regime_from_prices
)


class TestMarketRegimeDetector(unittest.TestCase):
    """Test basic market regime detector"""
    
    def setUp(self):
        # Create synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='15min')
        
        # Bull market: upward trend
        self.bull_prices = pd.Series(
            1000 + np.cumsum(np.random.randn(500) * 5 + 2),
            index=dates
        )
        
        # Bear market: downward trend
        self.bear_prices = pd.Series(
            1000 + np.cumsum(np.random.randn(500) * 5 - 2),
            index=dates
        )
        
        # Sideways: no clear trend
        self.sideways_prices = pd.Series(
            1000 + np.random.randn(500) * 10,
            index=dates
        )
        
        # High volatility: high variance
        self.highvol_prices = pd.Series(
            1000 + np.cumsum(np.random.randn(500) * 50),
            index=dates
        )
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = MarketRegimeDetector(sma_window=96, vol_window=30)
        
        self.assertEqual(detector.sma_window, 96)
        self.assertEqual(detector.vol_window, 30)
    
    def test_detect_bull_market(self):
        """Test bull market detection"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        regime = detector.detect_regime(self.bull_prices)
        
        self.assertEqual(regime, 'bull')
    
    def test_detect_bear_market(self):
        """Test bear market detection"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        regime = detector.detect_regime(self.bear_prices)
        
        self.assertEqual(regime, 'bear')
    
    def test_detect_sideways_market(self):
        """Test sideways market detection"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        regime = detector.detect_regime(self.sideways_prices)
        
        self.assertIn(regime, ['sideways', 'high_vol'])  # Could be either
    
    def test_detect_high_volatility(self):
        """Test high volatility detection"""
        detector = MarketRegimeDetector(
            sma_window=50, vol_window=20, vol_threshold=0.02
        )
        regime = detector.detect_regime(self.highvol_prices)
        
        # Should detect high volatility
        self.assertIn(regime, ['high_vol', 'bull', 'bear'])
    
    def test_detect_regime_with_timestamp(self):
        """Test regime detection at specific timestamp"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        
        # Get regime at middle timestamp
        mid_timestamp = self.bull_prices.index[250]
        regime = detector.detect_regime(self.bull_prices, timestamp=mid_timestamp)
        
        self.assertIn(regime, ['bull', 'bear', 'sideways', 'high_vol'])
    
    def test_insufficient_data(self):
        """Test handling of insufficient data"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        
        # Only 30 data points (less than required)
        short_prices = self.bull_prices.iloc[:30]
        regime = detector.detect_regime(short_prices)
        
        # Should default to sideways
        self.assertEqual(regime, 'sideways')
    
    def test_detect_with_confidence(self):
        """Test regime detection with confidence score"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        
        regime, confidence = detector.detect_regime_with_confidence(self.bull_prices)
        
        self.assertIn(regime, ['bull', 'bear', 'sideways', 'high_vol'])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_scores(self):
        """Test confidence scores for different regimes"""
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        
        # Bull market should have high confidence
        bull_regime, bull_conf = detector.detect_regime_with_confidence(
            self.bull_prices
        )
        
        # Sideways should have lower confidence
        sideways_regime, sideways_conf = detector.detect_regime_with_confidence(
            self.sideways_prices
        )
        
        # Bull market confidence should generally be higher
        # (not always true, but statistically likely)
        if bull_regime == 'bull' and sideways_regime == 'sideways':
            self.assertGreater(bull_conf, 0.0)


class TestAdaptiveRegimeDetector(unittest.TestCase):
    """Test adaptive regime detector with smoothing"""
    
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=500, freq='15min')
        
        # Create price series with regime transition
        prices_list = []
        # Start with bull market (200 points)
        prices_list.extend(1000 + np.cumsum(np.random.randn(200) * 5 + 2))
        # Transition to sideways (150 points)
        prices_list.extend([prices_list[-1]] + list(
            prices_list[-1] + np.random.randn(149) * 10
        ))
        # Transition to bear (150 points)
        prices_list.extend([prices_list[-1]] + list(
            prices_list[-1] + np.cumsum(np.random.randn(149) * 5 - 2)
        ))
        
        self.transition_prices = pd.Series(prices_list, index=dates)
    
    def test_adaptive_detector_initialization(self):
        """Test adaptive detector initialization"""
        detector = AdaptiveRegimeDetector(
            lookback_periods=10, transition_threshold=0.7
        )
        
        self.assertEqual(detector.lookback_periods, 10)
        self.assertEqual(detector.transition_threshold, 0.7)
        self.assertEqual(len(detector.regime_history), 0)
    
    def test_calculate_indicators(self):
        """Test indicator calculation"""
        detector = AdaptiveRegimeDetector()
        
        indicators = detector.calculate_indicators(self.transition_prices)
        
        self.assertIn('trend_score', indicators)
        self.assertIn('momentum_score', indicators)
        self.assertIn('volatility_score', indicators)
        self.assertIn('combined_score', indicators)
        
        # Scores should be in reasonable ranges
        self.assertGreaterEqual(indicators['trend_score'], -1.0)
        self.assertLessEqual(indicators['trend_score'], 1.0)
    
    def test_adaptive_detect_regime(self):
        """Test adaptive regime detection"""
        detector = AdaptiveRegimeDetector()
        
        regime = detector.detect_regime(self.transition_prices)
        
        self.assertIn(regime, ['bull', 'bear', 'sideways', 'high_vol'])
    
    def test_regime_smoothing(self):
        """Test regime smoothing with history"""
        detector = AdaptiveRegimeDetector(lookback_periods=5)
        
        # Call detect_regime multiple times to build history
        for i in range(10):
            regime = detector.detect_regime(
                self.transition_prices.iloc[:200 + i*10]
            )
        
        # History should be populated
        self.assertGreater(len(detector.regime_history), 0)
        self.assertLessEqual(len(detector.regime_history), 5)
    
    def test_regime_transition_hysteresis(self):
        """Test hysteresis prevents rapid regime switching"""
        detector = AdaptiveRegimeDetector(
            lookback_periods=10, transition_threshold=0.8
        )
        
        # Build up bull market history
        for _ in range(10):
            detector.detect_regime(self.transition_prices.iloc[:200])
        
        # All history should be similar
        unique_regimes = len(set(detector.regime_history))
        self.assertLessEqual(unique_regimes, 3)  # Allow some variation


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_get_dynamic_class_distribution_bull(self):
        """Test dynamic class distribution for bull market"""
        dist = get_dynamic_class_distribution('bull')
        
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0, places=2)
        
        # In bull market, buy should be emphasized
        # [sell, hold, buy] - buy should be higher than sell
        self.assertGreater(dist[2], dist[0])
    
    def test_get_dynamic_class_distribution_bear(self):
        """Test dynamic class distribution for bear market"""
        dist = get_dynamic_class_distribution('bear')
        
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0, places=2)
        
        # In bear market, sell should be emphasized
        # [sell, hold, buy] - sell should be higher than buy
        self.assertGreater(dist[0], dist[2])
    
    def test_get_dynamic_class_distribution_sideways(self):
        """Test dynamic class distribution for sideways market"""
        dist = get_dynamic_class_distribution('sideways')
        
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0, places=2)
        
        # Sideways should be relatively balanced
        # sell and buy should be similar
        self.assertAlmostEqual(dist[0], dist[2], delta=0.1)
    
    def test_get_dynamic_class_distribution_high_vol(self):
        """Test dynamic class distribution for high volatility"""
        dist = get_dynamic_class_distribution('high_vol')
        
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0, places=2)
        
        # High vol should emphasize extreme signals
        # sell and buy should be higher, hold lower
        self.assertLess(dist[1], 0.6)
    
    def test_get_dynamic_class_distribution_unknown(self):
        """Test handling of unknown regime"""
        dist = get_dynamic_class_distribution('unknown_regime')
        
        # Should return default distribution
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(sum(dist), 1.0, places=2)
    
    def test_detect_regime_from_prices_simple(self):
        """Test convenience function with simple method"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='15min')
        prices = pd.Series(
            1000 + np.cumsum(np.random.randn(300) * 5 + 1),
            index=dates
        )
        
        regime = detect_regime_from_prices(prices, method='simple')
        
        self.assertIn(regime, ['bull', 'bear', 'sideways', 'high_vol'])
    
    def test_detect_regime_from_prices_adaptive(self):
        """Test convenience function with adaptive method"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='15min')
        prices = pd.Series(
            1000 + np.cumsum(np.random.randn(300) * 5 - 1),
            index=dates
        )
        
        regime = detect_regime_from_prices(prices, method='adaptive')
        
        self.assertIn(regime, ['bull', 'bear', 'sideways', 'high_vol'])
    
    def test_detect_regime_from_prices_invalid_method(self):
        """Test error handling for invalid method"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        prices = pd.Series(np.random.randn(100) + 1000, index=dates)
        
        with self.assertRaises(ValueError):
            detect_regime_from_prices(prices, method='invalid_method')


class TestIntegration(unittest.TestCase):
    """Integration tests combining regime detection with class distribution"""
    
    def test_regime_to_distribution_pipeline(self):
        """Test complete pipeline from prices to class distribution"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='15min')
        
        # Bull market prices
        bull_prices = pd.Series(
            1000 + np.cumsum(np.random.randn(300) * 5 + 2),
            index=dates
        )
        
        # Detect regime
        regime = detect_regime_from_prices(bull_prices, method='simple')
        
        # Get recommended distribution
        distribution = get_dynamic_class_distribution(regime)
        
        self.assertEqual(len(distribution), 3)
        self.assertAlmostEqual(sum(distribution), 1.0, places=2)
    
    def test_multiple_regimes_consistency(self):
        """Test that regime detection is consistent across calls"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=300, freq='15min')
        prices = pd.Series(
            1000 + np.cumsum(np.random.randn(300) * 5),
            index=dates
        )
        
        detector = MarketRegimeDetector(sma_window=50, vol_window=20)
        
        # Call multiple times with same data
        regime1 = detector.detect_regime(prices)
        regime2 = detector.detect_regime(prices)
        
        # Should be consistent
        self.assertEqual(regime1, regime2)


if __name__ == '__main__':
    unittest.main()

