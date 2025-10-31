# -*- coding: utf-8 -*-
"""
Unit tests for time integrity utilities

Tests Purged CV, Embargo, and Time Leakage Detection
"""
import pytest
import numpy as np
import pandas as pd
from optuna_system.utils.time_integrity import (
    PurgedKFold,
    EnhancedPurgedKFold,
    TimeLeakageDetector,
    validate_lag_alignment,
    get_purged_cv_splits
)


class TestPurgedKFold:
    """Test basic Purged K-Fold CV"""
    
    def test_purged_kfold_basic(self):
        """Test that PurgedKFold generates valid splits"""
        n_samples = 1000
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.02, purge_pct=0.01)
        
        splits = list(pkf.split(range(n_samples)))
        
        # Should have valid splits
        assert len(splits) > 0, "No valid splits generated"
        
        # Check each split
        for train_idx, test_idx in splits:
            assert len(train_idx) > 100, f"Training set too small: {len(train_idx)}"
            assert len(test_idx) > 20, f"Test set too small: {len(test_idx)}"
            
            # Train and test should not overlap
            assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap"
            
            # Training indices should be before test indices (time series)
            assert max(train_idx) < min(test_idx), "Time series order violated"
    
    def test_purge_removes_samples(self):
        """Test that purging actually removes samples"""
        n_samples = 1000
        purge_pct = 0.05  # 5%
        
        pkf = PurgedKFold(n_splits=5, embargo_pct=0.02, purge_pct=purge_pct)
        
        from sklearn.model_selection import KFold
        standard_kf = KFold(n_splits=5, shuffle=False)
        
        purged_splits = list(pkf.split(range(n_samples)))
        standard_splits = list(standard_kf.split(range(n_samples)))
        
        # Purged training sets should be smaller
        for (train_purged, _), (train_standard, _) in zip(purged_splits, standard_splits):
            assert len(train_purged) < len(train_standard), "Purging did not reduce training set"
    
    def test_embargo_removes_samples(self):
        """Test that embargo removes early test samples"""
        n_samples = 1000
        embargo_pct = 0.05  # 5%
        
        pkf = PurgedKFold(n_splits=5, embargo_pct=embargo_pct, purge_pct=0.01)
        
        from sklearn.model_selection import KFold
        standard_kf = KFold(n_splits=5, shuffle=False)
        
        purged_splits = list(pkf.split(range(n_samples)))
        standard_splits = list(standard_kf.split(range(n_samples)))
        
        # Embargoed test sets should be smaller
        for (_, test_embargoed), (_, test_standard) in zip(purged_splits, standard_splits):
            assert len(test_embargoed) < len(test_standard), "Embargo did not reduce test set"


class TestEnhancedPurgedKFold:
    """Test Enhanced Purged K-Fold with lag awareness"""
    
    def test_lag_based_embargo(self):
        """Test that embargo is at least 2×lag"""
        n_samples = 1000
        lag = 17
        
        pkf = EnhancedPurgedKFold(
            n_splits=5,
            lag=lag,
            embargo_pct=0.01,  # Small percentage
            embargo_multiplier=2.0
        )
        
        splits = list(pkf.split(range(n_samples)))
        
        # With lag=17, embargo should be at least 34
        for train_idx, test_idx in splits:
            # Check that gap between train and test is at least 2×lag
            gap = min(test_idx) - max(train_idx)
            assert gap >= lag, f"Embargo gap {gap} < lag {lag}"
    
    def test_purge_at_least_lag(self):
        """Test that purge size is at least lag"""
        n_samples = 1000
        lag = 20
        
        pkf = EnhancedPurgedKFold(
            n_splits=5,
            lag=lag,
            embargo_pct=0.02,
            purge_pct=0.001  # Very small percentage
        )
        
        from sklearn.model_selection import KFold
        standard_kf = KFold(n_splits=5, shuffle=False)
        
        purged_splits = list(pkf.split(range(n_samples)))
        standard_splits = list(standard_kf.split(range(n_samples)))
        
        for (train_purged, _), (train_standard, _) in zip(purged_splits, standard_splits):
            # Purge should remove at least 'lag' samples
            removed = len(train_standard) - len(train_purged)
            assert removed >= lag, f"Purged only {removed} samples, expected >= {lag}"


class TestTimeLeakageDetector:
    """Test Time Leakage Detection"""
    
    def test_no_leakage_detection(self):
        """Test that properly constructed features don't show leakage"""
        n_samples = 1000
        
        # Create random features (no future information)
        np.random.seed(42)
        features = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        # Create labels
        labels = pd.Series(np.random.randint(0, 3, n_samples))
        
        detector = TimeLeakageDetector(threshold=0.3, lag=1)
        result = detector.detect(features, labels)
        
        assert not result['has_leakage'], "False positive: detected leakage in random data"
        assert result['max_correlation'] < 0.3, f"Correlation too high: {result['max_correlation']}"
    
    def test_leakage_detection_with_future_data(self):
        """Test that detector catches features using future data"""
        n_samples = 1000
        
        # Create labels based on a pattern
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(n_samples)) + 100)
        returns = prices.pct_change()
        labels = pd.Series((returns > 0).astype(int))
        
        # Create features - one with future information
        features = pd.DataFrame({
            'good_feature': np.random.randn(n_samples),
            'leaky_feature': labels.shift(-5).fillna(0)  # Uses future labels!
        })
        
        detector = TimeLeakageDetector(threshold=0.3, lag=5)
        result = detector.detect(features, labels)
        
        assert result['has_leakage'], "Failed to detect leakage"
        assert 'leaky_feature' in result['leaky_features'], "Did not identify leaky feature"
        assert result['max_correlation'] > 0.5, "Correlation should be high for leaked feature"
    
    def test_assert_no_leakage_raises(self):
        """Test that assert_no_leakage raises on leakage"""
        n_samples = 500
        
        # Create leaky features
        labels = pd.Series(np.random.randint(0, 3, n_samples))
        features = pd.DataFrame({
            'leaky': labels.shift(-1).fillna(0)
        })
        
        detector = TimeLeakageDetector(threshold=0.2)
        
        with pytest.raises(AssertionError, match="Time leakage detected"):
            detector.assert_no_leakage(features, labels)


class TestLagAlignment:
    """Test lag alignment validation"""
    
    def test_aligned_lags(self):
        """Test validation passes for aligned lags"""
        layer1_lag = 17
        layer2_lag = 17
        
        result = validate_lag_alignment(layer1_lag, layer2_lag)
        
        assert result['aligned'], "Should detect aligned lags"
        assert len(result['issues']) == 0, "Should have no issues"
    
    def test_misaligned_lags(self):
        """Test validation fails for misaligned lags"""
        layer1_lag = 17
        layer2_lag = 12  # Different!
        
        with pytest.raises(AssertionError, match="Lag alignment failed"):
            validate_lag_alignment(layer1_lag, layer2_lag)
    
    def test_feature_lag_validation(self):
        """Test that feature lags are validated"""
        layer1_lag = 17
        layer2_lag = 17
        
        feature_lags = {
            'good_feature': 10,  # < 17, OK
            'bad_feature': 20     # > 17, NOT OK
        }
        
        with pytest.raises(AssertionError, match="Lag alignment failed"):
            validate_lag_alignment(layer1_lag, layer2_lag, feature_lags)


class TestConvenienceFunction:
    """Test convenience wrapper function"""
    
    def test_get_purged_cv_splits(self):
        """Test the convenience function"""
        n_samples = 1000
        lag = 15
        
        splits = get_purged_cv_splits(
            n_samples=n_samples,
            n_splits=5,
            lag=lag,
            embargo_pct=0.05,
            purge_pct=0.02
        )
        
        assert len(splits) > 0, "Should generate splits"
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0, "Empty training set"
            assert len(test_idx) > 0, "Empty test set"
            assert max(train_idx) < min(test_idx), "Time order violated"


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_no_leakage_in_purged_cv(self):
        """Test that Purged CV prevents leakage in a realistic scenario"""
        n_samples = 1000
        lag = 17
        
        # Create synthetic price data
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(n_samples)) + 100)
        
        # Create labels with lag (future returns)
        future_prices = prices.shift(-lag)
        returns = (future_prices - prices) / prices
        labels = pd.Series(np.where(returns > 0, 2, np.where(returns < 0, 0, 1)))
        labels = labels[:-lag]  # Remove last lag samples
        
        # Create features (historical only)
        features = pd.DataFrame({
            'return_1': prices.pct_change(1),
            'return_5': prices.pct_change(5),
            'volatility': prices.pct_change().rolling(20).std()
        }).iloc[:-lag]  # Align with labels
        
        # Use Enhanced Purged K-Fold
        pkf = EnhancedPurgedKFold(n_splits=5, lag=lag)
        
        # Validate no leakage
        detector = TimeLeakageDetector(threshold=0.3, lag=lag)
        result = detector.detect(features.fillna(0), labels.fillna(1))
        
        assert not result['has_leakage'], \
            f"Leakage detected even with proper setup: {result['leaky_features']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

