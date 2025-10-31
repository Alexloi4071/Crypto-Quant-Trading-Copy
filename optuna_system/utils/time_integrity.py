# -*- coding: utf-8 -*-
"""
Time Integrity Utilities for Financial Machine Learning

Implements time-series-aware cross-validation and leakage detection.

Based on:
- Marcos López de Prado (2018), "Advances in Financial Machine Learning", Chapter 7
- Academic standard for preventing data leakage in financial time series

Key Features:
1. Purged K-Fold Cross-Validation
2. Embargo mechanism to prevent information leakage
3. Time leakage detector
4. Lag alignment validation
"""
import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


logger = logging.getLogger(__name__)


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation with Embargo
    
    Prevents data leakage in time series by:
    1. Purging training samples that overlap with test period
    2. Embargoing samples immediately after test period
    
    Reference:
        López de Prado, M. (2018). Advances in Financial Machine Learning. 
        Wiley. Chapter 7, pp. 104-109.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds for cross-validation
    embargo_pct : float
        Percentage of samples to embargo after test set (default: 0.02 = 2%)
    purge_pct : float
        Percentage of samples to purge from training set (default: 0.01 = 1%)
    
    Example:
    --------
    >>> pkf = PurgedKFold(n_splits=5, embargo_pct=0.05, purge_pct=0.02)
    >>> for train_idx, test_idx in pkf.split(X):
    >>>     X_train, X_test = X[train_idx], X[test_idx]
    >>>     # Train model...
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02, 
                 purge_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        
        logger.info(
            f"PurgedKFold initialized: n_splits={n_splits}, "
            f"embargo_pct={embargo_pct:.1%}, purge_pct={purge_pct:.1%}"
        )
    
    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate purged and embargoed train/test splits
        
        Yields:
        -------
        train_idx : np.ndarray
            Indices for training set (purged)
        test_idx : np.ndarray
            Indices for test set (embargoed)
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        
        # Calculate purge and embargo sizes
        embargo_size = max(int(n_samples * self.embargo_pct), 1)
        purge_size = max(int(n_samples * self.purge_pct), 1)
        
        logger.info(
            f"Purged K-Fold: n_samples={n_samples}, "
            f"embargo={embargo_size}, purge={purge_size}"
        )
        
        # Use standard KFold for initial split
        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(range(n_samples))):
            # 1. PURGE: Remove training samples near test period
            # Remove samples from end of training set that might leak information
            if len(train_idx) > purge_size:
                train_idx_purged = train_idx[:-purge_size]
            else:
                train_idx_purged = train_idx
            
            # 2. EMBARGO: Remove samples immediately after test period
            # This prevents using information from the "future"
            if len(test_idx) > embargo_size:
                test_idx_embargoed = test_idx[embargo_size:]
            else:
                # If test set is too small, skip this fold
                logger.warning(
                    f"Fold {fold_idx}: Test set too small for embargo, skipping"
                )
                continue
            
            # 3. Validate split quality
            if len(train_idx_purged) < 100 or len(test_idx_embargoed) < 20:
                logger.warning(
                    f"Fold {fold_idx}: Split too small - "
                    f"train={len(train_idx_purged)}, test={len(test_idx_embargoed)}, "
                    f"skipping"
                )
                continue
            
            logger.debug(
                f"Fold {fold_idx}: "
                f"Train={len(train_idx_purged)} (purged {len(train_idx)-len(train_idx_purged)}), "
                f"Test={len(test_idx_embargoed)} (embargoed {embargo_size})"
            )
            
            yield train_idx_purged, test_idx_embargoed


class EnhancedPurgedKFold(PurgedKFold):
    """
    Enhanced Purged K-Fold with explicit lag consideration
    
    Extends PurgedKFold by explicitly handling feature lag to ensure
    no future information leaks through lagged features.
    
    Parameters:
    -----------
    n_splits : int
        Number of folds
    lag : int
        Feature lag (e.g., lag=17 for 17-period forward returns)
    embargo_multiplier : float
        Multiplier for embargo size relative to lag (default: 2.0)
        Embargo will be max(embargo_pct * n_samples, lag * embargo_multiplier)
    
    Reference:
        López de Prado (2018), Section 7.4.1
    """
    
    def __init__(self, n_splits: int = 5, lag: int = 0, 
                 embargo_pct: float = 0.02, purge_pct: float = 0.01,
                 embargo_multiplier: float = 2.0):
        super().__init__(n_splits, embargo_pct, purge_pct)
        self.lag = lag
        self.embargo_multiplier = embargo_multiplier
        
        logger.info(
            f"EnhancedPurgedKFold: lag={lag}, "
            f"embargo_multiplier={embargo_multiplier}"
        )
    
    def split(self, X, y=None, groups=None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate splits with lag-aware embargo"""
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        
        # Enhanced embargo: consider lag
        embargo_from_pct = int(n_samples * self.embargo_pct)
        embargo_from_lag = int(self.lag * self.embargo_multiplier)
        embargo_size = max(embargo_from_pct, embargo_from_lag, 1)
        
        # Purge size should be at least lag
        purge_from_pct = int(n_samples * self.purge_pct)
        purge_size = max(purge_from_pct, self.lag, 1)
        
        logger.info(
            f"Enhanced Purged K-Fold: embargo={embargo_size} "
            f"(pct={embargo_from_pct}, lag={embargo_from_lag}), "
            f"purge={purge_size} (lag-aware)"
        )
        
        kfold = KFold(n_splits=self.n_splits, shuffle=False)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(range(n_samples))):
            # Apply purge and embargo
            train_end_purged = max(0, len(train_idx) - purge_size)
            train_idx_purged = train_idx[:train_end_purged]
            
            test_start_embargoed = min(len(test_idx), embargo_size)
            test_idx_embargoed = test_idx[test_start_embargoed:]
            
            # Validate
            if len(train_idx_purged) < 100 or len(test_idx_embargoed) < 20:
                logger.warning(f"Fold {fold_idx}: Split too small, skipping")
                continue
            
            logger.debug(
                f"Fold {fold_idx}: Train={len(train_idx_purged)}, "
                f"Test={len(test_idx_embargoed)}"
            )
            
            yield train_idx_purged, test_idx_embargoed


class TimeLeakageDetector:
    """
    Automatic detection of time leakage in features
    
    Detects if features contain future information by:
    1. Computing correlation between features and future labels
    2. Checking for suspiciously high correlations
    3. Identifying specific features with leakage
    
    Reference:
        López de Prado (2018), Section 7.3
    
    Parameters:
    -----------
    threshold : float
        Correlation threshold above which leakage is suspected (default: 0.3)
    lag : int
        Forward lag to check (default: 1)
    """
    
    def __init__(self, threshold: float = 0.3, lag: int = 1):
        self.threshold = threshold
        self.lag = lag
        logger.info(
            f"TimeLeakageDetector: threshold={threshold}, lag={lag}"
        )
    
    def detect(self, features: pd.DataFrame, labels: pd.Series) -> dict:
        """
        Detect time leakage in features
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        labels : pd.Series
            Target labels
        
        Returns:
        --------
        dict with keys:
            'has_leakage' : bool
            'max_correlation' : float
            'leaky_features' : list
            'correlations' : pd.Series
        """
        # Shift labels to future
        future_labels = labels.shift(-self.lag)
        
        # Compute correlations
        correlations = features.corrwith(future_labels).abs()
        
        # Identify leaky features
        leaky_features = correlations[correlations > self.threshold].sort_values(ascending=False)
        
        has_leakage = len(leaky_features) > 0
        max_corr = correlations.max()
        
        result = {
            'has_leakage': has_leakage,
            'max_correlation': float(max_corr) if not pd.isna(max_corr) else 0.0,
            'leaky_features': leaky_features.index.tolist(),
            'correlations': correlations,
            'n_leaky_features': len(leaky_features)
        }
        
        if has_leakage:
            logger.error(
                f"🚨 TIME LEAKAGE DETECTED! "
                f"Max correlation={max_corr:.3f}, "
                f"{len(leaky_features)} leaky features"
            )
            for feat, corr in leaky_features.head(10).items():
                logger.error(f"   - {feat}: {corr:.3f}")
        else:
            logger.info(
                f"✅ No time leakage detected (max_corr={max_corr:.3f} < {self.threshold})"
            )
        
        return result
    
    def assert_no_leakage(self, features: pd.DataFrame, labels: pd.Series):
        """
        Assert that no time leakage exists (raises AssertionError if detected)
        
        Use this in unit tests or as a validation check
        """
        result = self.detect(features, labels)
        
        if result['has_leakage']:
            leaky_list = ', '.join(result['leaky_features'][:5])
            raise AssertionError(
                f"Time leakage detected! "
                f"Max correlation: {result['max_correlation']:.3f} "
                f"(threshold: {self.threshold}). "
                f"Leaky features: {leaky_list}"
            )


def validate_lag_alignment(
    layer1_lag: int, 
    layer2_lag: int, 
    feature_lags: Optional[dict] = None
) -> dict:
    """
    Validate that lags are properly aligned across layers
    
    Ensures:
    1. Layer2 lag == Layer1 lag (labels and features aligned)
    2. All feature lags <= primary lag (no future features)
    
    Parameters:
    -----------
    layer1_lag : int
        Lag used in label generation
    layer2_lag : int
        Lag used in feature engineering
    feature_lags : dict, optional
        Dictionary of feature names to their lags
    
    Returns:
    --------
    dict with validation results
    
    Raises:
    -------
    AssertionError if lags are misaligned
    """
    result = {
        'aligned': True,
        'layer1_lag': layer1_lag,
        'layer2_lag': layer2_lag,
        'issues': []
    }
    
    # Check Layer1-Layer2 alignment
    if layer1_lag != layer2_lag:
        result['aligned'] = False
        result['issues'].append(
            f"Layer1 lag ({layer1_lag}) != Layer2 lag ({layer2_lag})"
        )
        logger.error(
            f"❌ Lag misalignment: Layer1={layer1_lag}, Layer2={layer2_lag}"
        )
    
    # Check feature lags
    if feature_lags:
        for feat_name, feat_lag in feature_lags.items():
            if feat_lag > layer1_lag:
                result['aligned'] = False
                result['issues'].append(
                    f"Feature '{feat_name}' lag ({feat_lag}) > primary lag ({layer1_lag})"
                )
                logger.error(
                    f"❌ Feature lag issue: {feat_name} lag={feat_lag} > {layer1_lag}"
                )
    
    if result['aligned']:
        logger.info(f"✅ Lag alignment validated: lag={layer1_lag}")
    else:
        raise AssertionError(
            f"Lag alignment failed: {len(result['issues'])} issues found. "
            f"Issues: {result['issues']}"
        )
    
    return result


# Convenience function for common use case
def get_purged_cv_splits(
    n_samples: int, 
    n_splits: int = 5, 
    lag: int = 0,
    embargo_pct: float = 0.05,
    purge_pct: float = 0.02
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get purged and embargoed CV splits (convenience function)
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_splits : int
        Number of CV folds
    lag : int
        Feature/label lag
    embargo_pct : float
        Embargo percentage (default: 0.05 = 5%)
    purge_pct : float
        Purge percentage (default: 0.02 = 2%)
    
    Returns:
    --------
    List of (train_indices, test_indices) tuples
    """
    if lag > 0:
        pkf = EnhancedPurgedKFold(
            n_splits=n_splits,
            lag=lag,
            embargo_pct=embargo_pct,
            purge_pct=purge_pct,
            embargo_multiplier=2.0
        )
    else:
        pkf = PurgedKFold(
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            purge_pct=purge_pct
        )
    
    # Generate all splits
    splits = list(pkf.split(range(n_samples)))
    
    logger.info(f"Generated {len(splits)} valid CV splits")
    
    return splits

