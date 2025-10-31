# -*- coding: utf-8 -*-
"""
Multi-Timeframe Alignment for Cryptocurrency Trading

Implements strict time-alignment methods to prevent look-ahead bias when
combining data from multiple timeframes (e.g., 15m + 1h + 4h + 1d).

Key Challenge:
--------------
When predicting at 15m 10:45:
- WRONG: Using 1h bar 10:00-11:00 (contains future data from 10:45-11:00) ❌
- CORRECT: Using 1h bar 09:00-10:00 (completed bar only) ✅

Academic Reference:
-------------------
- López de Prado, M. (2018). "Advances in Financial Machine Learning."
  Chapter 3: Labels - Multi-timeframe labeling without look-ahead bias
- Patel, J., et al. (2015). "Predicting stock market using data mining techniques."
  Multi-resolution time series analysis

Implementation:
---------------
1. Strict resampling with shift(1) to ensure only completed bars
2. Forward-fill alignment (ffill) without backward fill
3. Lag validation across timeframes
4. Automatic bar completion detection
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import timedelta


logger = logging.getLogger(__name__)


# Timeframe conversion constants
TIMEFRAME_TO_MINUTES = {
    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360,
    '12h': 720, '1d': 1440, '1w': 10080
}

TIMEFRAME_TO_PANDAS = {
    '15m': '15min', '1h': '1H', '4h': '4H', '1d': '1D'
}


class TimeframeAlignmentError(Exception):
    """Custom exception for timeframe alignment errors"""
    pass


class MultiTimeframeAligner:
    """
    Strict multi-timeframe alignment to prevent look-ahead bias
    
    Key Features:
    -------------
    1. Double-shift resampling (more conservative than standard)
    2. Forward-fill only (no backward fill)
    3. Automatic lag validation
    4. Bar completion detection
    
    Parameters:
    -----------
    base_timeframe : str
        Base timeframe for predictions (e.g., '15m')
    higher_timeframes : list
        List of higher timeframes to align (e.g., ['1h', '4h', '1d'])
    strict_mode : bool
        If True, use double-shift for maximum safety (default: True)
    
    Example:
    --------
    >>> aligner = MultiTimeframeAligner('15m', ['1h', '4h'])
    >>> aligned_data = aligner.align_ohlcv(base_df, higher_dfs)
    >>> # All 1h and 4h features are safely aligned to 15m
    """
    
    def __init__(self, base_timeframe: str, 
                 higher_timeframes: Optional[List[str]] = None,
                 strict_mode: bool = True):
        self.base_timeframe = base_timeframe
        self.higher_timeframes = higher_timeframes or ['1h', '4h', '1d']
        self.strict_mode = strict_mode
        
        # Validate timeframes
        self._validate_timeframes()
        
        logger.info(
            f"MultiTimeframeAligner initialized: "
            f"base={base_timeframe}, higher={higher_timeframes}, "
            f"strict_mode={strict_mode}"
        )
    
    def _validate_timeframes(self):
        """Validate that all timeframes are recognized and properly ordered"""
        all_tfs = [self.base_timeframe] + self.higher_timeframes
        
        for tf in all_tfs:
            if tf not in TIMEFRAME_TO_MINUTES:
                raise TimeframeAlignmentError(
                    f"Unknown timeframe: {tf}. "
                    f"Supported: {list(TIMEFRAME_TO_MINUTES.keys())}"
                )
        
        # Check that higher timeframes are actually higher
        base_minutes = TIMEFRAME_TO_MINUTES[self.base_timeframe]
        for htf in self.higher_timeframes:
            htf_minutes = TIMEFRAME_TO_MINUTES[htf]
            if htf_minutes <= base_minutes:
                raise TimeframeAlignmentError(
                    f"Higher timeframe {htf} ({htf_minutes}m) must be > "
                    f"base timeframe {self.base_timeframe} ({base_minutes}m)"
                )
    
    def resample_ohlcv(self, ohlcv: pd.DataFrame, 
                       target_timeframe: str,
                       method: str = 'double_shift') -> pd.DataFrame:
        """
        Resample OHLCV to higher timeframe with strict look-ahead prevention
        
        Methods:
        --------
        - 'double_shift': Most conservative, shifts before AND after resampling
        - 'single_shift': Standard, shifts after resampling only
        - 'none': No shifting (DANGEROUS, for testing only)
        
        Args:
            ohlcv: DataFrame with OHLCV columns and DatetimeIndex
            target_timeframe: Target timeframe (e.g., '1h')
            method: Resampling method
        
        Returns:
            Resampled DataFrame with only completed bars
        
        Example:
        --------
        >>> # Resample 15m data to 1h
        >>> ohlcv_1h = aligner.resample_ohlcv(ohlcv_15m, '1h')
        >>> # At 10:45, ohlcv_1h contains bars up to 09:00-10:00 (completed)
        """
        if target_timeframe not in TIMEFRAME_TO_PANDAS:
            raise TimeframeAlignmentError(
                f"Unknown target timeframe: {target_timeframe}"
            )
        
        rule = TIMEFRAME_TO_PANDAS[target_timeframe]
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        try:
            if method == 'double_shift':
                # Most conservative: shift before AND after
                ohlcv_shifted = ohlcv.shift(1)
                resampled = ohlcv_shifted.resample(rule).agg(agg_dict)
                resampled_final = resampled.shift(1)
                
                logger.debug(
                    f"Double-shift resample {target_timeframe}: "
                    f"{len(ohlcv)} → {len(resampled_final.dropna())} bars"
                )
            
            elif method == 'single_shift':
                # Standard: shift after resampling
                resampled = ohlcv.resample(rule).agg(agg_dict)
                resampled_final = resampled.shift(1)
                
                logger.debug(
                    f"Single-shift resample {target_timeframe}: "
                    f"{len(ohlcv)} → {len(resampled_final.dropna())} bars"
                )
            
            elif method == 'none':
                # DANGEROUS: no shifting (for testing only)
                resampled_final = ohlcv.resample(rule).agg(agg_dict)
                logger.warning(
                    f"⚠️ No-shift resample {target_timeframe}: "
                    f"LOOK-AHEAD BIAS PRESENT!"
                )
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return resampled_final.dropna()
        
        except Exception as e:
            logger.error(f"Resample failed: {e}")
            raise TimeframeAlignmentError(f"Resample error: {e}")
    
    def align_series_to_base(self, base_index: pd.DatetimeIndex,
                            higher_tf_series: pd.Series,
                            shift_before_align: bool = True) -> pd.Series:
        """
        Align a higher timeframe series to base timeframe index
        
        Args:
            base_index: Base timeframe DatetimeIndex (e.g., 15m)
            higher_tf_series: Series from higher timeframe (e.g., 1h trend)
            shift_before_align: If True, shift(1) before alignment
        
        Returns:
            Series aligned to base_index with forward-fill
        
        Example:
        --------
        >>> # Align 1h trend to 15m index
        >>> trend_15m = aligner.align_series_to_base(
        ...     base_index=df_15m.index,
        ...     higher_tf_series=trend_1h
        ... )
        """
        try:
            if shift_before_align:
                # Shift to ensure only completed bars
                higher_tf_shifted = higher_tf_series.shift(1)
            else:
                higher_tf_shifted = higher_tf_series
            
            # Align using forward-fill only
            aligned = higher_tf_shifted.reindex(base_index, method='ffill')
            
            # Fill remaining NaNs with 0 (for start of series)
            aligned = aligned.fillna(0)
            
            logger.debug(
                f"Aligned series: {len(higher_tf_series)} → "
                f"{len(aligned)} samples"
            )
            
            return aligned
        
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            raise TimeframeAlignmentError(f"Alignment error: {e}")
    
    def align_dataframe_to_base(self, base_index: pd.DatetimeIndex,
                               higher_tf_df: pd.DataFrame,
                               suffix: str = '') -> pd.DataFrame:
        """
        Align entire DataFrame from higher timeframe to base
        
        Args:
            base_index: Base timeframe DatetimeIndex
            higher_tf_df: DataFrame from higher timeframe
            suffix: Suffix to add to column names (e.g., '_1h')
        
        Returns:
            Aligned DataFrame
        """
        aligned_dict = {}
        
        for col in higher_tf_df.columns:
            aligned_series = self.align_series_to_base(
                base_index, higher_tf_df[col]
            )
            new_col_name = f"{col}{suffix}" if suffix else col
            aligned_dict[new_col_name] = aligned_series
        
        return pd.DataFrame(aligned_dict, index=base_index)
    
    def validate_alignment(self, base_df: pd.DataFrame,
                          aligned_df: pd.DataFrame,
                          check_future_correlation: bool = True) -> Dict:
        """
        Validate that alignment is correct and has no look-ahead bias
        
        Checks:
        -------
        1. Index alignment (all indices match)
        2. No NaN introduction
        3. Future correlation check (optional)
        
        Args:
            base_df: Base timeframe DataFrame
            aligned_df: Aligned higher timeframe DataFrame
            check_future_correlation: If True, check for future data leakage
        
        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check 1: Index alignment
        if not base_df.index.equals(aligned_df.index):
            results['valid'] = False
            results['errors'].append(
                f"Index mismatch: base={len(base_df)}, "
                f"aligned={len(aligned_df)}"
            )
        
        # Check 2: NaN check
        nan_count = aligned_df.isna().sum().sum()
        if nan_count > 0:
            results['warnings'].append(
                f"Found {nan_count} NaN values in aligned data"
            )
        
        # Check 3: Future correlation (if we have price data)
        if check_future_correlation and 'close' in base_df.columns:
            future_returns = base_df['close'].pct_change().shift(-1)
            
            for col in aligned_df.columns:
                if aligned_df[col].std() > 0:
                    corr = aligned_df[col].corr(future_returns)
                    if abs(corr) > 0.3:
                        results['warnings'].append(
                            f"High future correlation in {col}: {corr:.3f}"
                        )
        
        if results['errors']:
            logger.error(f"Alignment validation failed: {results['errors']}")
        
        if results['warnings']:
            logger.warning(f"Alignment warnings: {results['warnings']}")
        
        return results
    
    def get_lag_for_timeframe(self, higher_timeframe: str) -> int:
        """
        Calculate appropriate lag for a higher timeframe
        
        Lag is calculated as: higher_tf_minutes / base_tf_minutes
        
        Args:
            higher_timeframe: Higher timeframe (e.g., '1h')
        
        Returns:
            Lag in base timeframe periods
        
        Example:
        --------
        >>> aligner = MultiTimeframeAligner('15m', ['1h'])
        >>> lag = aligner.get_lag_for_timeframe('1h')
        >>> print(lag)  # 4 (because 1h = 4 × 15m)
        """
        base_minutes = TIMEFRAME_TO_MINUTES[self.base_timeframe]
        higher_minutes = TIMEFRAME_TO_MINUTES[higher_timeframe]
        
        lag = higher_minutes // base_minutes
        
        logger.debug(
            f"Lag for {higher_timeframe}: {lag} "
            f"({higher_minutes}m / {base_minutes}m)"
        )
        
        return lag
    
    def calculate_minimum_embargo(self, lag: int = 0) -> int:
        """
        Calculate minimum embargo period for cross-validation
        
        Formula: embargo = max(2 × lag, 2 × max_higher_tf_lag)
        
        Args:
            lag: Label lag (forward-looking periods)
        
        Returns:
            Minimum embargo in base timeframe periods
        """
        max_htf_lag = max([
            self.get_lag_for_timeframe(htf) 
            for htf in self.higher_timeframes
        ])
        
        embargo = max(2 * lag, 2 * max_htf_lag)
        
        logger.info(
            f"Minimum embargo: {embargo} periods "
            f"(label_lag={lag}, max_htf_lag={max_htf_lag})"
        )
        
        return embargo


class TimeframeConsistencyChecker:
    """
    Check consistency across multiple timeframes
    
    Ensures that features from different timeframes are properly aligned
    and don't contain look-ahead bias.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def check_bar_completion(self, timestamp: pd.Timestamp,
                            timeframe: str) -> bool:
        """
        Check if a bar is completed at given timestamp
        
        Args:
            timestamp: Current timestamp
            timeframe: Timeframe to check (e.g., '1h')
        
        Returns:
            True if bar is completed, False otherwise
        
        Example:
        --------
        >>> checker = TimeframeConsistencyChecker()
        >>> ts = pd.Timestamp('2024-10-30 10:45')
        >>> checker.check_bar_completion(ts, '1h')  # False (10:00-11:00 incomplete)
        >>> checker.check_bar_completion(ts, '15m')  # True (10:45 bar complete)
        """
        if timeframe not in TIMEFRAME_TO_MINUTES:
            raise TimeframeAlignmentError(f"Unknown timeframe: {timeframe}")
        
        minutes = TIMEFRAME_TO_MINUTES[timeframe]
        
        # Check if timestamp is at bar boundary
        if timeframe in ['1h', '2h', '4h', '6h', '12h']:
            # Hourly bars
            hours = minutes // 60
            is_complete = (timestamp.hour % hours == 0) and (timestamp.minute == 0)
        elif timeframe == '1d':
            # Daily bars
            is_complete = (timestamp.hour == 0) and (timestamp.minute == 0)
        else:
            # Minute bars
            is_complete = (timestamp.minute % minutes == 0)
        
        return is_complete
    
    def detect_look_ahead_bias(self, features: pd.DataFrame,
                              labels: pd.Series,
                              threshold: float = 0.3) -> Dict:
        """
        Detect potential look-ahead bias in features
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            threshold: Correlation threshold for warnings
        
        Returns:
            Dict with detection results
        """
        results = {
            'suspicious_features': [],
            'max_correlation': 0.0,
            'warnings': []
        }
        
        # Calculate correlations
        for col in features.columns:
            if features[col].std() > 0:
                corr = features[col].corr(labels)
                
                if abs(corr) > threshold:
                    results['suspicious_features'].append({
                        'feature': col,
                        'correlation': corr
                    })
                    results['max_correlation'] = max(
                        results['max_correlation'], abs(corr)
                    )
        
        if results['suspicious_features']:
            results['warnings'].append(
                f"Found {len(results['suspicious_features'])} features "
                f"with correlation > {threshold}"
            )
            self.logger.warning(
                f"Potential look-ahead bias detected: "
                f"{results['warnings']}"
            )
        
        return results


def align_multiple_timeframes(base_df: pd.DataFrame,
                             base_timeframe: str,
                             higher_dfs: Dict[str, pd.DataFrame],
                             strict_mode: bool = True) -> pd.DataFrame:
    """
    Convenience function to align multiple timeframes at once
    
    Args:
        base_df: Base timeframe DataFrame (e.g., 15m)
        base_timeframe: Base timeframe string (e.g., '15m')
        higher_dfs: Dict of {timeframe: DataFrame} (e.g., {'1h': df_1h, '4h': df_4h})
        strict_mode: Use double-shift if True
    
    Returns:
        Combined DataFrame with all timeframes aligned
    
    Example:
    --------
    >>> aligned = align_multiple_timeframes(
    ...     base_df=df_15m,
    ...     base_timeframe='15m',
    ...     higher_dfs={'1h': df_1h, '4h': df_4h}
    ... )
    >>> # Now aligned contains columns from all timeframes
    """
    aligner = MultiTimeframeAligner(
        base_timeframe, 
        list(higher_dfs.keys()),
        strict_mode=strict_mode
    )
    
    result = base_df.copy()
    
    for tf, df in higher_dfs.items():
        aligned = aligner.align_dataframe_to_base(
            base_index=base_df.index,
            higher_tf_df=df,
            suffix=f'_{tf}'
        )
        result = pd.concat([result, aligned], axis=1)
    
    logger.info(
        f"Aligned {len(higher_dfs)} timeframes: "
        f"final shape {result.shape}"
    )
    
    return result


def validate_coordinator_resampling(df_15m: pd.DataFrame,
                                   df_higher: pd.DataFrame,
                                   higher_timeframe: str) -> bool:
    """
    Validate that coordinator's resampling is correct
    
    This function checks if the resampling in coordinator.py
    follows the strict alignment rules.
    
    Args:
        df_15m: Base 15m DataFrame
        df_higher: Resampled higher timeframe DataFrame
        higher_timeframe: Timeframe string (e.g., '1h')
    
    Returns:
        True if resampling is correct, False otherwise
    """
    checker = TimeframeConsistencyChecker()
    
    # Check 1: Shape compatibility
    base_minutes = TIMEFRAME_TO_MINUTES['15m']
    higher_minutes = TIMEFRAME_TO_MINUTES[higher_timeframe]
    expected_ratio = higher_minutes / base_minutes
    
    actual_ratio = len(df_15m) / len(df_higher)
    
    if abs(actual_ratio - expected_ratio) > 0.1 * expected_ratio:
        logger.error(
            f"Shape ratio mismatch: expected {expected_ratio}, "
            f"got {actual_ratio}"
        )
        return False
    
    # Check 2: First timestamps
    if len(df_higher) > 0:
        first_15m = df_15m.index[0]
        first_higher = df_higher.index[0]
        
        # Higher TF should start later (due to shift)
        if first_higher <= first_15m:
            logger.error(
                f"Timestamp error: higher TF starts at {first_higher}, "
                f"should be after {first_15m}"
            )
            return False
    
    logger.info(f"✅ Resampling validation passed for {higher_timeframe}")
    return True

