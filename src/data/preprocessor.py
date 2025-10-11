"""
Data Preprocessor Module
Handles data cleaning, transformation, and preparation for ML models
Includes missing value handling, outlier detection, and data validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator

logger = setup_logger(__name__)

class DataPreprocessor:
    """Advanced data preprocessing and cleaning"""

    def __init__(self):
        # Preprocessing configuration
        self.outlier_methods = {
            'iqr': self._detect_outliers_iqr,
            'zscore': self._detect_outliers_zscore,
            'modified_zscore': self._detect_outliers_modified_zscore,
            'isolation_forest': self._detect_outliers_isolation_forest
        }

        self.imputation_methods = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'mode': SimpleImputer(strategy='most_frequent'),
            'knn': KNNImputer(n_neighbors=5),
            'forward_fill': None,  # Custom implementation
            'backward_fill': None,  # Custom implementation
            'linear_interpolate': None  # Custom implementation
        }

        self.scaling_methods = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        # Preprocessing history for tracking
        self.preprocessing_history = []

        logger.info("Data preprocessor initialized")

    @timing_decorator

    def preprocess_ohlcv_data(self, df: pd.DataFrame,
                            cleaning_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Comprehensive OHLCV data preprocessing

        Args:
            df: Raw OHLCV DataFrame
            cleaning_config: Configuration for cleaning parameters

        Returns:
            Cleaned and preprocessed DataFrame
        """
        try:
            if df.empty:
                logger.warning("Empty DataFrame provided for preprocessing")
                return df

            logger.debug(f"Preprocessing OHLCV data: {len(df)} records")

            # Default configuration
            if cleaning_config is None:
                cleaning_config = {
                    'remove_duplicates': True,
                    'handle_missing_values': True,
                    'detect_outliers': True,
                    'validate_ohlc': True,
                    'remove_zero_volume': False,
                    'price_gap_threshold': 0.1,  # 10% price gap threshold
                    'volume_outlier_threshold': 3,  # Z-score threshold
                    'imputation_method': 'forward_fill',
                    'max_short_gap': 3  # interpolate <=3 bars, drop longer gaps
                }

            processed_df = df.copy()
            processing_log = []

            # 0. Ensure DateTimeIndex in UTC and basic index hygiene
            try:
                # If timestamp column exists, use it as index
                if 'timestamp' in processed_df.columns:
                    processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'], utc=True, errors='coerce')
                    processed_df = processed_df.set_index('timestamp')

                # Ensure index is datetime in UTC
                if not isinstance(processed_df.index, pd.DatetimeIndex):
                    processed_df.index = pd.to_datetime(processed_df.index, utc=True, errors='coerce')
                elif processed_df.index.tz is None:
                    processed_df.index = pd.to_datetime(processed_df.index, utc=True, errors='coerce')

                # Drop NaT index rows if any
                if processed_df.index.hasnans:
                    before = len(processed_df)
                    processed_df = processed_df[~processed_df.index.isna()]
                    removed_nats = before - len(processed_df)
                    if removed_nats > 0:
                        processing_log.append(f"Removed {removed_nats} rows with invalid timestamps")

                # Drop duplicate index keeping the last (latest write wins)
                if processed_df.index.duplicated().any():
                    before = len(processed_df)
                    processed_df = processed_df[~processed_df.index.duplicated(keep='last')]
                    removed_dup_idx = before - len(processed_df)
                    processing_log.append(f"Removed {removed_dup_idx} duplicate index entries (kept last)")
            except Exception:
                pass

            # 1. Remove duplicates
            if cleaning_config.get('remove_duplicates', True):
                initial_len = len(processed_df)
                processed_df = processed_df.drop_duplicates()
                removed_duplicates = initial_len - len(processed_df)
                if removed_duplicates > 0:
                    processing_log.append(f"Removed {removed_duplicates} duplicate records")

            # 2. Sort by index (timestamp)
            if not processed_df.index.is_monotonic_increasing:
                processed_df = processed_df.sort_index()
                processing_log.append("Sorted data by timestamp")

            # 3. Align to uniform time grid and interpolate short gaps
            try:
                processed_df = self._reindex_and_fill_short_gaps(
                    processed_df,
                    max_gap=cleaning_config.get('max_short_gap', 3)
                )
                processing_log.append("Aligned to uniform time grid; interpolated short gaps; kept volume=0 for inserted bars")
            except Exception:
                # best-effort; continue
                pass

            # 4. Validate OHLC relationships
            if cleaning_config.get('validate_ohlc', True):
                processed_df, ohlc_fixes = self._validate_ohlc_relationships(processed_df)
                if ohlc_fixes > 0:
                    processing_log.append(f"Fixed {ohlc_fixes} OHLC relationship errors")

            # 5. Handle missing values (fallback safety net)
            if cleaning_config.get('handle_missing_values', True):
                processed_df, missing_handled = self._handle_missing_values(
                    processed_df, cleaning_config.get('imputation_method', 'forward_fill')
                )
                if missing_handled > 0:
                    processing_log.append(f"Handled {missing_handled} missing values")

            # 6. Median smoothing for volume to reduce noise (window=5)
            try:
                if 'volume' in processed_df.columns:
                    processed_df['volume'] = pd.to_numeric(processed_df['volume'], errors='coerce')
                    processed_df['volume'] = processed_df['volume'].rolling(window=5, center=True, min_periods=1).median()
                    processed_df['volume'] = processed_df['volume'].fillna(0)
                    processing_log.append("Applied median smoothing to volume (window=5)")
            except Exception:
                pass

            # 7. Optionally remove zero volume bars (disabled by default)
            if cleaning_config.get('remove_zero_volume', False):
                initial_len = len(processed_df)
                processed_df = processed_df[processed_df['volume'] > 0]
                removed_zero_vol = initial_len - len(processed_df)
                if removed_zero_vol > 0:
                    processing_log.append(f"Removed {removed_zero_vol} zero-volume bars")

            # 8. Detect and handle price gaps
            price_gap_threshold = cleaning_config.get('price_gap_threshold', 0.1)
            processed_df, gaps_handled = self._handle_price_gaps(processed_df, price_gap_threshold)
            if gaps_handled > 0:
                processing_log.append(f"Handled {gaps_handled} price gaps")

            # 9. Detect outliers
            if cleaning_config.get('detect_outliers', True):
                outlier_results = self._detect_and_handle_outliers(processed_df, cleaning_config)
                if outlier_results['outliers_handled'] > 0:
                    processing_log.append(f"Handled {outlier_results['outliers_handled']} outliers")

            # 10. Final validation
            processed_df = self._final_validation(processed_df)

            # Log preprocessing summary
            self._log_preprocessing_summary(df, processed_df, processing_log)

            logger.debug(f"Preprocessing completed: {len(processed_df)} records remaining")
            return processed_df

        except Exception as e:
            logger.error(f"OHLCV preprocessing failed: {e}")
            return df

    def _reindex_and_fill_short_gaps(self, df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
        """Reindex to a uniform time grid and interpolate only short gaps.

        - Determine expected frequency from index diffs (mode, then median fallback)
        - Reindex to full range
        - Interpolate OHLC for gaps of length <= max_gap using time interpolation
        - Keep volume=0 for inserted gaps
        - Drop rows belonging to long gaps (> max_gap) to avoid misleading interpolation
        """
        try:
            if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
                return df

            # Ensure UTC tz-aware
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index, utc=True, errors='coerce')

            # Determine expected frequency
            diffs = df.index.to_series().diff().dropna()
            if diffs.empty:
                return df

            try:
                expected_gap = diffs.mode().iloc[0]
            except Exception:
                expected_gap = diffs.median()

            if pd.isna(expected_gap) or expected_gap <= pd.Timedelta(0):
                return df

            full_index = pd.date_range(df.index.min(), df.index.max(), freq=expected_gap)
            reindexed = df.reindex(full_index)

            price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in reindexed.columns]
            # Interpolate price columns across time
            if price_cols:
                reindexed[price_cols] = reindexed[price_cols].interpolate(method='time', limit=max_gap)

            # Volume: fill missing as 0 for inserted bars
            if 'volume' in reindexed.columns:
                reindexed['volume'] = reindexed['volume'].fillna(0)

            # Drop rows that still have NaN in price columns (long gaps)
            if price_cols:
                reindexed = reindexed.dropna(subset=price_cols, how='any')

            return reindexed

        except Exception:
            return df

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Validate and fix OHLC relationships"""
        try:
            fixes_count = 0
            processed_df = df.copy()

            # Check for each row: low <= open,close <= high
            for idx in processed_df.index:
                row = processed_df.loc[idx]

                # Fix cases where high < low (swap them)
                if row['high'] < row['low']:
                    processed_df.loc[idx, 'high'], processed_df.loc[idx, 'low'] = row['low'], row['high']
                    fixes_count += 1

                # Fix cases where open/close outside high/low range
                if row['open'] > row['high']:
                    processed_df.loc[idx, 'high'] = row['open']
                    fixes_count += 1
                elif row['open'] < row['low']:
                    processed_df.loc[idx, 'low'] = row['open']
                    fixes_count += 1

                if row['close'] > row['high']:
                    processed_df.loc[idx, 'high'] = row['close']
                    fixes_count += 1
                elif row['close'] < row['low']:
                    processed_df.loc[idx, 'low'] = row['close']
                    fixes_count += 1

            return processed_df, fixes_count

        except Exception as e:
            logger.error(f"OHLC validation failed: {e}")
            return df, 0

    def _handle_missing_values(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, int]:
        """Handle missing values using specified method"""
        try:
            processed_df = df.copy()
            initial_missing = processed_df.isnull().sum().sum()

            if initial_missing == 0:
                return processed_df, 0

            if method == 'forward_fill':
                processed_df = processed_df.fillna(method='ffill')
            elif method == 'backward_fill':
                # 🔧 禁用後向填充以防止數據洩漏
                print("⚠️ backward_fill已禁用以防止數據洩漏，改用forward_fill")
                processed_df = processed_df.fillna(method='ffill')
            elif method == 'linear_interpolate':
                processed_df = processed_df.interpolate(method='linear')
            elif method in self.imputation_methods:
                imputer = self.imputation_methods[method]
                if imputer is not None:
                    numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
                    processed_df[numeric_columns] = imputer.fit_transform(processed_df[numeric_columns])
            else:
                logger.warning(f"Unknown imputation method: {method}, using forward fill")
                processed_df = processed_df.fillna(method='ffill')

            # 🔧 修復數據洩漏：禁用後向填充
            # processed_df = processed_df.fillna(method='bfill')  # 已禁用
            processed_df = processed_df.fillna(0)  # 用0填充剩餘NaN值

            final_missing = processed_df.isnull().sum().sum()
            handled_missing = initial_missing - final_missing

            return processed_df, handled_missing

        except Exception as e:
            logger.error(f"Missing value handling failed: {e}")
            return df, 0

    def _handle_price_gaps(self, df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, int]:
        """Detect and handle large price gaps"""
        try:
            processed_df = df.copy()
            gaps_handled = 0

            # Calculate price changes
            price_changes = processed_df['close'].pct_change()

            # Identify large gaps
            large_gaps = abs(price_changes) > threshold
            gap_indices = large_gaps[large_gaps].index

            for idx in gap_indices:
                try:
                    # Get previous and current prices
                    prev_idx = processed_df.index.get_loc(idx) - 1
                    if prev_idx < 0:
                        continue

                    prev_close = processed_df.iloc[prev_idx]['close']
                    current_row = processed_df.loc[idx]

                    # Adjust open to previous close for gap handling
                    processed_df.loc[idx, 'open'] = prev_close

                    # Ensure OHLC consistency after adjustment
                    processed_df.loc[idx, 'high'] = max(current_row['high'], prev_close)
                    processed_df.loc[idx, 'low'] = min(current_row['low'], prev_close)

                    gaps_handled += 1

                except Exception:
                    continue

            return processed_df, gaps_handled

        except Exception as e:
            logger.error(f"Price gap handling failed: {e}")
            return df, 0

    def _detect_and_handle_outliers(self, df: pd.DataFrame,
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and handle outliers in the data"""
        try:
            outlier_method = config.get('outlier_method', 'iqr')
            volume_threshold = config.get('volume_outlier_threshold', 3)

            results = {
                'price_outliers': 0,
                'volume_outliers': 0,
                'outliers_handled': 0
            }

            processed_df = df.copy()

            # Detect price outliers (using returns)
            returns = processed_df['close'].pct_change().dropna()

            if outlier_method in self.outlier_methods:
                price_outliers = self.outlier_methods[outlier_method](returns)
                results['price_outliers'] = len(price_outliers)

                # Handle price outliers by winsorizing
                if len(price_outliers) > 0:
                    returns_clean = self._winsorize_outliers(returns, price_outliers)
                    # Note: In practice, you might want to adjust the actual prices
                    # Here we just count them for now

            # Detect volume outliers
            if 'volume' in processed_df.columns:
                log_volumes = np.log1p(processed_df['volume'])
                volume_outliers = self._detect_outliers_zscore(log_volumes, threshold=volume_threshold)
                results['volume_outliers'] = len(volume_outliers)

                # Handle volume outliers by capping
                if len(volume_outliers) > 0:
                    volume_95th = processed_df['volume'].quantile(0.95)
                    outlier_mask = processed_df.index.isin(volume_outliers)
                    processed_df.loc[outlier_mask, 'volume'] = volume_95th
                    results['outliers_handled'] += len(volume_outliers)

            return results

        except Exception as e:
            logger.error(f"Outlier detection/handling failed: {e}")
            return {'price_outliers': 0, 'volume_outliers': 0, 'outliers_handled': 0}

    def _detect_outliers_iqr(self, series: pd.Series, iqr_factor: float = 1.5) -> List[Any]:
        """Detect outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR

            outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
            return outliers

        except Exception as e:
            logger.error(f"IQR outlier detection failed: {e}")
            return []

    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[Any]:
        """Detect outliers using Z-score method"""
        try:
            z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
            outliers = series[z_scores > threshold].index.tolist()
            return outliers

        except Exception as e:
            logger.error(f"Z-score outlier detection failed: {e}")
            return []

    def _detect_outliers_modified_zscore(self, series: pd.Series, threshold: float = 3.5) -> List[Any]:
        """Detect outliers using Modified Z-score method"""
        try:
            median = series.median()
            mad = np.median(np.abs(series - median))

            if mad == 0:
                return []

            modified_z_scores = 0.6745 * (series - median) / mad
            outliers = series[np.abs(modified_z_scores) > threshold].index.tolist()
            return outliers

        except Exception as e:
            logger.error(f"Modified Z-score outlier detection failed: {e}")
            return []

    def _detect_outliers_isolation_forest(self, series: pd.Series,
                                        contamination: float = 0.1) -> List[Any]:
        """Detect outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest

            # Reshape for sklearn
            X = series.values.reshape(-1, 1)

            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers_mask = iso_forest.fit_predict(X) == -1

            outliers = series[outliers_mask].index.tolist()
            return outliers

        except Exception as e:
            logger.error(f"Isolation Forest outlier detection failed: {e}")
            return []

    def _winsorize_outliers(self, series: pd.Series, outlier_indices: List[Any],
                          percentile: float = 0.95) -> pd.Series:
        """Winsorize outliers by capping at percentiles"""
        try:
            series_clean = series.copy()

            # Calculate percentile bounds
            lower_bound = series.quantile(1 - percentile)
            upper_bound = series.quantile(percentile)

            # Cap outliers
            for idx in outlier_indices:
                if series[idx] > upper_bound:
                    series_clean[idx] = upper_bound
                elif series[idx] < lower_bound:
                    series_clean[idx] = lower_bound

            return series_clean

        except Exception as e:
            logger.error(f"Winsorization failed: {e}")
            return series

    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup"""
        try:
            processed_df = df.copy()

            # Ensure all numeric columns are proper numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            # Remove any rows that became all NaN
            processed_df = processed_df.dropna(how='all')

            # Ensure positive prices and volumes
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in processed_df.columns:
                    processed_df = processed_df[processed_df[col] > 0]

            if 'volume' in processed_df.columns:
                processed_df = processed_df[processed_df['volume'] >= 0]

            return processed_df

        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            return df

    def _log_preprocessing_summary(self, original_df: pd.DataFrame,
                                 processed_df: pd.DataFrame,
                                 processing_log: List[str]):
        """Log preprocessing summary"""
        try:
            summary = {
                'original_records': len(original_df),
                'processed_records': len(processed_df),
                'records_removed': len(original_df) - len(processed_df),
                'removal_percentage': ((len(original_df) - len(processed_df)) / len(original_df)) * 100 if len(original_df) > 0 else 0,
                'processing_steps': processing_log,
                'timestamp': datetime.now()
            }

            self.preprocessing_history.append(summary)

            # Keep only recent history
            if len(self.preprocessing_history) > 100:
                self.preprocessing_history = self.preprocessing_history[-100:]

            logger.debug(f"Preprocessing summary: {summary['records_removed']} records removed "
                        f"({summary['removal_percentage']:.1f}%), {len(processing_log)} steps")

        except Exception as e:
            logger.error(f"Preprocessing summary logging failed: {e}")

    @timing_decorator

    def preprocess_features(self, df: pd.DataFrame,
                          preprocessing_config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Preprocess feature data for ML models

        Args:
            df: Features DataFrame
            preprocessing_config: Configuration for preprocessing

        Returns:
            Preprocessed features DataFrame
        """
        try:
            if df.empty:
                return df

            # Default configuration
            if preprocessing_config is None:
                preprocessing_config = {
                    'handle_missing': True,
                    'remove_constant_features': True,
                    'handle_inf_values': True,
                    'scaling_method': None,  # 'standard', 'minmax', 'robust', or None
                    'correlation_threshold': 0.95,  # Remove highly correlated features
                    'variance_threshold': 0.01  # Remove low-variance features
                }

            processed_df = df.copy()
            logger.debug(f"Preprocessing features: {processed_df.shape}")

            # 1. Handle infinite values
            if preprocessing_config.get('handle_inf_values', True):
                processed_df = self._handle_infinite_values(processed_df)

            # 2. Handle missing values
            if preprocessing_config.get('handle_missing', True):
                processed_df = self._handle_feature_missing_values(processed_df)

            # 3. Remove constant features
            if preprocessing_config.get('remove_constant_features', True):
                processed_df = self._remove_constant_features(processed_df)

            # 4. Remove low variance features
            variance_threshold = preprocessing_config.get('variance_threshold')
            if variance_threshold is not None:
                processed_df = self._remove_low_variance_features(processed_df, variance_threshold)

            # 5. Remove highly correlated features
            correlation_threshold = preprocessing_config.get('correlation_threshold')
            if correlation_threshold is not None:
                processed_df = self._remove_correlated_features(processed_df, correlation_threshold)

            # 6. Apply scaling
            scaling_method = preprocessing_config.get('scaling_method')
            if scaling_method and scaling_method in self.scaling_methods:
                processed_df = self._apply_scaling(processed_df, scaling_method)

            logger.debug(f"Feature preprocessing completed: {processed_df.shape}")
            return processed_df

        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}")
            return df

    def _handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite values in the DataFrame"""
        try:
            processed_df = df.copy()

            # Replace inf/-inf with NaN
            processed_df = processed_df.replace([np.inf, -np.inf], np.nan)

            return processed_df

        except Exception as e:
            logger.error(f"Infinite value handling failed: {e}")
            return df

    def _handle_feature_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature data"""
        try:
            processed_df = df.copy()

            # Use median imputation for numeric features
            numeric_columns = processed_df.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if processed_df[col].isnull().any():
                    median_value = processed_df[col].median()
                    processed_df[col] = processed_df[col].fillna(median_value)

            return processed_df

        except Exception as e:
            logger.error(f"Feature missing value handling failed: {e}")
            return df

    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with constant values"""
        try:
            processed_df = df.copy()

            # Identify constant columns
            constant_columns = []
            for col in processed_df.columns:
                if processed_df[col].nunique() <= 1:
                    constant_columns.append(col)

            if constant_columns:
                processed_df = processed_df.drop(columns=constant_columns)
                logger.debug(f"Removed {len(constant_columns)} constant features")

            return processed_df

        except Exception as e:
            logger.error(f"Constant feature removal failed: {e}")
            return df

    def _remove_low_variance_features(self, df: pd.DataFrame,
                                    threshold: float) -> pd.DataFrame:
        """Remove features with low variance"""
        try:
            from sklearn.feature_selection import VarianceThreshold

            # Apply variance threshold
            selector = VarianceThreshold(threshold=threshold)

            # Get numeric columns only
            numeric_df = df.select_dtypes(include=[np.number])

            if numeric_df.empty:
                return df

            # Fit and transform
            selected_features = selector.fit_transform(numeric_df)

            # Get selected column names
            selected_columns = numeric_df.columns[selector.get_support()]

            # Create new DataFrame with selected features
            processed_df = pd.DataFrame(
                selected_features,
                columns=selected_columns,
                index=df.index
            )

            removed_count = len(numeric_df.columns) - len(selected_columns)
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} low-variance features")

            return processed_df

        except Exception as e:
            logger.error(f"Low variance feature removal failed: {e}")
            return df

    def _remove_correlated_features(self, df: pd.DataFrame,
                                  threshold: float) -> pd.DataFrame:
        """Remove highly correlated features"""
        try:
            processed_df = df.copy()

            # Calculate correlation matrix
            corr_matrix = processed_df.corr().abs()

            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to remove
            to_remove = [column for column in upper_triangle.columns
                        if any(upper_triangle[column] > threshold)]

            if to_remove:
                processed_df = processed_df.drop(columns=to_remove)
                logger.debug(f"Removed {len(to_remove)} highly correlated features")

            return processed_df

        except Exception as e:
            logger.error(f"Correlated feature removal failed: {e}")
            return df

    def _apply_scaling(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply feature scaling"""
        try:
            if method not in self.scaling_methods:
                logger.warning(f"Unknown scaling method: {method}")
                return df

            scaler = self.scaling_methods[method]

            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) == 0:
                return df

            # Apply scaling
            scaled_data = scaler.fit_transform(df[numeric_columns])

            # Create scaled DataFrame
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=numeric_columns,
                index=df.index
            )

            # Add back non-numeric columns if any
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_columns) > 0:
                for col in non_numeric_columns:
                    scaled_df[col] = df[col]

            logger.debug(f"Applied {method} scaling to {len(numeric_columns)} features")
            return scaled_df

        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            return df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        try:
            quality_report = {
                'timestamp': datetime.now(),
                'total_records': len(df),
                'total_features': len(df.columns) if not df.empty else 0,
                'missing_values': {},
                'data_types': {},
                'duplicates': 0,
                'outliers_detected': 0,
                'index_gap_stats': {},
                'max_consecutive_missing': 0,
                'quality_score': 0.0,
                'issues': []
            }

            if df.empty:
                quality_report['quality_score'] = 0.0
                quality_report['issues'].append("Empty DataFrame")
                return quality_report

            # Check missing values
            missing_counts = df.isnull().sum()
            quality_report['missing_values'] = missing_counts.to_dict()

            missing_percentage = (missing_counts.sum() / (len(df) * len(df.columns))) * 100

            # Check data types
            quality_report['data_types'] = df.dtypes.astype(str).to_dict()

            # Check duplicates
            quality_report['duplicates'] = df.duplicated().sum()

            # Check for outliers (simple Z-score method)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            outlier_count = 0

            for col in numeric_columns:
                try:
                    z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                    outlier_count += len(z_scores[z_scores > 3])
                except:
                    continue

            quality_report['outliers_detected'] = outlier_count

            # Index gap statistics (for DateTimeIndex)
            if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
                diffs = df.index.to_series().diff().dropna()
                if not diffs.empty:
                    try:
                        expected_gap = diffs.mode().iloc[0]
                    except Exception:
                        expected_gap = diffs.median()

                    gap_multiples = (diffs / expected_gap).round().astype('Int64')
                    # Identify gaps > 1x expected interval
                    large_gaps = gap_multiples[gap_multiples > 1]
                    quality_report['index_gap_stats'] = {
                        'expected_interval': str(expected_gap),
                        'num_large_gaps': int(large_gaps.count()),
                        'max_gap_multiple': int(large_gaps.max()) if not large_gaps.empty else 1
                    }

            # Max consecutive missing across columns
            if df.isnull().values.any():
                # For each column compute max run of NaN, take overall max
                max_runs = []
                for col in df.columns:
                    s = df[col].isna().astype(int)
                    if s.sum() == 0:
                        continue
                    # Compute run lengths
                    runs = (s.groupby((s != s.shift()).cumsum()).cumsum() * s)
                    max_runs.append(int(runs.max()))
                quality_report['max_consecutive_missing'] = max(max_runs) if max_runs else 0

            # Calculate quality score (0-100)
            score_components = []

            # Missing value penalty
            missing_penalty = max(0, 100 - missing_percentage * 2)
            score_components.append(missing_penalty)

            # Duplicate penalty
            duplicate_percentage = (quality_report['duplicates'] / len(df)) * 100
            duplicate_penalty = max(0, 100 - duplicate_percentage * 5)
            score_components.append(duplicate_penalty)

            # Outlier penalty
            outlier_percentage = (outlier_count / (len(df) * len(numeric_columns))) * 100
            outlier_penalty = max(0, 100 - outlier_percentage)
            score_components.append(outlier_penalty)

            quality_report['quality_score'] = np.mean(score_components)

            # Generate issues list
            if missing_percentage > 10:
                quality_report['issues'].append(f"High missing values: {missing_percentage:.1f}%")

            if duplicate_percentage > 1:
                quality_report['issues'].append(f"Duplicates found: {duplicate_percentage:.1f}%")

            if outlier_percentage > 5:
                quality_report['issues'].append(f"High outlier rate: {outlier_percentage:.1f}%")

            return quality_report

        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            return {'error': str(e)}

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get preprocessing history summary"""
        try:
            if not self.preprocessing_history:
                return {'message': 'No preprocessing history available'}

            summary = {
                'total_preprocessing_runs': len(self.preprocessing_history),
                'avg_records_processed': np.mean([h['original_records'] for h in self.preprocessing_history]),
                'avg_removal_rate': np.mean([h['removal_percentage'] for h in self.preprocessing_history]),
                'common_processing_steps': [],
                'last_run': self.preprocessing_history[-1] if self.preprocessing_history else None
            }

            # Find most common processing steps
            all_steps = []
            for history in self.preprocessing_history:
                all_steps.extend(history['processing_steps'])

            if all_steps:
                from collections import Counter
                step_counts = Counter(all_steps)
                summary['common_processing_steps'] = step_counts.most_common(5)

            return summary

        except Exception as e:
            logger.error(f"Preprocessing summary generation failed: {e}")
            return {'error': str(e)}

# Convenience functions

def preprocess_ohlcv_quick(df: pd.DataFrame,
                          remove_outliers: bool = True) -> pd.DataFrame:
    """Quick OHLCV preprocessing with default settings"""
    preprocessor = DataPreprocessor()

    config = {
        'remove_duplicates': True,
        'handle_missing_values': True,
        'detect_outliers': remove_outliers,
        'validate_ohlc': True,
        'remove_zero_volume': True
    }

    return preprocessor.preprocess_ohlcv_data(df, config)

def preprocess_features_quick(df: pd.DataFrame,
                            scale: bool = False) -> pd.DataFrame:
    """Quick feature preprocessing with default settings"""
    preprocessor = DataPreprocessor()

    config = {
        'handle_missing': True,
        'remove_constant_features': True,
        'handle_inf_values': True,
        'scaling_method': 'standard' if scale else None
    }

    return preprocessor.preprocess_features(df, config)

# Usage example
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')

    # Create OHLCV data with some issues
    prices = 45000 + np.cumsum(np.random.randn(1000) * 50)

    # Add some data quality issues
    sample_df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.randint(1000, 5000, 1000)
    }, index=dates)

    # Add some missing values
    sample_df.loc[sample_df.index[100:110], 'volume'] = np.nan

    # Add some outliers
    sample_df.loc[sample_df.index[500], 'high'] = prices[500] * 2

    # Add some zero volumes
    sample_df.loc[sample_df.index[200:205], 'volume'] = 0

    # Test preprocessing
    preprocessor = DataPreprocessor()

    print("Original data shape:", sample_df.shape)
    print("Missing values:", sample_df.isnull().sum().sum())

    # Preprocess OHLCV data
    cleaned_df = preprocessor.preprocess_ohlcv_data(sample_df)

    print("Cleaned data shape:", cleaned_df.shape)
    print("Missing values after cleaning:", cleaned_df.isnull().sum().sum())

    # Test data quality validation
    quality_report = preprocessor.validate_data_quality(cleaned_df)
    print(f"Data quality score: {quality_report['quality_score']:.1f}/100")
    print(f"Issues found: {quality_report['issues']}")

    # Test feature preprocessing
    feature_df = pd.DataFrame(np.random.randn(100, 20), columns=[f'feature_{i}' for i in range(20)])

    # Add some correlated features
    feature_df['feature_20'] = feature_df['feature_0'] * 0.9 + np.random.randn(100) * 0.1

    processed_features = preprocessor.preprocess_features(feature_df, {'correlation_threshold': 0.8})

    print(f"Features before: {feature_df.shape[1]}, after: {processed_features.shape[1]}")

    print("Data preprocessor test completed!")
