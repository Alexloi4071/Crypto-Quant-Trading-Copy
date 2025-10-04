"""
Feature Optimizer Module
Specialized feature selection and engineering optimization
Supports correlation analysis, mutual information, and custom feature scoring
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import itertools
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings

warnings.filterwarnings('ignore')

# ML libraries
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, RFECV,
    mutual_info_classif, f_classif, chi2
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# Statistical libraries
try:
    from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFwe
    SCIPY_EXTENDED_AVAILABLE = True
except ImportError:
    SCIPY_EXTENDED_AVAILABLE = False

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator

logger = setup_logger(__name__)


class FeatureOptimizer:
    """Advanced feature selection and optimization"""

    def __init__(self):
        self.feature_importance_methods = {
            'mutual_info': self._mutual_info_selection,
            'f_classif': self._f_classif_selection,
            'chi2': self._chi2_selection,
            'lasso': self._lasso_selection,
            'random_forest': self._random_forest_selection,
            'lightgbm': self._lightgbm_selection,
            'correlation': self._correlation_selection,
            'variance': self._variance_selection,
            'rfe': self._rfe_selection,
            'boruta': self._boruta_selection if self._is_boruta_available() else None
        }

        # Remove unavailable methods
        self.feature_importance_methods = {
            k: v for k, v in self.feature_importance_methods.items() if v is not None
        }

        self.selected_features = {}
        self.feature_scores = {}
        self.optimization_history = []


    def _is_boruta_available(self) -> bool:
        """Check if Boruta is available"""
        try:
            import boruta
            return True
        except ImportError:
            return False

    @timing_decorator
    def optimize_features(self, X: pd.DataFrame, y: pd.Series,
                         method: str = 'combined',
                         max_features: int = 50,
                         min_features: int = 10,
                         correlation_threshold: float = 0.95,
                         **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """
        Optimize feature selection using specified method

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Feature selection method
            max_features: Maximum number of features to select
            min_features: Minimum number of features to select
            correlation_threshold: Threshold for correlation-based filtering
            **kwargs: Additional parameters for specific methods

        Returns:
            Tuple of (selected_features, feature_scores)
        """
        try:
            logger.info(f"Starting feature optimization with method: {method}")

            if X.empty or y.empty:
                logger.warning("Empty input data")
                return [], {}

            # Handle missing values
            X_clean, y_clean = self._clean_data(X, y)

            if method == 'combined':
                return self._combined_selection(X_clean, y_clean, max_features, min_features,
                                              correlation_threshold, **kwargs)
            elif method in self.feature_importance_methods:
                return self.feature_importance_methods[method](
                    X_clean, y_clean, max_features, **kwargs
                )
            else:
                logger.error(f"Unknown feature selection method: {method}")
                return list(X_clean.columns[:max_features]), {}

        except Exception as e:
            logger.error(f"Feature optimization failed: {e}")
            return list(X.columns[:min(max_features, len(X.columns))]), {}


    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data for feature selection"""
        try:
            # Remove rows with NaN targets
            valid_mask = ~y.isna()
            X_clean = X.loc[valid_mask].copy()
            y_clean = y.loc[valid_mask].copy()

            # Handle infinite values
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

            # Fill NaN features with median
            for col in X_clean.columns:
                if X_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    X_clean[col] = X_clean[col].fillna(X_clean[col].median())
                else:
                    X_clean[col] = X_clean[col].fillna(0)

            # Remove constant features
            constant_cols = X_clean.columns[X_clean.nunique() <= 1].tolist()
            if constant_cols:
                X_clean = X_clean.drop(columns=constant_cols)
                logger.info(f"Removed {len(constant_cols)} constant features")

            return X_clean, y_clean

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return X, y


    def _combined_selection(self, X: pd.DataFrame, y: pd.Series,
                          max_features: int, min_features: int,
                          correlation_threshold: float, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Combined feature selection using multiple methods"""
        try:
            logger.info("Running combined feature selection")

            all_scores = {}
            method_weights = {
                'mutual_info': 0.3,
                'lightgbm': 0.3,
                'f_classif': 0.2,
                'correlation': 0.2
            }

            # Run each method and collect scores
            for method, weight in method_weights.items():
                if method in self.feature_importance_methods:
                    try:
                        _, scores = self.feature_importance_methods[method](
                            X, y, len(X.columns), **kwargs
                        )

                        # Normalize scores to 0-1 range
                        if scores:
                            max_score = max(scores.values())
                            min_score = min(scores.values())
                            if max_score > min_score:
                                normalized_scores = {
                                    feat: weight * (score - min_score) / (max_score - min_score)
                                    for feat, score in scores.items()
                                }
                            else:
                                normalized_scores = {feat: weight for feat in scores.keys()}

                            # Add to combined scores
                            for feat, score in normalized_scores.items():
                                all_scores[feat] = all_scores.get(feat, 0) + score

                        logger.info(f"Completed {method} selection: {len(scores)} features scored")

                    except Exception as e:
                        logger.warning(f"Method {method} failed: {e}")
                        continue

            if not all_scores:
                logger.warning("No valid feature scores found, using all features")
                return list(X.columns), {}

            # Remove highly correlated features
            selected_features = self._remove_correlated_features(
                X, all_scores, correlation_threshold
            )

            # Select top features
            sorted_features = sorted(
                selected_features.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Ensure we have between min_features and max_features
            n_features = max(min_features, min(max_features, len(sorted_features)))
            final_features = [feat for feat, score in sorted_features[:n_features]]
            final_scores = {feat: score for feat, score in sorted_features[:n_features]}

            logger.info(f"Combined selection completed: {len(final_features)} features selected")
            return final_features, final_scores

        except Exception as e:
            logger.error(f"Combined feature selection failed: {e}")
            return list(X.columns[:max_features]), {}


    def _remove_correlated_features(self, X: pd.DataFrame, feature_scores: Dict[str, float],
                                   threshold: float) -> Dict[str, float]:
        """Remove highly correlated features, keeping the one with highest score"""
        try:
            if threshold >= 1.0:  # No correlation filtering
                return feature_scores

            # Calculate correlation matrix
            corr_matrix = X[list(feature_scores.keys())].corr().abs()

            # Find highly correlated pairs
            highly_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] >= threshold:
                        feat1 = corr_matrix.columns[i]
                        feat2 = corr_matrix.columns[j]
                        highly_corr_pairs.append((feat1, feat2, corr_matrix.iloc[i, j]))

            # Remove features with lower scores from correlated pairs
            features_to_remove = set()
            for feat1, feat2, corr in highly_corr_pairs:
                score1 = feature_scores.get(feat1, 0)
                score2 = feature_scores.get(feat2, 0)

                if score1 >= score2:
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)

            # Return filtered scores
            filtered_scores = {
                feat: score for feat, score in feature_scores.items()
                if feat not in features_to_remove
            }

            logger.info(f"Removed {len(features_to_remove)} highly correlated features")
            return filtered_scores

        except Exception as e:
            logger.error(f"Correlation filtering failed: {e}")
            return feature_scores


    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series,
                             k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Mutual information feature selection"""
        try:
            # Ensure discrete target for mutual info
            if y.dtype not in ['int64', 'int32']:
                y_discrete = pd.cut(y, bins=10, labels=False)
            else:
                y_discrete = y

            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X, y_discrete, random_state=42)

            # Create feature scores dictionary
            feature_scores = dict(zip(X.columns, mi_scores))

            # Select top k features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            selected_features = [feat for feat, score in top_features]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Mutual information selection failed: {e}")
            return list(X.columns[:k]), {}


    def _f_classif_selection(self, X: pd.DataFrame, y: pd.Series,
                           k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """F-classification feature selection"""
        try:
            # Calculate F-scores
            f_scores, p_values = f_classif(X, y)

            # Create feature scores dictionary (using F-scores)
            feature_scores = dict(zip(X.columns, f_scores))

            # Select top k features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            selected_features = [feat for feat, score in top_features]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"F-classification selection failed: {e}")
            return list(X.columns[:k]), {}


    def _chi2_selection(self, X: pd.DataFrame, y: pd.Series,
                       k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Chi-square feature selection"""
        try:
            # Make data non-negative for chi2
            X_pos = X - X.min() + 1

            # Calculate chi2 scores
            chi2_scores, p_values = chi2(X_pos, y)

            # Create feature scores dictionary
            feature_scores = dict(zip(X.columns, chi2_scores))

            # Select top k features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            selected_features = [feat for feat, score in top_features]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Chi2 selection failed: {e}")
            return list(X.columns[:k]), {}


    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series,
                        k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """LASSO feature selection"""
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit LASSO
            lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_scaled, y)

            # Get feature coefficients
            coefficients = np.abs(lasso.coef_)
            feature_scores = dict(zip(X.columns, coefficients))

            # Select top k non-zero features
            non_zero_features = {feat: score for feat, score in feature_scores.items() if score > 0}

            if len(non_zero_features) == 0:
                # If all coefficients are zero, take top k by absolute value
                top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            else:
                # Take top k non-zero features
                top_features = sorted(non_zero_features.items(), key=lambda x: x[1], reverse=True)[:k]

            selected_features = [feat for feat, score in top_features]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"LASSO selection failed: {e}")
            return list(X.columns[:k]), {}


    def _random_forest_selection(self, X: pd.DataFrame, y: pd.Series,
                               k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Random Forest feature importance selection"""
        try:
            # Fit Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
            rf.fit(X, y)

            # Get feature importances
            importances = rf.feature_importances_
            feature_scores = dict(zip(X.columns, importances))

            # Select top k features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            selected_features = [feat for feat, score in top_features]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Random Forest selection failed: {e}")
            return list(X.columns[:k]), {}


    def _lightgbm_selection(self, X: pd.DataFrame, y: pd.Series,
                          k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """LightGBM feature importance selection"""
        try:
            # Prepare data
            train_data = lgb.Dataset(X, label=y)

            # LightGBM parameters
            params = {
                'objective': 'multiclass' if len(np.unique(y)) > 2 else 'binary',
                'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else None,
                'metric': 'multi_logloss' if len(np.unique(y)) > 2 else 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'verbosity': -1,
                'random_state': 42
            }

            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100,
                callbacks=[lgb.log_evaluation(0)]
            )

            # Get feature importances
            importances = model.feature_importance(importance_type='gain')
            feature_scores = dict(zip(X.columns, importances))

            # Select top k features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            selected_features = [feat for feat, score in top_features]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"LightGBM selection failed: {e}")
            return list(X.columns[:k]), {}


    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series,
                             k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Correlation-based feature selection"""
        try:
            # Calculate correlations with target
            correlations = {}
            for col in X.columns:
                try:
                    # Use Spearman correlation (more robust)
                    corr, p_value = spearmanr(X[col], y)
                    correlations[col] = abs(corr) if not np.isnan(corr) else 0
                except:
                    correlations[col] = 0

            # Select top k features
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:k]
            selected_features = [feat for feat, score in top_features]

            return selected_features, correlations

        except Exception as e:
            logger.error(f"Correlation selection failed: {e}")
            return list(X.columns[:k]), {}


    def _variance_selection(self, X: pd.DataFrame, y: pd.Series,
                          k: int, threshold: float = 0.0, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Variance-based feature selection"""
        try:
            # Calculate variances
            variances = X.var()

            # Filter by threshold
            high_var_features = variances[variances > threshold]

            # Select top k by variance
            top_features = high_var_features.nlargest(k)
            selected_features = top_features.index.tolist()
            feature_scores = top_features.to_dict()

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Variance selection failed: {e}")
            return list(X.columns[:k]), {}


    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series,
                      k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Recursive Feature Elimination"""
        try:
            # Use Random Forest as estimator
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1,
                max_depth=8
            )

            # Perform RFE
            rfe = RFE(estimator, n_features_to_select=k, step=1)
            rfe.fit(X, y)

            # Get selected features
            selected_mask = rfe.support_
            selected_features = X.columns[selected_mask].tolist()

            # Get feature rankings (convert to scores, lower rank = higher score)
            rankings = rfe.ranking_
            max_rank = max(rankings)
            feature_scores = dict(zip(X.columns, [max_rank - rank + 1 for rank in rankings]))

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"RFE selection failed: {e}")
            return list(X.columns[:k]), {}


    def _boruta_selection(self, X: pd.DataFrame, y: pd.Series,
                         k: int, **kwargs) -> Tuple[List[str], Dict[str, float]]:
        """Boruta feature selection"""
        try:
            import boruta
            from boruta import BorutaPy

            # Use Random Forest as estimator
            rf = RandomForestClassifier(n_jobs=-1, random_state=42, max_depth=8)

            # Boruta feature selection
            boruta_selector = BorutaPy(
                rf,
                n_estimators='auto',
                verbose=0,
                random_state=42,
                max_iter=50  # Limit iterations for performance
            )

            boruta_selector.fit(X.values, y.values)

            # Get selected features
            selected_mask = boruta_selector.support_
            selected_features = X.columns[selected_mask].tolist()

            # Get feature rankings
            rankings = boruta_selector.ranking_
            max_rank = max(rankings)
            feature_scores = dict(zip(X.columns, [max_rank - rank + 1 for rank in rankings]))

            # If too many features selected, take top k
            if len(selected_features) > k:
                top_selected = sorted(
                    [(feat, feature_scores[feat]) for feat in selected_features],
                    key=lambda x: x[1], reverse=True
                )[:k]
                selected_features = [feat for feat, score in top_selected]

            return selected_features, feature_scores

        except Exception as e:
            logger.error(f"Boruta selection failed: {e}")
            return list(X.columns[:k]), {}


    def evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series,
                           features: List[str], cv: int = 5) -> Dict[str, float]:
        """Evaluate a feature set using cross-validation"""
        try:
            if not features or len(features) == 0:
                return {'score': 0.0, 'std': 0.0}

            X_selected = X[features]

            # Use LightGBM for evaluation
            model = lgb.LGBMClassifier(
                n_estimators=50,
                random_state=42,
                verbosity=-1
            )

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv)
            scores = []

            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                from sklearn.metrics import f1_score
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)

            return {
                'score': np.mean(scores),
                'std': np.std(scores),
                'n_features': len(features)
            }

        except Exception as e:
            logger.error(f"Feature set evaluation failed: {e}")
            return {'score': 0.0, 'std': 0.0}


    def save_feature_analysis(self, symbol: str, timeframe: str, version: str,
                            selected_features: List[str], feature_scores: Dict[str, float],
                            method_used: str) -> str:
        """Save feature analysis results"""
        try:
            results_path = config.get_processed_path(symbol, timeframe, version)
            results_path.mkdir(parents=True, exist_ok=True)

            analysis_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'version': version,
                'analysis_date': datetime.now().isoformat(),
                'method_used': method_used,
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'n_features_selected': len(selected_features),
                'n_features_total': len(feature_scores)
            }

            # Save analysis
            analysis_file = results_path / 'feature_analysis.yaml'
            import yaml
            with open(analysis_file, 'w') as f:
                yaml.dump(analysis_data, f, default_flow_style=False)

            logger.info(f"Feature analysis saved to {results_path}")
            return str(results_path)

        except Exception as e:
            logger.error(f"Failed to save feature analysis: {e}")
            return ""

# Convenience functions


def optimize_features_for_symbol(X: pd.DataFrame, y: pd.Series,
                                method: str = 'combined',
                                max_features: int = 30) -> Tuple[List[str], Dict[str, float]]:
    """Convenient function for feature optimization"""
    optimizer = FeatureOptimizer()
    return optimizer.optimize_features(X, y, method, max_features)


def evaluate_multiple_feature_sets(X: pd.DataFrame, y: pd.Series,
                                 feature_sets: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple feature sets"""
    optimizer = FeatureOptimizer()
    results = {}

    for name, features in feature_sets.items():
        results[name] = optimizer.evaluate_feature_set(X, y, features)

    return results

# Usage example
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with some relationship to features
    y = (X['feature_0'] + X['feature_1'] - X['feature_2'] > 0).astype(int)

    # Test feature optimization
    optimizer = FeatureOptimizer()

    selected_features, scores = optimizer.optimize_features(
        X, y, method='combined', max_features=10
    )

    print(f"Selected {len(selected_features)} features: {selected_features}")
    print(f"Top feature scores: {dict(list(scores.items())[:5])}")

    # Evaluate selected features
    evaluation = optimizer.evaluate_feature_set(X, y, selected_features)
    print(f"Feature set evaluation: {evaluation}")

    print("Feature optimization example completed!")
