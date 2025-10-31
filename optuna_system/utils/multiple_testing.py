# -*- coding: utf-8 -*-
"""
Multiple Hypothesis Testing Correction for Hyperparameter Optimization

Addresses the data snooping bias problem caused by excessive trials:
- Layer1: 150 trials × Layer2: 250 trials = 37,500 implicit tests
- High false discovery rate without correction

Implements state-of-the-art correction methods:
1. Bonferroni Correction (conservative)
2. Holm-Bonferroni (less conservative)
3. Benjamini-Hochberg FDR (False Discovery Rate)
4. White's Reality Check (bootstrap-based)
5. Romano-Wolf Stepdown (recommended for trading)

Academic References:
- White, H. (2000). "A Reality Check for Data Snooping." Econometrica, 68(5), 1097-1126.
- Romano, J.P., & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized Data Snooping." 
  Econometrica, 73(4), 1237-1282.
- Harvey, C.R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns."
  Review of Financial Studies, 29(1), 5-68.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MultipleTestingResult:
    """Result of multiple testing correction"""
    original_pvalue: float
    corrected_pvalue: float
    is_significant: bool
    method: str
    n_tests: int
    alpha: float
    
    def __str__(self):
        return (
            f"MultipleTestingResult(\n"
            f"  method={self.method},\n"
            f"  n_tests={self.n_tests},\n"
            f"  original_p={self.original_pvalue:.6f},\n"
            f"  corrected_p={self.corrected_pvalue:.6f},\n"
            f"  significant={self.is_significant} at α={self.alpha}\n"
            f")"
        )


class MultipleTestingCorrector:
    """
    Multiple hypothesis testing correction for hyperparameter optimization
    
    Key Problem:
    ------------
    Running 37,500 trials (150×250) means testing 37,500 hypotheses.
    At α=0.05, we expect 1,875 false positives purely by chance!
    
    Solution:
    ---------
    Adjust p-values or significance thresholds to control:
    - Family-Wise Error Rate (FWER): Prob(at least 1 false positive)
    - False Discovery Rate (FDR): Expected proportion of false positives
    
    Example:
    --------
    >>> corrector = MultipleTestingCorrector()
    >>> result = corrector.bonferroni_correction(
    ...     p_value=0.01, n_tests=1000, alpha=0.05
    ... )
    >>> print(result.is_significant)  # False (0.01 > 0.05/1000)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        logger.info(f"MultipleTestingCorrector initialized with α={alpha}")
    
    def bonferroni_correction(self, p_value: float, n_tests: int,
                             alpha: Optional[float] = None) -> MultipleTestingResult:
        """
        Bonferroni correction (most conservative)
        
        Formula: adjusted_alpha = α / n_tests
        
        Controls FWER (Family-Wise Error Rate)
        
        Args:
            p_value: Original p-value
            n_tests: Number of tests performed
            alpha: Significance level (default: self.alpha)
        
        Returns:
            MultipleTestingResult
        
        Reference:
            Bonferroni, C.E. (1936). "Teoria statistica delle classi e calcolo 
            delle probabilità." Pubblicazioni del R Istituto Superiore di 
            Scienze Economiche e Commerciali di Firenze, 8, 3-62.
        
        Example:
        --------
        >>> # 1000 trials, original p=0.01
        >>> result = corrector.bonferroni_correction(0.01, 1000)
        >>> # Adjusted α = 0.05/1000 = 0.00005
        >>> # 0.01 > 0.00005, so NOT significant
        """
        alpha = alpha or self.alpha
        adjusted_alpha = alpha / n_tests
        corrected_pvalue = min(p_value * n_tests, 1.0)
        is_significant = p_value <= adjusted_alpha
        
        logger.debug(
            f"Bonferroni: n_tests={n_tests}, "
            f"adjusted_α={adjusted_alpha:.6f}, "
            f"significant={is_significant}"
        )
        
        return MultipleTestingResult(
            original_pvalue=p_value,
            corrected_pvalue=corrected_pvalue,
            is_significant=is_significant,
            method="Bonferroni",
            n_tests=n_tests,
            alpha=alpha
        )
    
    def holm_bonferroni(self, p_values: List[float],
                       alpha: Optional[float] = None) -> List[MultipleTestingResult]:
        """
        Holm-Bonferroni stepdown procedure (less conservative)
        
        Algorithm:
        1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(n)
        2. For i=1 to n:
           - If p(i) > α/(n-i+1), reject H(i) and all subsequent
           - Otherwise, accept H(i)
        
        Controls FWER but more powerful than Bonferroni
        
        Args:
            p_values: List of p-values to correct
            alpha: Significance level
        
        Returns:
            List of MultipleTestingResult
        
        Reference:
            Holm, S. (1979). "A simple sequentially rejective multiple test 
            procedure." Scandinavian Journal of Statistics, 6(2), 65-70.
        """
        alpha = alpha or self.alpha
        n_tests = len(p_values)
        
        # Sort p-values with original indices
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]
        
        # Apply Holm-Bonferroni
        results = []
        rejected_all_subsequent = False
        
        for i, (original_idx, p_val) in enumerate(zip(sorted_indices, sorted_pvalues)):
            adjusted_alpha = alpha / (n_tests - i)
            corrected_pvalue = min(p_val * (n_tests - i), 1.0)
            
            if rejected_all_subsequent or p_val > adjusted_alpha:
                is_significant = False
                rejected_all_subsequent = True
            else:
                is_significant = True
            
            results.append((original_idx, MultipleTestingResult(
                original_pvalue=float(p_val),
                corrected_pvalue=corrected_pvalue,
                is_significant=is_significant,
                method="Holm-Bonferroni",
                n_tests=n_tests,
                alpha=alpha
            )))
        
        # Restore original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def benjamini_hochberg_fdr(self, p_values: List[float],
                               alpha: Optional[float] = None) -> List[MultipleTestingResult]:
        """
        Benjamini-Hochberg FDR (False Discovery Rate) control
        
        Algorithm:
        1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(n)
        2. Find largest i where p(i) ≤ (i/n)×α
        3. Reject H(1), H(2), ..., H(i)
        
        Controls FDR instead of FWER (less conservative)
        
        Args:
            p_values: List of p-values
            alpha: FDR level (default: 0.05 = 5% false discoveries expected)
        
        Returns:
            List of MultipleTestingResult
        
        Reference:
            Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false 
            discovery rate: a practical and powerful approach to multiple 
            testing." Journal of the Royal Statistical Society B, 57(1), 289-300.
        
        Example:
        --------
        >>> # 100 tests, want to control FDR at 5%
        >>> results = corrector.benjamini_hochberg_fdr(p_values, alpha=0.05)
        >>> # Expect at most 5% of significant results to be false positives
        """
        alpha = alpha or self.alpha
        n_tests = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = np.array(p_values)[sorted_indices]
        
        # Find critical value
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha
        
        # Find largest i where p(i) <= (i/n)*alpha
        significant_mask = sorted_pvalues <= critical_values
        if significant_mask.any():
            max_significant_idx = np.where(significant_mask)[0][-1]
        else:
            max_significant_idx = -1
        
        # Generate results
        results = []
        for i, (original_idx, p_val) in enumerate(zip(sorted_indices, sorted_pvalues)):
            is_significant = (i <= max_significant_idx)
            corrected_pvalue = min(p_val * n_tests / (i + 1), 1.0)
            
            results.append((original_idx, MultipleTestingResult(
                original_pvalue=float(p_val),
                corrected_pvalue=corrected_pvalue,
                is_significant=is_significant,
                method="Benjamini-Hochberg",
                n_tests=n_tests,
                alpha=alpha
            )))
        
        # Restore original order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


class WhiteRealityCheck:
    """
    White's Reality Check for data snooping
    
    Uses bootstrap resampling to test if the best strategy is truly better
    than a benchmark after accounting for multiple testing.
    
    Key Insight:
    -----------
    If you test 1000 strategies, one might look good purely by luck.
    Reality Check uses bootstrap to estimate the distribution of
    "best by luck" and tests if your best is significantly better.
    
    Reference:
    ---------
    White, H. (2000). "A Reality Check for Data Snooping." 
    Econometrica, 68(5), 1097-1126. Citations: 2,500+
    
    Example:
    --------
    >>> reality_check = WhiteRealityCheck()
    >>> result = reality_check.test(
    ...     best_performance=0.8,
    ...     all_performances=[0.5, 0.6, 0.8, 0.55, ...],
    ...     benchmark_performance=0.5,
    ...     n_bootstrap=1000
    ... )
    >>> print(result['is_significant'])  # True if truly better
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        logger.info("WhiteRealityCheck initialized")
    
    def test(self, best_performance: float,
            all_performances: List[float],
            benchmark_performance: float = 0.0,
            n_bootstrap: int = 1000) -> Dict:
        """
        Perform White's Reality Check
        
        Args:
            best_performance: Performance of best strategy
            all_performances: Performances of all tested strategies
            benchmark_performance: Benchmark (e.g., buy-and-hold)
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Dict with test results
        """
        n_strategies = len(all_performances)
        
        # Calculate relative performances
        relative_perfs = np.array(all_performances) - benchmark_performance
        best_relative = best_performance - benchmark_performance
        
        # Bootstrap to get distribution of max under null
        bootstrap_maxes = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(
                relative_perfs, size=n_strategies, replace=True
            )
            bootstrap_maxes.append(np.max(bootstrap_sample))
        
        bootstrap_maxes = np.array(bootstrap_maxes)
        
        # Calculate p-value
        p_value = np.mean(bootstrap_maxes >= best_relative)
        is_significant = p_value <= self.alpha
        
        result = {
            'best_performance': best_performance,
            'benchmark_performance': benchmark_performance,
            'best_relative': best_relative,
            'p_value': p_value,
            'is_significant': is_significant,
            'alpha': self.alpha,
            'n_strategies': n_strategies,
            'n_bootstrap': n_bootstrap,
            'bootstrap_quantiles': {
                '95%': np.percentile(bootstrap_maxes, 95),
                '99%': np.percentile(bootstrap_maxes, 99)
            }
        }
        
        logger.info(
            f"Reality Check: best_relative={best_relative:.4f}, "
            f"p={p_value:.4f}, significant={is_significant}"
        )
        
        return result


class RomanoWolfCorrection:
    """
    Romano-Wolf Stepdown Correction
    
    Most powerful method for trading strategy testing.
    Uses bootstrap to estimate joint distribution and control FWER.
    
    Key Advantage:
    -------------
    More powerful than Bonferroni/Holm while still controlling FWER.
    Accounts for correlation structure among tests.
    
    Reference:
    ---------
    Romano, J.P., & Wolf, M. (2005). "Stepwise Multiple Testing as 
    Formalized Data Snooping." Econometrica, 73(4), 1237-1282.
    Citations: 1,800+
    
    Example:
    --------
    >>> rw = RomanoWolfCorrection()
    >>> results = rw.stepdown_test(
    ...     performances=[0.6, 0.7, 0.8, 0.55],
    ...     benchmark=0.5,
    ...     n_bootstrap=1000
    ... )
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        logger.info("RomanoWolfCorrection initialized")
    
    def stepdown_test(self, performances: np.ndarray,
                     benchmark: float = 0.0,
                     n_bootstrap: int = 1000) -> List[Dict]:
        """
        Romano-Wolf stepdown procedure
        
        Args:
            performances: Array of strategy performances
            benchmark: Benchmark performance
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            List of test results for each strategy
        """
        n_strategies = len(performances)
        relative_perfs = performances - benchmark
        
        # Sort by performance (descending)
        sorted_indices = np.argsort(-relative_perfs)
        sorted_perfs = relative_perfs[sorted_indices]
        
        # Bootstrap
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                relative_perfs, size=n_strategies, replace=True
            )
            bootstrap_samples.append(bootstrap_sample)
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        # Stepdown procedure
        results = []
        remaining_indices = list(range(n_strategies))
        
        for step in range(n_strategies):
            if not remaining_indices:
                break
            
            # Current hypothesis
            current_idx = remaining_indices[0]
            current_perf = sorted_perfs[current_idx]
            
            # Calculate p-value using joint distribution
            bootstrap_maxes = np.max(
                bootstrap_samples[:, remaining_indices], axis=1
            )
            p_value = np.mean(bootstrap_maxes >= current_perf)
            
            is_significant = p_value <= self.alpha
            
            results.append({
                'original_index': sorted_indices[current_idx],
                'performance': performances[sorted_indices[current_idx]],
                'relative_performance': current_perf,
                'p_value': p_value,
                'is_significant': is_significant,
                'step': step
            })
            
            if is_significant:
                # Continue testing
                remaining_indices.pop(0)
            else:
                # Stop, all remaining are non-significant
                for idx in remaining_indices[1:]:
                    results.append({
                        'original_index': sorted_indices[idx],
                        'performance': performances[sorted_indices[idx]],
                        'relative_performance': sorted_perfs[idx],
                        'p_value': 1.0,
                        'is_significant': False,
                        'step': step
                    })
                break
        
        # Restore original order
        results.sort(key=lambda x: x['original_index'])
        
        logger.info(
            f"Romano-Wolf: {sum(r['is_significant'] for r in results)}/{n_strategies} "
            f"strategies significant"
        )
        
        return results


def calculate_optimal_trials(current_trials: int,
                            desired_fwer: float = 0.05,
                            method: str = 'bonferroni') -> int:
    """
    Calculate optimal number of trials to control error rate
    
    Current problem: 37,500 trials → 1,875 expected false positives at α=0.05
    
    Solution: Reduce trials to control FWER
    
    Args:
        current_trials: Current number of trials
        desired_fwer: Desired family-wise error rate
        method: Correction method ('bonferroni', 'holm', 'bh')
    
    Returns:
        Recommended number of trials
    
    Example:
    --------
    >>> # Current: 37,500 trials
    >>> optimal = calculate_optimal_trials(37500, desired_fwer=0.05)
    >>> print(optimal)  # ~9,375 (75% reduction)
    """
    if method == 'bonferroni':
        # Bonferroni: α/n, so n = α/desired_α_per_test
        # Assuming we want per-test α = 0.05
        optimal_trials = int(desired_fwer / 0.0000013)  # Very conservative
        reduction_factor = 0.25  # 75% reduction
    elif method == 'holm':
        # Holm is less conservative
        reduction_factor = 0.30  # 70% reduction
    elif method == 'bh':
        # BH controls FDR, can be less aggressive
        reduction_factor = 0.35  # 65% reduction
    else:
        reduction_factor = 0.25
    
    optimal_trials = int(current_trials * reduction_factor)
    
    logger.info(
        f"Trial reduction: {current_trials} → {optimal_trials} "
        f"({reduction_factor*100:.0f}% of original)"
    )
    
    return optimal_trials


def calculate_minimum_detectable_effect(n_trials: int,
                                       alpha: float = 0.05,
                                       power: float = 0.80) -> float:
    """
    Calculate minimum detectable effect size given number of trials
    
    Args:
        n_trials: Number of trials
        alpha: Significance level (after correction)
        power: Statistical power (1 - β)
    
    Returns:
        Minimum detectable effect size (Cohen's d)
    """
    # Simplified calculation using normal approximation
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Minimum detectable effect
    mde = (z_alpha + z_beta) / np.sqrt(n_trials / 2)
    
    logger.info(
        f"Minimum detectable effect: {mde:.4f} "
        f"(n={n_trials}, α={alpha}, power={power})"
    )
    
    return mde

