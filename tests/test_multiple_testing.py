# -*- coding: utf-8 -*-
"""
Tests for Multiple Hypothesis Testing Correction

Tests all correction methods:
1. Bonferroni
2. Holm-Bonferroni
3. Benjamini-Hochberg FDR
4. White's Reality Check
5. Romano-Wolf Stepdown
"""
import unittest
import numpy as np
from optuna_system.utils.multiple_testing import (
    MultipleTestingCorrector, WhiteRealityCheck, RomanoWolfCorrection,
    calculate_optimal_trials, calculate_minimum_detectable_effect
)


class TestMultipleTestingCorrector(unittest.TestCase):
    """Test MultipleTestingCorrector class"""
    
    def setUp(self):
        self.corrector = MultipleTestingCorrector(alpha=0.05)
    
    def test_bonferroni_significant(self):
        """Test Bonferroni with significant result"""
        # p=0.00001, n=1000, adjusted_alpha=0.05/1000=0.00005
        # 0.00001 < 0.00005, should be significant
        result = self.corrector.bonferroni_correction(0.00001, 1000)
        
        self.assertTrue(result.is_significant)
        self.assertEqual(result.method, "Bonferroni")
        self.assertEqual(result.n_tests, 1000)
    
    def test_bonferroni_not_significant(self):
        """Test Bonferroni with non-significant result"""
        # p=0.01, n=1000, adjusted_alpha=0.05/1000=0.00005
        # 0.01 > 0.00005, should not be significant
        result = self.corrector.bonferroni_correction(0.01, 1000)
        
        self.assertFalse(result.is_significant)
    
    def test_bonferroni_corrected_pvalue(self):
        """Test corrected p-value calculation"""
        result = self.corrector.bonferroni_correction(0.01, 100)
        
        # Corrected p-value = original * n_tests = 0.01 * 100 = 1.0
        self.assertAlmostEqual(result.corrected_pvalue, 1.0, places=4)
    
    def test_holm_bonferroni(self):
        """Test Holm-Bonferroni stepdown"""
        # 5 p-values
        p_values = [0.001, 0.01, 0.03, 0.04, 0.06]
        results = self.corrector.holm_bonferroni(p_values)
        
        self.assertEqual(len(results), 5)
        
        # First should be significant (0.001 < 0.05/5=0.01)
        self.assertTrue(results[0].is_significant)
        
        # All should have method name
        self.assertEqual(results[0].method, "Holm-Bonferroni")
    
    def test_benjamini_hochberg_fdr(self):
        """Test Benjamini-Hochberg FDR control"""
        # 10 p-values
        p_values = [0.001, 0.005, 0.01, 0.02, 0.03, 
                   0.04, 0.05, 0.06, 0.08, 0.1]
        results = self.corrector.benjamini_hochberg_fdr(p_values)
        
        self.assertEqual(len(results), 10)
        
        # At least first few should be significant
        self.assertTrue(results[0].is_significant)
        
        # Method name
        self.assertEqual(results[0].method, "Benjamini-Hochberg")
    
    def test_methods_comparison(self):
        """Compare conservativeness of different methods"""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
        
        bonf_results = [
            self.corrector.bonferroni_correction(p, len(p_values))
            for p in p_values
        ]
        holm_results = self.corrector.holm_bonferroni(p_values)
        bh_results = self.corrector.benjamini_hochberg_fdr(p_values)
        
        # Count significant results
        bonf_sig = sum(r.is_significant for r in bonf_results)
        holm_sig = sum(r.is_significant for r in holm_results)
        bh_sig = sum(r.is_significant for r in bh_results)
        
        # BH should be least conservative (most significant)
        # Bonferroni should be most conservative (least significant)
        self.assertLessEqual(bonf_sig, holm_sig)
        self.assertLessEqual(holm_sig, bh_sig)


class TestWhiteRealityCheck(unittest.TestCase):
    """Test White's Reality Check"""
    
    def setUp(self):
        self.reality_check = WhiteRealityCheck(alpha=0.05)
        np.random.seed(42)
    
    def test_reality_check_significant(self):
        """Test with truly superior strategy"""
        # Best strategy significantly better than others
        all_perfs = [0.5, 0.52, 0.51, 0.48, 0.49]
        all_perfs.extend([0.45 + np.random.rand() * 0.05 for _ in range(95)])
        best_perf = 0.8  # Much better
        benchmark = 0.5
        
        result = self.reality_check.test(
            best_perf, all_perfs, benchmark, n_bootstrap=500
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('p_value', result)
        self.assertIn('is_significant', result)
        self.assertTrue(result['is_significant'])
    
    def test_reality_check_not_significant(self):
        """Test with lucky strategy (not truly better)"""
        # Best is just lucky draw from same distribution
        np.random.seed(42)
        all_perfs = [0.5 + np.random.randn() * 0.1 for _ in range(100)]
        best_perf = max(all_perfs)  # Just the lucky best
        benchmark = 0.5
        
        result = self.reality_check.test(
            best_perf, all_perfs, benchmark, n_bootstrap=500
        )
        
        # Should not be significant (just luck)
        self.assertFalse(result['is_significant'])
    
    def test_reality_check_structure(self):
        """Test result structure"""
        result = self.reality_check.test(
            0.6, [0.5, 0.52, 0.55], 0.5, n_bootstrap=100
        )
        
        required_keys = ['best_performance', 'benchmark_performance',
                        'best_relative', 'p_value', 'is_significant',
                        'alpha', 'n_strategies', 'n_bootstrap',
                        'bootstrap_quantiles']
        
        for key in required_keys:
            self.assertIn(key, result)


class TestRomanoWolfCorrection(unittest.TestCase):
    """Test Romano-Wolf Stepdown Correction"""
    
    def setUp(self):
        self.rw = RomanoWolfCorrection(alpha=0.05)
        np.random.seed(42)
    
    def test_stepdown_procedure(self):
        """Test Romano-Wolf stepdown"""
        # Mix of good and bad strategies
        perfs = np.array([0.6, 0.7, 0.8, 0.55, 0.52])
        benchmark = 0.5
        
        results = self.rw.stepdown_test(perfs, benchmark, n_bootstrap=500)
        
        self.assertEqual(len(results), 5)
        
        # Each result should have required fields
        for r in results:
            self.assertIn('p_value', r)
            self.assertIn('is_significant', r)
            self.assertIn('performance', r)
    
    def test_stepdown_ordering(self):
        """Test that better strategies are tested first"""
        perfs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        benchmark = 0.5
        
        results = self.rw.stepdown_test(perfs, benchmark, n_bootstrap=500)
        
        # Results should be in original order
        for i, r in enumerate(results):
            self.assertEqual(r['original_index'], i)
    
    def test_all_null(self):
        """Test when all strategies are null (not better)"""
        # All around benchmark
        perfs = np.array([0.5 + np.random.randn() * 0.01 for _ in range(10)])
        benchmark = 0.5
        
        results = self.rw.stepdown_test(perfs, benchmark, n_bootstrap=500)
        
        # Most should not be significant
        sig_count = sum(r['is_significant'] for r in results)
        self.assertLess(sig_count, 5)  # Less than half


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_calculate_optimal_trials(self):
        """Test optimal trials calculation"""
        # Current: 37,500 trials
        current_trials = 37500
        optimal = calculate_optimal_trials(current_trials, method='bonferroni')
        
        # Should be significantly reduced (75% reduction)
        self.assertLess(optimal, current_trials)
        self.assertAlmostEqual(optimal, current_trials * 0.25, delta=100)
    
    def test_optimal_trials_different_methods(self):
        """Test different methods give different reductions"""
        current = 10000
        
        bonf = calculate_optimal_trials(current, method='bonferroni')
        holm = calculate_optimal_trials(current, method='holm')
        bh = calculate_optimal_trials(current, method='bh')
        
        # Bonferroni should be most conservative (smallest)
        self.assertLessEqual(bonf, holm)
        self.assertLessEqual(holm, bh)
    
    def test_minimum_detectable_effect(self):
        """Test MDE calculation"""
        # Large sample
        mde_large = calculate_minimum_detectable_effect(1000)
        
        # Small sample
        mde_small = calculate_minimum_detectable_effect(100)
        
        # Smaller sample needs larger effect to detect
        self.assertGreater(mde_small, mde_large)
    
    def test_mde_power_relationship(self):
        """Test MDE increases with higher power requirement"""
        n_trials = 500
        
        mde_80 = calculate_minimum_detectable_effect(n_trials, power=0.80)
        mde_90 = calculate_minimum_detectable_effect(n_trials, power=0.90)
        
        # Higher power needs larger effect
        self.assertGreater(mde_90, mde_80)


class TestIntegration(unittest.TestCase):
    """Integration tests simulating real hyperparameter optimization"""
    
    def test_crypto_trading_scenario(self):
        """Test scenario: 37,500 trials in crypto trading optimization"""
        # Simulate Layer1×Layer2 trials
        np.random.seed(42)
        layer1_trials = 150
        layer2_trials = 250
        total_trials = layer1_trials * layer2_trials  # 37,500
        
        # Simulate F1 scores (most are around 0.5, some are better by luck)
        f1_scores = np.random.beta(5, 5, total_trials)  # Mean ~0.5
        
        # Find "best" strategy (might just be lucky)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        
        # Test with White's Reality Check
        reality_check = WhiteRealityCheck()
        result = reality_check.test(
            best_performance=best_f1,
            all_performances=f1_scores.tolist(),
            benchmark_performance=0.5,
            n_bootstrap=1000
        )
        
        # With 37,500 trials, best might not be truly significant
        print(f"\nReality Check: p={result['p_value']:.4f}, "
              f"significant={result['is_significant']}")
        
        # Calculate recommended trials
        optimal_trials = calculate_optimal_trials(total_trials)
        reduction = (total_trials - optimal_trials) / total_trials * 100
        
        print(f"Recommended: {total_trials} → {optimal_trials} trials "
              f"({reduction:.1f}% reduction)")
        
        self.assertLess(optimal_trials, total_trials)
    
    def test_multiple_correction_comparison(self):
        """Compare all correction methods on same data"""
        np.random.seed(42)
        
        # Simulate 100 p-values
        p_values = np.random.beta(2, 5, 100)  # Skewed toward smaller
        
        corrector = MultipleTestingCorrector()
        
        # Bonferroni
        bonf_results = [
            corrector.bonferroni_correction(p, len(p_values))
            for p in p_values
        ]
        bonf_sig = sum(r.is_significant for r in bonf_results)
        
        # Holm-Bonferroni
        holm_results = corrector.holm_bonferroni(p_values.tolist())
        holm_sig = sum(r.is_significant for r in holm_results)
        
        # Benjamini-Hochberg
        bh_results = corrector.benjamini_hochberg_fdr(p_values.tolist())
        bh_sig = sum(r.is_significant for r in bh_results)
        
        print(f"\nSignificant results:")
        print(f"  Bonferroni: {bonf_sig}/100")
        print(f"  Holm: {holm_sig}/100")
        print(f"  BH-FDR: {bh_sig}/100")
        
        # BH should find most, Bonferroni least
        self.assertLessEqual(bonf_sig, holm_sig)
        self.assertLessEqual(holm_sig, bh_sig)


if __name__ == '__main__':
    unittest.main()

