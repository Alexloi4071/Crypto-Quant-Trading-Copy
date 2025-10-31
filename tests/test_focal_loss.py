# -*- coding: utf-8 -*-
"""
Tests for Focal Loss implementations

Tests all components:
1. Standard Focal Loss
2. Class-Balanced Focal Loss (CB-FL)
3. Focal Loss with OHEM
4. Adaptive Focal Loss
5. Regime-Aware Focal Loss
6. Crypto Trading Focal Loss (unified)
"""
import unittest
import numpy as np
import logging

try:
    import torch
    import torch.nn.functional as F
    from optuna_system.utils.focal_loss import (
        FocalLoss, ClassBalancedFocalLoss, FocalLossWithOHEM,
        AdaptiveFocalLoss, RegimeAwareFocalLoss, CryptoTradingFocalLoss,
        compute_class_counts, get_crypto_trading_loss
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


logging.basicConfig(level=logging.INFO)


@unittest.skipIf(not HAS_TORCH, "PyTorch not available")
class TestFocalLoss(unittest.TestCase):
    """Test standard Focal Loss"""
    
    def setUp(self):
        # Create synthetic balanced data
        self.batch_size = 100
        self.num_classes = 3
        
        # Synthetic logits (N, C)
        torch.manual_seed(42)
        self.logits = torch.randn(self.batch_size, self.num_classes)
        
        # Synthetic labels (N,)
        self.labels = torch.randint(0, self.num_classes, (self.batch_size,))
    
    def test_focal_loss_basic(self):
        """Test basic Focal Loss computation"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        loss = focal_loss(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_focal_loss_gamma_zero(self):
        """Test Focal Loss with gamma=0 equals Cross Entropy"""
        focal_loss = FocalLoss(alpha=1.0, gamma=0.0)
        focal_result = focal_loss(self.logits, self.labels)
        
        ce_loss = F.cross_entropy(self.logits, self.labels)
        
        # Should be approximately equal
        self.assertAlmostEqual(focal_result.item(), ce_loss.item(), places=4)
    
    def test_focal_loss_alpha_list(self):
        """Test Focal Loss with per-class alpha"""
        alpha = [0.2, 0.5, 0.3]
        focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
        loss = focal_loss(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_focal_loss_reduction(self):
        """Test different reduction modes"""
        focal_loss_mean = FocalLoss(gamma=2.0, reduction='mean')
        focal_loss_sum = FocalLoss(gamma=2.0, reduction='sum')
        focal_loss_none = FocalLoss(gamma=2.0, reduction='none')
        
        loss_mean = focal_loss_mean(self.logits, self.labels)
        loss_sum = focal_loss_sum(self.logits, self.labels)
        loss_none = focal_loss_none(self.logits, self.labels)
        
        self.assertEqual(loss_mean.shape, torch.Size([]))
        self.assertEqual(loss_sum.shape, torch.Size([]))
        self.assertEqual(loss_none.shape, torch.Size([self.batch_size]))


@unittest.skipIf(not HAS_TORCH, "PyTorch not available")
class TestClassBalancedFocalLoss(unittest.TestCase):
    """Test Class-Balanced Focal Loss"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        # Imbalanced data: [sell, hold, buy] = [120, 580, 150]
        self.class_counts = [120, 580, 150]
        
        # Generate synthetic imbalanced data
        labels_list = []
        labels_list.extend([0] * self.class_counts[0])
        labels_list.extend([1] * self.class_counts[1])
        labels_list.extend([2] * self.class_counts[2])
        np.random.shuffle(labels_list)
        
        self.labels = torch.tensor(labels_list, dtype=torch.long)
        self.logits = torch.randn(len(labels_list), 3)
    
    def test_cb_focal_loss_basic(self):
        """Test CB Focal Loss computation"""
        cb_focal = ClassBalancedFocalLoss(self.class_counts, beta=0.9999, gamma=2.0)
        loss = cb_focal(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_cb_weights_calculation(self):
        """Test effective number calculation"""
        beta = 0.9999
        cb_focal = ClassBalancedFocalLoss(self.class_counts, beta=beta, gamma=2.0)
        
        # Check that weights are inversely proportional to class frequency
        weights = cb_focal.cb_weights.numpy()
        
        # Minority classes should have higher weights
        self.assertGreater(weights[0], weights[1])  # sell > hold
        self.assertGreater(weights[2], weights[1])  # buy > hold
    
    def test_cb_focal_vs_standard(self):
        """Test CB Focal gives different loss than standard Focal"""
        cb_focal = ClassBalancedFocalLoss(self.class_counts, beta=0.9999, gamma=2.0)
        standard_focal = FocalLoss(alpha=0.25, gamma=2.0)
        
        cb_loss = cb_focal(self.logits, self.labels)
        standard_loss = standard_focal(self.logits, self.labels)
        
        # Should be different due to class balancing
        self.assertNotAlmostEqual(cb_loss.item(), standard_loss.item(), places=2)


@unittest.skipIf(not HAS_TORCH, "PyTorch not available")
class TestFocalLossWithOHEM(unittest.TestCase):
    """Test Focal Loss with Online Hard Example Mining"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 200
        self.logits = torch.randn(self.batch_size, 3)
        self.labels = torch.randint(0, 3, (self.batch_size,))
    
    def test_ohem_focal_loss_basic(self):
        """Test OHEM Focal Loss computation"""
        ohem_focal = FocalLossWithOHEM(alpha=0.25, gamma=2.0, ohem_ratio=0.7)
        loss = ohem_focal(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_ohem_selects_hard_examples(self):
        """Test that OHEM selects hard examples"""
        ohem_ratio = 0.5
        ohem_focal = FocalLossWithOHEM(alpha=0.25, gamma=2.0, ohem_ratio=ohem_ratio)
        
        # Compute per-sample loss
        focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        per_sample_loss = focal_loss_fn(self.logits, self.labels)
        
        # OHEM should select top 50% hardest
        num_hard = int(ohem_ratio * self.batch_size)
        _, top_indices = torch.topk(per_sample_loss, num_hard)
        
        # Compute OHEM loss
        ohem_loss = ohem_focal(self.logits, self.labels)
        
        # Manual OHEM
        manual_ohem_loss = per_sample_loss[top_indices].mean()
        
        # Should be approximately equal
        self.assertAlmostEqual(ohem_loss.item(), manual_ohem_loss.item(), places=4)


@unittest.skipIf(not HAS_TORCH, "PyTorch not available")
class TestAdaptiveFocalLoss(unittest.TestCase):
    """Test Adaptive Focal Loss"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 100
        self.logits = torch.randn(self.batch_size, 3)
        self.labels = torch.randint(0, 3, (self.batch_size,))
    
    def test_adaptive_focal_loss_basic(self):
        """Test Adaptive Focal Loss computation"""
        adaptive_focal = AdaptiveFocalLoss(gamma_init=2.0)
        loss = adaptive_focal(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_gamma_update_overconfident(self):
        """Test gamma increases when model is overconfident"""
        adaptive_focal = AdaptiveFocalLoss(gamma_init=2.0, adaptation_rate=0.1)
        
        initial_gamma = adaptive_focal.gamma
        
        # Model is overconfident on train (val_conf < train_conf)
        adaptive_focal.update_gamma(val_confidence=0.7, train_confidence=0.9)
        
        # Gamma should increase
        self.assertGreater(adaptive_focal.gamma, initial_gamma)
    
    def test_gamma_update_underconfident(self):
        """Test gamma decreases when model is underconfident"""
        adaptive_focal = AdaptiveFocalLoss(gamma_init=2.0, adaptation_rate=0.1)
        
        initial_gamma = adaptive_focal.gamma
        
        # Model is underconfident on val (val_conf > train_conf)
        adaptive_focal.update_gamma(val_confidence=0.9, train_confidence=0.7)
        
        # Gamma should decrease
        self.assertLess(adaptive_focal.gamma, initial_gamma)
    
    def test_gamma_clipping(self):
        """Test gamma is clipped to valid range"""
        adaptive_focal = AdaptiveFocalLoss(
            gamma_init=2.0, gamma_min=0.5, gamma_max=4.0, adaptation_rate=0.5
        )
        
        # Try to push gamma below min
        for _ in range(10):
            adaptive_focal.update_gamma(val_confidence=0.9, train_confidence=0.5)
        
        self.assertGreaterEqual(adaptive_focal.gamma, 0.5)
        
        # Reset and try to push gamma above max
        adaptive_focal.gamma = 2.0
        for _ in range(10):
            adaptive_focal.update_gamma(val_confidence=0.5, train_confidence=0.9)
        
        self.assertLessEqual(adaptive_focal.gamma, 4.0)
    
    def test_stage_gamma(self):
        """Test stage-aware gamma recommendations"""
        adaptive_focal = AdaptiveFocalLoss(gamma_init=2.0)
        
        total_epochs = 100
        
        # Early stage (10% progress)
        early_gamma = adaptive_focal.get_stage_gamma(10, total_epochs)
        self.assertGreater(early_gamma, 0.5)
        self.assertLess(early_gamma, 1.5)
        
        # Mid stage (50% progress)
        mid_gamma = adaptive_focal.get_stage_gamma(50, total_epochs)
        self.assertGreater(mid_gamma, 1.5)
        self.assertLess(mid_gamma, 3.5)
        
        # Late stage (90% progress)
        late_gamma = adaptive_focal.get_stage_gamma(90, total_epochs)
        self.assertGreater(late_gamma, 1.0)
        self.assertLess(late_gamma, 2.5)


@unittest.skipIf(not HAS_TORCH, "PyTorch not available")
class TestRegimeAwareFocalLoss(unittest.TestCase):
    """Test Regime-Aware Focal Loss"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 100
        self.logits = torch.randn(self.batch_size, 3)
        self.labels = torch.randint(0, 3, (self.batch_size,))
    
    def test_regime_aware_basic(self):
        """Test Regime-Aware Focal Loss computation"""
        regime_focal = RegimeAwareFocalLoss(gamma=2.0)
        loss = regime_focal(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_regime_switching(self):
        """Test loss changes when regime changes"""
        regime_focal = RegimeAwareFocalLoss(gamma=2.0)
        
        # Bull market
        regime_focal.set_regime('bull')
        bull_loss = regime_focal(self.logits, self.labels)
        
        # Bear market
        regime_focal.set_regime('bear')
        bear_loss = regime_focal(self.logits, self.labels)
        
        # Sideways
        regime_focal.set_regime('sideways')
        sideways_loss = regime_focal(self.logits, self.labels)
        
        # Losses should be different
        self.assertNotAlmostEqual(bull_loss.item(), bear_loss.item(), places=2)
        self.assertNotAlmostEqual(bull_loss.item(), sideways_loss.item(), places=2)
    
    def test_unknown_regime(self):
        """Test handling of unknown regime"""
        regime_focal = RegimeAwareFocalLoss(gamma=2.0)
        
        # Set unknown regime, should default to sideways
        regime_focal.set_regime('unknown_regime')
        self.assertEqual(regime_focal.current_regime, 'sideways')


@unittest.skipIf(not HAS_TORCH, "PyTorch not available")
class TestCryptoTradingFocalLoss(unittest.TestCase):
    """Test unified Crypto Trading Focal Loss"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        # Realistic imbalanced crypto data
        self.class_counts = [120, 580, 150]  # [sell, hold, buy]
        
        # Generate synthetic data
        labels_list = []
        labels_list.extend([0] * self.class_counts[0])
        labels_list.extend([1] * self.class_counts[1])
        labels_list.extend([2] * self.class_counts[2])
        np.random.shuffle(labels_list)
        
        self.labels = torch.tensor(labels_list, dtype=torch.long)
        self.logits = torch.randn(len(labels_list), 3)
    
    def test_crypto_trading_focal_basic(self):
        """Test unified crypto trading focal loss"""
        crypto_focal = CryptoTradingFocalLoss(
            self.class_counts, beta=0.9999, gamma_init=2.0, ohem_ratio=0.7
        )
        loss = crypto_focal(self.logits, self.labels)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
    
    def test_crypto_trading_with_regime(self):
        """Test crypto trading focal with regime awareness"""
        crypto_focal = CryptoTradingFocalLoss(
            self.class_counts, enable_regime_aware=True
        )
        
        # Test different regimes
        regimes = ['bull', 'bear', 'sideways', 'high_vol']
        losses = []
        
        for regime in regimes:
            crypto_focal.set_regime(regime)
            loss = crypto_focal(self.logits, self.labels)
            losses.append(loss.item())
        
        # Losses should vary by regime
        self.assertGreater(len(set(np.round(losses, 2))), 1)
    
    def test_crypto_trading_gamma_update(self):
        """Test adaptive gamma in crypto trading focal"""
        crypto_focal = CryptoTradingFocalLoss(self.class_counts, gamma_init=2.0)
        
        initial_gamma = crypto_focal.gamma
        
        # Update gamma
        crypto_focal.update_gamma(val_confidence=0.7, train_confidence=0.9)
        
        # Gamma should have changed
        self.assertNotEqual(crypto_focal.gamma, initial_gamma)
    
    def test_crypto_trading_without_regime(self):
        """Test crypto trading focal without regime awareness"""
        crypto_focal = CryptoTradingFocalLoss(
            self.class_counts, enable_regime_aware=False
        )
        
        loss = crypto_focal(self.logits, self.labels)
        self.assertIsInstance(loss.item(), float)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_compute_class_counts(self):
        """Test class count computation"""
        labels = np.array([0, 1, 2, 1, 1, 0, 2, 1, 1])
        counts = compute_class_counts(labels)
        
        self.assertEqual(counts, [2, 5, 2])
    
    @unittest.skipIf(not HAS_TORCH, "PyTorch not available")
    def test_get_crypto_trading_loss(self):
        """Test factory function"""
        class_counts = [120, 580, 150]
        loss_fn = get_crypto_trading_loss(class_counts, regime='bull')
        
        self.assertIsInstance(loss_fn, CryptoTradingFocalLoss)
        self.assertEqual(loss_fn.current_regime, 'bull')


if __name__ == '__main__':
    unittest.main()

