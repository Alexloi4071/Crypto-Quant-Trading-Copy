# -*- coding: utf-8 -*-
"""
测试Financial-First目标函数（阶段10）
"""

import unittest
import numpy as np
import pandas as pd
from optuna_system.utils.financial_objectives import (
    FinancialMetricsCalculator,
    FinancialFirstObjective,
    create_default_financial_objective,
    compare_old_vs_new_objectives
)


class TestFinancialMetricsCalculator(unittest.TestCase):
    """测试金融指标计算器"""
    
    def setUp(self):
        self.calculator = FinancialMetricsCalculator(risk_free_rate=0.02)
        
        # 创建模拟收益率数据
        np.random.seed(42)
        # 策略1：高收益高风险
        self.returns_high_risk = pd.Series(np.random.normal(0.001, 0.02, 1000))
        # 策略2：稳健收益
        self.returns_stable = pd.Series(np.random.normal(0.0005, 0.005, 1000))
        # 策略3：亏损策略
        self.returns_losing = pd.Series(np.random.normal(-0.0005, 0.01, 1000))
    
    def test_sharpe_ratio_calculation(self):
        """测试Sharpe Ratio计算"""
        # 高风险策略
        sharpe_high = self.calculator.calculate_sharpe_ratio(self.returns_high_risk)
        self.assertIsInstance(sharpe_high, float)
        self.assertGreater(sharpe_high, 0)  # 正收益应该有正Sharpe
        
        # 稳健策略
        sharpe_stable = self.calculator.calculate_sharpe_ratio(self.returns_stable)
        self.assertGreater(sharpe_stable, 0)
        
        # 亏损策略
        sharpe_losing = self.calculator.calculate_sharpe_ratio(self.returns_losing)
        self.assertLess(sharpe_losing, 0)  # 负收益应该有负Sharpe
        
        print(f"Sharpe - High Risk: {sharpe_high:.4f}")
        print(f"Sharpe - Stable: {sharpe_stable:.4f}")
        print(f"Sharpe - Losing: {sharpe_losing:.4f}")
    
    def test_sortino_ratio_calculation(self):
        """测试Sortino Ratio计算"""
        sortino_high = self.calculator.calculate_sortino_ratio(self.returns_high_risk)
        sortino_stable = self.calculator.calculate_sortino_ratio(self.returns_stable)
        
        self.assertIsInstance(sortino_high, float)
        self.assertIsInstance(sortino_stable, float)
        
        # Sortino通常高于Sharpe（因为只考虑下行风险）
        sharpe_high = self.calculator.calculate_sharpe_ratio(self.returns_high_risk)
        self.assertGreaterEqual(sortino_high, sharpe_high * 0.8)  # 允许一定误差
        
        print(f"Sortino - High Risk: {sortino_high:.4f}")
        print(f"Sortino - Stable: {sortino_stable:.4f}")
    
    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        # 创建有明显回撤的数据
        returns_with_dd = pd.Series([0.05, 0.03, -0.10, -0.15, -0.05, 0.10, 0.08])
        max_dd = self.calculator.calculate_max_drawdown(returns_with_dd)
        
        self.assertIsInstance(max_dd, float)
        self.assertGreaterEqual(max_dd, 0)  # 回撤应该是正数
        self.assertLessEqual(max_dd, 1)  # 回撤不超过100%
        
        # 检查回撤大于0（因为有负收益）
        self.assertGreater(max_dd, 0.05)
        
        print(f"Max Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    
    def test_calmar_ratio_calculation(self):
        """测试Calmar Ratio计算"""
        calmar_high = self.calculator.calculate_calmar_ratio(self.returns_high_risk)
        calmar_stable = self.calculator.calculate_calmar_ratio(self.returns_stable)
        
        self.assertIsInstance(calmar_high, float)
        self.assertIsInstance(calmar_stable, float)
        
        print(f"Calmar - High Risk: {calmar_high:.4f}")
        print(f"Calmar - Stable: {calmar_stable:.4f}")
    
    def test_win_rate_calculation(self):
        """测试胜率计算"""
        # 60%胜率的策略
        returns_60pct = pd.Series([0.01] * 60 + [-0.005] * 40)
        win_rate = self.calculator.calculate_win_rate(returns_60pct)
        
        self.assertAlmostEqual(win_rate, 0.6, places=2)
        
        # 测试真实数据
        wr_high = self.calculator.calculate_win_rate(self.returns_high_risk)
        wr_stable = self.calculator.calculate_win_rate(self.returns_stable)
        
        self.assertGreaterEqual(wr_high, 0)
        self.assertLessEqual(wr_high, 1)
        
        print(f"Win Rate - 60%: {win_rate:.4f}")
        print(f"Win Rate - High Risk: {wr_high:.4f}")
        print(f"Win Rate - Stable: {wr_stable:.4f}")
    
    def test_profit_factor_calculation(self):
        """测试盈亏比计算"""
        # 盈亏比2:1的策略
        returns_2to1 = pd.Series([0.02] * 50 + [-0.01] * 50)
        pf = self.calculator.calculate_profit_factor(returns_2to1)
        
        self.assertAlmostEqual(pf, 2.0, places=1)
        
        # 测试真实数据
        pf_high = self.calculator.calculate_profit_factor(self.returns_high_risk)
        
        self.assertGreater(pf_high, 0)
        
        print(f"Profit Factor - 2:1: {pf:.4f}")
        print(f"Profit Factor - High Risk: {pf_high:.4f}")
    
    def test_all_metrics(self):
        """测试计算所有指标"""
        metrics = self.calculator.calculate_all_metrics(self.returns_stable)
        
        # 检查所有指标都存在
        required_metrics = [
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
            'calmar_ratio', 'win_rate', 'profit_factor',
            'total_return', 'volatility'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        print("\n所有指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


class TestFinancialFirstObjective(unittest.TestCase):
    """测试Financial-First目标函数"""
    
    def setUp(self):
        self.objective = FinancialFirstObjective(risk_free_rate=0.02)
        
        # 创建模拟数据
        np.random.seed(42)
        self.returns_good = pd.Series(np.random.normal(0.001, 0.005, 1000))
        self.returns_bad = pd.Series(np.random.normal(-0.0005, 0.02, 1000))
        
        # ML指标
        self.ml_metrics_good = {
            'f1_macro': 0.75,
            'accuracy': 0.80,
            'auc_macro': 0.85
        }
        self.ml_metrics_bad = {
            'f1_macro': 0.55,
            'accuracy': 0.60,
            'auc_macro': 0.65
        }
    
    def test_objective_initialization(self):
        """测试目标函数初始化"""
        self.assertIsInstance(self.objective, FinancialFirstObjective)
        self.assertEqual(self.objective.risk_free_rate, 0.02)
        
        # 检查权重
        total_weight = sum(self.objective.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
        # 检查金融指标权重 > ML指标权重
        financial_weight = sum([
            self.objective.weights['sharpe_ratio'],
            self.objective.weights['max_drawdown'],
            self.objective.weights['calmar_ratio'],
            self.objective.weights['sortino_ratio'],
            self.objective.weights['win_rate'],
            self.objective.weights['profit_factor']
        ])
        ml_weight = self.objective.weights['f1_macro']
        
        self.assertGreater(financial_weight, 0.95)  # 金融指标 > 95%
        self.assertLess(ml_weight, 0.05)  # ML指标 < 5%
        
        print(f"金融指标权重: {financial_weight:.2%}")
        print(f"ML指标权重: {ml_weight:.2%}")
    
    def test_score_calculation(self):
        """测试得分计算"""
        # 好策略
        score_good = self.objective.calculate_score(self.returns_good, self.ml_metrics_good)
        
        # 差策略
        score_bad = self.objective.calculate_score(self.returns_bad, self.ml_metrics_bad)
        
        # 好策略得分应该更高
        self.assertGreater(score_good, score_bad)
        
        # 得分应该在[0, 1]范围
        self.assertGreaterEqual(score_good, 0)
        self.assertLessEqual(score_good, 1)
        self.assertGreaterEqual(score_bad, 0)
        self.assertLessEqual(score_bad, 1)
        
        print(f"好策略得分: {score_good:.4f}")
        print(f"差策略得分: {score_bad:.4f}")
    
    def test_component_scores(self):
        """测试组成部分得分"""
        components = self.objective.get_component_scores(self.returns_good, self.ml_metrics_good)
        
        # 检查所有组件都存在
        required_components = [
            'sharpe_component', 'maxdd_component', 'calmar_component',
            'sortino_component', 'winrate_component', 'profit_factor_component',
            'f1_component', 'total_score',
            'financial_contribution', 'ml_contribution'
        ]
        
        for component in required_components:
            self.assertIn(component, components)
        
        # 检查金融贡献 > ML贡献
        self.assertGreater(
            components['financial_contribution'],
            components['ml_contribution']
        )
        
        # 检查总分等于各组件之和（允许小误差）
        sum_components = sum([
            components['sharpe_component'],
            components['maxdd_component'],
            components['calmar_component'],
            components['sortino_component'],
            components['winrate_component'],
            components['profit_factor_component'],
            components['f1_component']
        ])
        self.assertAlmostEqual(sum_components, components['total_score'], places=6)
        
        print("\n组成部分得分:")
        for key, value in components.items():
            print(f"  {key}: {value:.4f}")
    
    def test_custom_weights(self):
        """测试自定义权重"""
        custom_weights = {
            'sharpe_ratio': 0.50,
            'max_drawdown': 0.30,
            'calmar_ratio': 0.10,
            'sortino_ratio': 0.05,
            'win_rate': 0.03,
            'profit_factor': 0.01,
            'f1_macro': 0.01
        }
        
        custom_objective = FinancialFirstObjective(weights=custom_weights)
        score = custom_objective.calculate_score(self.returns_good, self.ml_metrics_good)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        print(f"自定义权重得分: {score:.4f}")
    
    def test_financial_first_vs_ml_first(self):
        """测试Financial-First vs ML-First的差异"""
        # 场景1：高F1但低Sharpe
        returns_high_f1_low_sharpe = pd.Series(np.random.normal(0.0001, 0.02, 1000))
        ml_high_f1 = {'f1_macro': 0.90}
        
        # 场景2：低F1但高Sharpe
        returns_low_f1_high_sharpe = pd.Series(np.random.normal(0.002, 0.005, 1000))
        ml_low_f1 = {'f1_macro': 0.60}
        
        score_1 = self.objective.calculate_score(returns_high_f1_low_sharpe, ml_high_f1)
        score_2 = self.objective.calculate_score(returns_low_f1_high_sharpe, ml_low_f1)
        
        # Financial-First应该偏向场景2（高Sharpe）
        self.assertGreater(score_2, score_1)
        
        print(f"高F1低Sharpe得分: {score_1:.4f}")
        print(f"低F1高Sharpe得分: {score_2:.4f}")
        print(f"Financial-First正确选择了高Sharpe策略！")


class TestComparisonFunctions(unittest.TestCase):
    """测试对比函数"""
    
    def test_compare_old_vs_new(self):
        """测试旧目标vs新目标的对比"""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 1000))
        ml_metrics = {
            'f1_weighted': 0.75,
            'f1_macro': 0.72,
            'accuracy': 0.78,
            'auc_macro': 0.80
        }
        
        comparison = compare_old_vs_new_objectives(returns, ml_metrics)
        
        # 检查对比结果包含所有字段
        required_fields = [
            'old_score', 'new_score', 'score_change', 'score_change_pct',
            'old_financial_weight', 'new_financial_weight',
            'old_ml_weight', 'new_ml_weight',
            'component_scores'
        ]
        
        for field in required_fields:
            self.assertIn(field, comparison)
        
        # 检查权重变化
        self.assertEqual(comparison['old_financial_weight'], 0.10)
        self.assertEqual(comparison['new_financial_weight'], 0.98)
        self.assertEqual(comparison['old_ml_weight'], 0.90)
        self.assertEqual(comparison['new_ml_weight'], 0.02)
        
        print("\n旧目标 vs 新目标对比:")
        print(f"  旧得分: {comparison['old_score']:.4f}")
        print(f"  新得分: {comparison['new_score']:.4f}")
        print(f"  得分变化: {comparison['score_change']:+.4f} ({comparison['score_change_pct']:+.2f}%)")
        print(f"  金融权重: {comparison['old_financial_weight']:.0%} → {comparison['new_financial_weight']:.0%}")


class TestEdgeCases(unittest.TestCase):
    """测试边缘情况"""
    
    def setUp(self):
        self.calculator = FinancialMetricsCalculator()
        self.objective = FinancialFirstObjective()
    
    def test_empty_returns(self):
        """测试空收益率序列"""
        empty_returns = pd.Series([])
        metrics = self.calculator.calculate_all_metrics(empty_returns)
        
        # 应该返回默认值而不是崩溃
        self.assertEqual(metrics['sharpe_ratio'], 0.0)
        self.assertEqual(metrics['win_rate'], 0.0)
    
    def test_zero_returns(self):
        """测试全零收益率"""
        zero_returns = pd.Series([0.0] * 100)
        metrics = self.calculator.calculate_all_metrics(zero_returns)
        
        # Sharpe应该为0（无波动或无收益）
        self.assertEqual(metrics['sharpe_ratio'], 0.0)
        self.assertEqual(metrics['win_rate'], 0.0)
    
    def test_constant_positive_returns(self):
        """测试恒定正收益"""
        constant_returns = pd.Series([0.01] * 100)
        metrics = self.calculator.calculate_all_metrics(constant_returns)
        
        # 应该有很高的Sharpe（低波动，稳定收益）
        # 但由于波动率为0，Sharpe会是0或inf
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float))
        
        # Win Rate应该是100%
        self.assertEqual(metrics['win_rate'], 1.0)
    
    def test_extreme_drawdown(self):
        """测试极端回撤"""
        # 巨大回撤
        extreme_dd_returns = pd.Series([0.10, 0.05, -0.80, 0.10, 0.05])
        max_dd = self.calculator.calculate_max_drawdown(extreme_dd_returns)
        
        # 回撤应该接近80%
        self.assertGreater(max_dd, 0.5)
    
    def test_missing_ml_metrics(self):
        """测试缺少ML指标"""
        returns = pd.Series(np.random.normal(0.001, 0.01, 100))
        
        # 不提供ML指标
        score = self.objective.calculate_score(returns, ml_metrics=None)
        
        # 应该正常计算（F1默认为0）
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

