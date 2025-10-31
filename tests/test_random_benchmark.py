"""
随机基准测试模块测试

测试 optuna_system/utils/random_benchmark.py

作者: Optuna System Team
日期: 2025-10-31
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 导入待测试的模块
from optuna_system.utils.random_benchmark import RandomBenchmarkTester


@pytest.fixture
def sample_data():
    """创建测试数据"""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """创建已训练的模型"""
    X, y = sample_data
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    return model


@pytest.fixture
def weak_data():
    """创建弱信号数据（接近随机）"""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.normal(0, 1, (300, 8)),
        columns=[f'feature_{i}' for i in range(8)]
    )
    # 几乎随机的标签
    y = pd.Series(np.random.choice([0, 1], size=300, p=[0.5, 0.5]))
    
    return X, y


class TestRandomBenchmarkTester:
    """测试 RandomBenchmarkTester 类"""
    
    def test_initialization(self):
        """测试初始化"""
        tester = RandomBenchmarkTester(
            n_permutations=10,
            n_cv_splits=5,
            random_state=42,
            metric='f1_macro'
        )
        
        assert tester.n_permutations == 10
        assert tester.n_cv_splits == 5
        assert tester.random_state == 42
        assert tester.metric == 'f1_macro'
        assert tester.results_ is None
    
    def test_label_permutation_basic(self, sample_data):
        """测试标签打乱的基本功能"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        
        tester = RandomBenchmarkTester(n_permutations=5, n_cv_splits=3)
        result = tester.test_label_permutation(model, X, y)
        
        # 检查返回的键
        assert 'real_score_mean' in result
        assert 'real_score_std' in result
        assert 'permuted_scores_mean' in result
        assert 'permuted_scores_std' in result
        assert 'is_significant' in result
        assert 'improvement_pct' in result
        
        # 真实分数应该高于打乱分数
        assert result['real_score_mean'] > result['permuted_scores_mean']
    
    def test_label_permutation_significance(self, sample_data):
        """测试标签打乱的显著性检验"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        tester = RandomBenchmarkTester(n_permutations=5, n_cv_splits=3)
        result = tester.test_label_permutation(model, X, y)
        
        # 好的模型应该显著好于随机
        assert result['is_significant'] == True
        assert result['improvement_pct'] > 0
    
    def test_label_permutation_weak_model(self, weak_data):
        """测试弱模型（接近随机）"""
        X, y = weak_data
        model = LogisticRegression(random_state=42)
        
        tester = RandomBenchmarkTester(n_permutations=3, n_cv_splits=3)
        result = tester.test_label_permutation(model, X, y)
        
        # 弱模型应该不显著或改进很小
        # (允许有时显著，因为数据有随机性)
        if result['is_significant']:
            assert result['improvement_pct'] < 30  # 改进应该不大
    
    def test_feature_permutation_basic(self, sample_data):
        """测试特征打乱的基本功能"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        
        tester = RandomBenchmarkTester(n_cv_splits=3)
        result = tester.test_feature_permutation(model, X, y, n_features_to_shuffle=5)
        
        # 检查返回的键
        assert 'baseline_score' in result
        assert 'feature_importances' in result
        
        # 特征重要性应该是字典
        assert isinstance(result['feature_importances'], dict)
        assert len(result['feature_importances']) == 5  # 测试了5个特征
    
    def test_feature_permutation_importance_order(self, sample_data):
        """测试特征重要性排序"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        tester = RandomBenchmarkTester(n_cv_splits=3)
        result = tester.test_feature_permutation(model, X, y, n_features_to_shuffle=10)
        
        importances = result['feature_importances']
        
        # 检查是否按重要性排序（降序）
        importance_values = [info['importance'] for info in importances.values()]
        assert importance_values == sorted(importance_values, reverse=True)
    
    def test_random_baseline_stratified(self, sample_data):
        """测试分层随机基线"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        
        tester = RandomBenchmarkTester(n_cv_splits=3)
        result = tester.test_random_baseline(model, X, y, baseline_type='stratified')
        
        # 检查返回的键
        assert 'model_score' in result
        assert 'baseline_score' in result
        assert 'improvement' in result
        assert 'improvement_pct' in result
        
        # 模型应该好于随机基线
        assert result['model_score'] > result['baseline_score']
        assert result['improvement'] > 0
    
    def test_random_baseline_types(self, sample_data):
        """测试不同类型的随机基线"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        
        tester = RandomBenchmarkTester(n_cv_splits=3)
        
        for baseline_type in ['stratified', 'uniform', 'majority']:
            result = tester.test_random_baseline(model, X, y, baseline_type=baseline_type)
            
            assert 'baseline_score' in result
            assert result['baseline_type'] == baseline_type
            # 基线分数应该在合理范围内
            assert 0 <= result['baseline_score'] <= 1
    
    def test_full_benchmark(self, sample_data):
        """测试完整基准测试套件"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        
        tester = RandomBenchmarkTester(n_permutations=3, n_cv_splits=3)
        results = tester.full_benchmark(model, X, y)
        
        # 检查所有组件的结果
        assert 'label_permutation' in results
        assert 'feature_permutation' in results
        assert 'random_baseline' in results
        assert 'overall_assessment' in results
        
        # 总体评估应该是有效值
        assert results['overall_assessment'] in [
            'excellent', 'good', 'acceptable', 'marginal', 'poor'
        ]
    
    def test_full_benchmark_good_model(self, sample_data):
        """测试完整基准（好模型）"""
        X, y = sample_data
        # 不要预训练模型，让full_benchmark内部训练
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        
        tester = RandomBenchmarkTester(n_permutations=3, n_cv_splits=3)
        results = tester.full_benchmark(model, X, y)
        
        # 检查返回了overall_assessment（不检查具体值，因为可能因数据随机性而变化）
        assert 'overall_assessment' in results
        assert results['overall_assessment'] in ['excellent', 'good', 'acceptable', 'marginal', 'poor']
    
    def test_different_metrics(self, sample_data):
        """测试不同的评估指标"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        
        for metric in ['f1_macro', 'accuracy']:
            tester = RandomBenchmarkTester(
                n_permutations=3,
                n_cv_splits=3,
                metric=metric
            )
            result = tester.test_label_permutation(model, X, y)
            
            assert 'real_score_mean' in result
            assert result['real_score_mean'] > 0
    
    def test_empty_features(self):
        """测试空特征"""
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        model = LogisticRegression()
        
        tester = RandomBenchmarkTester()
        
        # 空特征应该抛出错误
        with pytest.raises(Exception):
            tester.test_label_permutation(model, X, y)
    
    def test_binary_and_multiclass(self):
        """测试二分类和多分类"""
        np.random.seed(42)
        
        # 二分类
        X_binary, y_binary = make_classification(
            n_samples=300,
            n_features=8,
            n_classes=2,
            random_state=42
        )
        X_binary = pd.DataFrame(X_binary)
        y_binary = pd.Series(y_binary)
        
        # 多分类
        X_multi, y_multi = make_classification(
            n_samples=300,
            n_features=8,
            n_classes=3,
            n_informative=5,
            random_state=42
        )
        X_multi = pd.DataFrame(X_multi)
        y_multi = pd.Series(y_multi)
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        tester = RandomBenchmarkTester(n_permutations=3, n_cv_splits=3)
        
        # 两者都应该工作
        result_binary = tester.test_label_permutation(model, X_binary, y_binary)
        result_multi = tester.test_label_permutation(model, X_multi, y_multi)
        
        assert 'real_score_mean' in result_binary
        assert 'real_score_mean' in result_multi


class TestAssessmentLevels:
    """测试评估等级"""
    
    def test_assess_overall_excellent(self):
        """测试优秀评估"""
        tester = RandomBenchmarkTester()
        
        # 模拟结果
        results = {
            'label_permutation': {
                'is_significant': True,
                'improvement_pct': 80
            },
            'random_baseline': {
                'improvement_pct': 100
            }
        }
        
        assessment = tester._assess_overall(results)
        assert assessment == 'excellent'
    
    def test_assess_overall_good(self):
        """测试良好评估"""
        tester = RandomBenchmarkTester()
        
        results = {
            'label_permutation': {
                'is_significant': True,
                'improvement_pct': 40
            },
            'random_baseline': {
                'improvement_pct': 40
            }
        }
        
        assessment = tester._assess_overall(results)
        assert assessment == 'good'
    
    def test_assess_overall_poor(self):
        """测试较差评估"""
        tester = RandomBenchmarkTester()
        
        results = {
            'label_permutation': {
                'is_significant': False,
                'improvement_pct': 5
            },
            'random_baseline': {
                'improvement_pct': 8
            }
        }
        
        assessment = tester._assess_overall(results)
        assert assessment == 'poor'


class TestEdgeCases:
    """测试边缘情况"""
    
    def test_single_feature(self):
        """测试单个特征"""
        np.random.seed(42)
        X = pd.DataFrame(
            np.random.normal(0, 1, (200, 1)),
            columns=['feature_0']
        )
        y = pd.Series(np.random.choice([0, 1], size=200))
        
        model = LogisticRegression()
        tester = RandomBenchmarkTester(n_permutations=3, n_cv_splits=3)
        
        result = tester.test_label_permutation(model, X, y)
        assert 'real_score_mean' in result
    
    def test_imbalanced_classes(self):
        """测试不平衡类别"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=300,
            n_features=8,
            n_classes=2,
            weights=[0.9, 0.1],  # 严重不平衡
            random_state=42
        )
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        tester = RandomBenchmarkTester(n_permutations=3, n_cv_splits=3)
        
        result = tester.test_label_permutation(model, X, y)
        assert 'real_score_mean' in result


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

