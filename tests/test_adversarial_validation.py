"""
对抗性验证模块测试

测试 optuna_system/utils/adversarial_validation.py

作者: Optuna System Team
日期: 2025-10-31
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# 导入待测试的模块
from optuna_system.utils.adversarial_validation import (
    AdversarialValidator,
    quick_adversarial_check,
    detect_covariate_shift
)


@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # 训练集（正态分布）
    X_train = pd.DataFrame(
        np.random.normal(0, 1, (n_samples, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 测试集（轻微偏移）
    X_test = pd.DataFrame(
        np.random.normal(0.2, 1.1, (n_samples // 2, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    return X_train, X_test


@pytest.fixture
def identical_data():
    """创建相同分布的数据"""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.normal(0, 1, (n_samples, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    X_test = pd.DataFrame(
        np.random.normal(0, 1, (n_samples // 2, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    return X_train, X_test


@pytest.fixture
def severe_shift_data():
    """创建严重分布偏移的数据"""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X_train = pd.DataFrame(
        np.random.normal(0, 1, (n_samples, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 严重偏移
    X_test = pd.DataFrame(
        np.random.normal(2, 2, (n_samples // 2, n_features)),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    return X_train, X_test


class TestAdversarialValidator:
    """测试 AdversarialValidator 类"""
    
    def test_initialization(self):
        """测试初始化"""
        validator = AdversarialValidator(
            model_type='rf',
            n_cv_splits=5,
            random_state=42
        )
        
        assert validator.model_type == 'rf'
        assert validator.n_cv_splits == 5
        assert validator.random_state == 42
        assert validator.results_ is None
    
    def test_invalid_model_type(self):
        """测试无效的模型类型"""
        # 无效的model_type在初始化时就会抛出错误
        with pytest.raises(ValueError, match="Unknown model_type"):
            validator = AdversarialValidator(model_type='invalid')
    
    def test_validate_basic(self, sample_data):
        """测试基本验证功能"""
        X_train, X_test = sample_data
        
        validator = AdversarialValidator(model_type='rf', n_cv_splits=3)
        result = validator.validate(X_train, X_test)
        
        # 检查返回的键
        assert 'train_test_auc' in result
        assert 'cv_auc_mean' in result
        assert 'cv_auc_std' in result
        assert 'distribution_shift' in result
        assert 'overfitting_risk' in result
        
        # 检查AUC值在合理范围内
        assert 0 <= result['cv_auc_mean'] <= 1
        assert result['cv_auc_std'] >= 0
        
        # 检查分布偏移评估
        assert result['distribution_shift'] in ['minimal', 'mild', 'moderate', 'severe']
    
    def test_identical_distributions(self, identical_data):
        """测试相同分布的数据（应该低AUC）"""
        X_train, X_test = identical_data
        
        validator = AdversarialValidator(model_type='rf', n_cv_splits=3)
        result = validator.validate(X_train, X_test)
        
        # 相同分布应该AUC接近0.5
        assert result['cv_auc_mean'] < 0.6  # 允许一些随机性
        assert result['distribution_shift'] in ['minimal', 'mild']
    
    def test_severe_distribution_shift(self, severe_shift_data):
        """测试严重分布偏移（应该高AUC）"""
        X_train, X_test = severe_shift_data
        
        validator = AdversarialValidator(model_type='rf', n_cv_splits=3)
        result = validator.validate(X_train, X_test)
        
        # 严重偏移应该AUC较高
        assert result['cv_auc_mean'] > 0.65
        assert result['distribution_shift'] in ['moderate', 'severe']
    
    def test_feature_importance_extraction(self, sample_data):
        """测试特征重要性提取"""
        X_train, X_test = sample_data
        
        validator = AdversarialValidator(model_type='rf', n_cv_splits=3)
        result = validator.validate(X_train, X_test)
        
        # 检查特征重要性
        assert 'top_discriminative_features' in result
        assert len(result['top_discriminative_features']) > 0
        
        # 检查特征重要性报告
        report = validator.get_feature_importance_report(top_n=5)
        assert len(report) <= 5
        assert 'feature' in report.columns
        assert 'importance' in report.columns
    
    def test_different_model_types(self, sample_data):
        """测试不同的模型类型"""
        X_train, X_test = sample_data
        
        for model_type in ['rf', 'gb', 'lr']:
            validator = AdversarialValidator(model_type=model_type, n_cv_splits=3)
            result = validator.validate(X_train, X_test)
            
            assert 'cv_auc_mean' in result
            assert 0 <= result['cv_auc_mean'] <= 1
    
    def test_empty_data(self):
        """测试空数据"""
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        
        validator = AdversarialValidator()
        
        # 空数据应该抛出错误或返回空结果
        with pytest.raises(Exception):  # 可能是ValueError或其他错误
            validator.validate(X_train, X_test)
    
    def test_mismatched_features(self):
        """测试特征不匹配的数据"""
        np.random.seed(42)
        
        X_train = pd.DataFrame(
            np.random.normal(0, 1, (100, 5)),
            columns=['a', 'b', 'c', 'd', 'e']
        )
        
        X_test = pd.DataFrame(
            np.random.normal(0, 1, (50, 5)),
            columns=['a', 'b', 'c', 'x', 'y']  # 不同的特征名
        )
        
        validator = AdversarialValidator(n_cv_splits=3)
        result = validator.validate(X_train, X_test)
        
        # 应该使用共同特征
        assert result['n_features'] == 3  # 只有 a, b, c 共同


class TestQuickAdversarialCheck:
    """测试快速检查函数"""
    
    def test_quick_check_basic(self, sample_data):
        """测试快速检查基本功能"""
        X_train, X_test = sample_data
        
        result = quick_adversarial_check(X_train, X_test)
        
        assert 'cv_auc_mean' in result
        assert 'distribution_shift' in result
        assert 0 <= result['cv_auc_mean'] <= 1
    
    def test_quick_check_different_models(self, sample_data):
        """测试不同模型类型的快速检查"""
        X_train, X_test = sample_data
        
        for model_type in ['rf', 'gb', 'lr']:
            result = quick_adversarial_check(X_train, X_test, model_type=model_type)
            assert 'cv_auc_mean' in result


class TestCovariateShift:
    """测试协变量偏移检测"""
    
    def test_detect_covariate_shift_basic(self):
        """测试基本的协变量偏移检测"""
        np.random.seed(42)
        
        # 创建按时间变化的数据
        monthly_features = []
        for i in range(6):
            # 逐渐增加均值（模拟分布漂移）
            data = pd.DataFrame(
                np.random.normal(i * 0.3, 1, (200, 5)),
                columns=[f'feature_{j}' for j in range(5)]
            )
            monthly_features.append(data)
        
        results = detect_covariate_shift(monthly_features, window_size=3)
        
        # 应该检测到多个窗口
        assert len(results) > 0
        
        # 每个结果应该有必要的键
        for result in results:
            assert 'cv_auc_mean' in result
            assert 'distribution_shift' in result
            assert 'time_window_start' in result
            assert 'time_window_end' in result
    
    def test_detect_covariate_shift_insufficient_data(self):
        """测试数据不足的情况"""
        monthly_features = [
            pd.DataFrame(np.random.normal(0, 1, (100, 5)))
            for _ in range(3)
        ]
        
        # window_size=5但只有3个时间段
        with pytest.raises(ValueError, match="需要至少"):
            detect_covariate_shift(monthly_features, window_size=5)
    
    def test_detect_covariate_shift_no_drift(self):
        """测试无漂移的情况"""
        np.random.seed(42)
        
        # 创建无漂移的数据
        monthly_features = []
        for i in range(6):
            data = pd.DataFrame(
                np.random.normal(0, 1, (200, 5)),
                columns=[f'feature_{j}' for j in range(5)]
            )
            monthly_features.append(data)
        
        results = detect_covariate_shift(monthly_features, window_size=3)
        
        # 无漂移应该AUC较低
        for result in results:
            assert result['cv_auc_mean'] < 0.65  # 大多数应该是minimal或mild


class TestRiskAssessment:
    """测试风险评估"""
    
    def test_distribution_shift_levels(self):
        """测试分布偏移等级评估"""
        validator = AdversarialValidator()
        
        # 测试不同AUC对应的风险等级
        assert validator._assess_distribution_shift(0.50) == 'minimal'
        assert validator._assess_distribution_shift(0.60) == 'mild'
        assert validator._assess_distribution_shift(0.70) == 'moderate'
        assert validator._assess_distribution_shift(0.80) == 'severe'
    
    def test_overfitting_risk_levels(self):
        """测试过拟合风险等级"""
        validator = AdversarialValidator()
        
        # 测试不同AUC对应的风险等级
        assert validator._assess_overfitting_risk(0.50, 0.52) == 'low'
        assert validator._assess_overfitting_risk(0.60, 0.62) == 'medium'
        assert validator._assess_overfitting_risk(0.70, 0.72) == 'high'
        assert validator._assess_overfitting_risk(0.80, 0.82) == 'very_high'


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

