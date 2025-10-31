"""
模型可解释性模块测试

测试 optuna_system/utils/model_interpretability.py

作者: Optuna System Team
日期: 2025-10-31
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 导入待测试的模块
from optuna_system.utils.model_interpretability import ModelInterpreter

# 检查可选依赖
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


@pytest.fixture
def sample_data():
    """创建测试数据"""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_rf_model(sample_data):
    """创建已训练的随机森林模型"""
    X_train, X_test, y_train, y_test = sample_data
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test


@pytest.fixture
def trained_lr_model(sample_data):
    """创建已训练的逻辑回归模型"""
    X_train, X_test, y_train, y_test = sample_data
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_test


class TestModelInterpreter:
    """测试 ModelInterpreter 类"""
    
    def test_initialization(self, trained_rf_model):
        """测试初始化"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(
            model=model,
            X_train=X_train,
            random_state=42
        )
        
        assert interpreter.model is not None
        assert len(interpreter.feature_names) == 10
        assert interpreter.random_state == 42
        assert interpreter.shap_values_ is None
    
    @pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed")
    def test_shap_explanation_tree(self, trained_rf_model):
        """测试SHAP解释（树模型）"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_shap(X_test, method='tree')
        
        # 检查返回的键
        assert 'shap_values' in result
        assert 'global_importance' in result
        assert 'method' in result
        
        # 检查全局重要性
        importance_df = result['global_importance']
        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    @pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed")
    def test_shap_explanation_auto(self, trained_rf_model):
        """测试SHAP自动选择方法"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_shap(X_test, method='auto')
        
        # 应该自动选择tree方法
        assert result['method'] == 'tree'
        assert 'global_importance' in result
    
    @pytest.mark.skipif(not HAS_LIME, reason="LIME not installed")
    def test_lime_explanation(self, trained_rf_model):
        """测试LIME解释"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_lime(X_test, instance_idx=0, n_features=5)
        
        # 检查返回的键
        assert 'explanation' in result
        assert 'feature_weights' in result
        assert 'instance_idx' in result
        assert 'prediction' in result
        
        # 检查特征权重
        assert len(result['feature_weights']) == 5
    
    def test_permutation_importance(self, trained_rf_model):
        """测试排列重要性"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_permutation(
            X_test, y_test,
            n_repeats=5,
            scoring='f1_macro'
        )
        
        # 检查返回的键
        assert 'importance_df' in result
        assert 'raw_importances' in result
        
        # 检查重要性DataFrame
        importance_df = result['importance_df']
        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance_mean' in importance_df.columns
        assert 'importance_std' in importance_df.columns
    
    def test_comprehensive_interpretation(self, trained_rf_model):
        """测试全面解释（不含SHAP/LIME）"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        results = interpreter.comprehensive_interpretation(X_test, y_test)
        
        # 排列重要性应该总是存在
        assert 'permutation' in results
        
        # SHAP只在安装时存在
        if HAS_SHAP:
            assert 'shap' in results
    
    @pytest.mark.skipif(not HAS_SHAP, reason="SHAP not installed")
    def test_consistency_check(self, trained_rf_model):
        """测试特征重要性一致性检查"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        results = interpreter.comprehensive_interpretation(X_test, y_test)
        
        # 应该有一致性检查
        assert 'consistency' in results
        
        consistency = results['consistency']
        assert 'overlap_features' in consistency
        assert 'overlap_pct' in consistency
        assert 'consistency_level' in consistency
        
        # 一致性等级应该有效
        assert consistency['consistency_level'] in ['high', 'medium', 'low']
    
    def test_different_model_types(self, trained_lr_model):
        """测试不同模型类型"""
        model, X_train, X_test, y_test = trained_lr_model
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_permutation(X_test, y_test, n_repeats=3)
        
        # 排列重要性应该适用于所有模型
        assert 'importance_df' in result
    
    def test_empty_test_data(self, trained_rf_model):
        """测试空测试数据"""
        model, X_train, X_test, y_test = trained_rf_model
        
        X_empty = pd.DataFrame(columns=X_train.columns)
        
        interpreter = ModelInterpreter(model, X_train)
        
        # 空数据应该失败或返回空结果
        # (具体行为取决于实现)
        try:
            result = interpreter.explain_with_permutation(X_empty, pd.Series([]))
            # 如果没有抛出错误，检查结果
            assert result == {} or 'importance_df' in result
        except (ValueError, IndexError):
            # 预期的错误
            pass
    
    def test_feature_names_consistency(self, trained_rf_model):
        """测试特征名称一致性"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_permutation(X_test, y_test)
        
        # 特征名称应该匹配
        importance_features = set(result['importance_df']['feature'])
        train_features = set(X_train.columns)
        
        assert importance_features == train_features


class TestWithoutOptionalDependencies:
    """测试无可选依赖时的行为"""
    
    def test_shap_without_installation(self, trained_rf_model):
        """测试SHAP未安装时的行为"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        
        if not HAS_SHAP:
            result = interpreter.explain_with_shap(X_test)
            # 应该返回空字典或错误信息
            assert result == {}
    
    def test_lime_without_installation(self, trained_rf_model):
        """测试LIME未安装时的行为"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        
        if not HAS_LIME:
            result = interpreter.explain_with_lime(X_test, instance_idx=0)
            # 应该返回空字典或错误信息
            assert result == {}
    
    def test_comprehensive_without_optional(self, trained_rf_model):
        """测试无可选依赖时的全面解释"""
        model, X_train, X_test, y_test = trained_rf_model
        
        interpreter = ModelInterpreter(model, X_train)
        results = interpreter.comprehensive_interpretation(X_test, y_test)
        
        # 排列重要性应该总是可用
        assert 'permutation' in results


class TestEdgeCases:
    """测试边缘情况"""
    
    def test_single_feature(self):
        """测试单个特征"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=1,
            n_informative=1,
            n_redundant=0,
            n_clusters_per_class=1,  # 单特征需要降低clusters
            random_state=42
        )
        X = pd.DataFrame(X, columns=['feature_0'])
        y = pd.Series(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_permutation(X_test, y_test)
        
        assert len(result['importance_df']) == 1
    
    def test_many_features(self):
        """测试大量特征"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=200,
            n_features=50,
            n_informative=10,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(50)])
        y = pd.Series(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        
        interpreter = ModelInterpreter(model, X_train)
        result = interpreter.explain_with_permutation(X_test, y_test, n_repeats=3)
        
        assert len(result['importance_df']) == 50


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

