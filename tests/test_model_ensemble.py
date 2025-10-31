"""
模型集成框架测试

测试 optuna_system/utils/model_ensemble.py

作者: Optuna System Team
日期: 2025-10-31
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 导入待测试的模块
from optuna_system.utils.model_ensemble import DiverseModelEnsemble


@pytest.fixture
def sample_data():
    """创建测试数据"""
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def binary_imbalanced_data():
    """创建不平衡二分类数据"""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


class TestDiverseModelEnsemble:
    """测试 DiverseModelEnsemble 类"""
    
    def test_initialization(self):
        """测试初始化"""
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False,
            random_state=42
        )
        
        assert ensemble.ensemble_method == 'stacking'
        assert ensemble.use_tree_models == True
        assert ensemble.use_linear_models == True
        assert ensemble.use_svm == False
        assert ensemble.random_state == 42
        assert ensemble.ensemble_model_ is None
    
    def test_create_base_models_tree_only(self):
        """测试仅创建树模型"""
        ensemble = DiverseModelEnsemble(
            use_tree_models=True,
            use_linear_models=False,
            use_svm=False
        )
        
        models = ensemble._create_base_models()
        
        # 应该有至少2个树模型（RF + GB）
        assert len(models) >= 2
        
        # 检查模型名称
        model_names = [name for name, _ in models]
        assert 'rf' in model_names
        assert 'gb' in model_names
    
    def test_create_base_models_all(self):
        """测试创建所有类型的模型"""
        ensemble = DiverseModelEnsemble(
            use_tree_models=True,
            use_linear_models=True,
            use_svm=True
        )
        
        models = ensemble._create_base_models()
        
        # 应该有多种模型
        assert len(models) >= 3
        
        model_names = [name for name, _ in models]
        # 至少应该有树模型和线性模型
        assert any('rf' in name or 'gb' in name or 'lgbm' in name for name in model_names)
        assert 'lr' in model_names
        assert 'svm' in model_names
    
    def test_voting_ensemble(self, sample_data):
        """测试投票集成"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='voting',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False,
            n_cv_folds=3
        )
        
        ensemble.fit(X_train, y_train)
        
        # 检查模型已训练
        assert ensemble.ensemble_model_ is not None
        
        # 测试预测
        y_pred = ensemble.predict(X_test)
        assert len(y_pred) == len(y_test)
        
        # 测试概率预测
        y_pred_proba = ensemble.predict_proba(X_test)
        assert y_pred_proba.shape[0] == len(y_test)
        assert y_pred_proba.shape[1] == 2  # 二分类
    
    def test_stacking_ensemble(self, sample_data):
        """测试堆叠集成"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False,
            n_cv_folds=3
        )
        
        ensemble.fit(X_train, y_train)
        
        # 检查模型已训练
        assert ensemble.ensemble_model_ is not None
        
        # 测试预测
        y_pred = ensemble.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})
    
    def test_blending_ensemble(self, sample_data):
        """测试混合集成"""
        X_train, X_test, y_train, y_test = sample_data
        
        # 分割训练集为训练+验证
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='blending',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False
        )
        
        ensemble.fit(X_train_sub, y_train_sub, X_val, y_val)
        
        # 检查模型已训练
        assert ensemble.ensemble_model_ is not None
        assert 'base_models' in ensemble.ensemble_model_
        assert 'meta_learner' in ensemble.ensemble_model_
        
        # 测试预测
        y_pred = ensemble.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_blending_without_validation_set(self, sample_data):
        """测试Blending需要验证集"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(ensemble_method='blending')
        
        # 没有提供验证集应该抛出错误
        with pytest.raises(ValueError, match="Blending需要提供验证集"):
            ensemble.fit(X_train, y_train)
    
    def test_evaluate(self, sample_data):
        """测试评估功能"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False,
            n_cv_folds=3
        )
        
        ensemble.fit(X_train, y_train)
        results = ensemble.evaluate(X_test, y_test)
        
        # 检查返回的键
        assert 'accuracy' in results
        assert 'f1_macro' in results
        assert 'roc_auc' in results
        assert 'base_models' in results
        assert 'diversity' in results
        
        # 检查指标在合理范围内
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['f1_macro'] <= 1
        assert 0 <= results['roc_auc'] <= 1
    
    def test_diversity_calculation(self, sample_data):
        """测试多样性计算"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='voting',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False
        )
        
        ensemble.fit(X_train, y_train)
        results = ensemble.evaluate(X_test, y_test)
        
        # 检查多样性指标
        diversity = results['diversity']
        assert 'avg_q_statistic' in diversity
        assert 'avg_disagreement' in diversity
        assert 'diversity_assessment' in diversity
        
        # 多样性评估应该有效
        assert diversity['diversity_assessment'] in ['high', 'medium', 'low']
    
    def test_get_model_weights_stacking(self, sample_data):
        """测试获取Stacking模型权重"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False
        )
        
        ensemble.fit(X_train, y_train)
        weights = ensemble.get_model_weights()
        
        # Stacking应该有权重
        if weights is not None:
            assert isinstance(weights, dict)
            assert len(weights) > 0
    
    def test_get_model_weights_voting(self, sample_data):
        """测试Voting集成的权重（应该返回None）"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='voting',
            use_tree_models=True,
            use_linear_models=False
        )
        
        ensemble.fit(X_train, y_train)
        weights = ensemble.get_model_weights()
        
        # Voting不应该有权重
        assert weights is None
    
    def test_invalid_ensemble_method(self, sample_data):
        """测试无效的集成方法"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(ensemble_method='invalid')
        
        with pytest.raises(ValueError, match="Unknown ensemble_method"):
            ensemble.fit(X_train, y_train)
    
    def test_imbalanced_data(self, binary_imbalanced_data):
        """测试不平衡数据"""
        X_train, X_test, y_train, y_test = binary_imbalanced_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False,
            n_cv_folds=3
        )
        
        ensemble.fit(X_train, y_train)
        results = ensemble.evaluate(X_test, y_test)
        
        # 应该能处理不平衡数据
        assert 'f1_macro' in results
        assert results['f1_macro'] > 0


class TestBaseModelsEvaluation:
    """测试基础模型评估"""
    
    def test_base_models_performance(self, sample_data):
        """测试基础模型性能"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False
        )
        
        ensemble.fit(X_train, y_train)
        results = ensemble.evaluate(X_test, y_test)
        
        base_results = results['base_models']
        
        # 每个基础模型都应该有性能指标
        for model_name, metrics in base_results.items():
            assert 'accuracy' in metrics
            assert 'f1_macro' in metrics
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['f1_macro'] <= 1


class TestEdgeCases:
    """测试边缘情况"""
    
    def test_small_dataset(self):
        """测试小数据集"""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='voting',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False,
            n_cv_folds=2  # 减少CV折数
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        assert len(y_pred) == len(y_test)
    
    def test_single_model_type(self, sample_data):
        """测试只使用一种模型类型"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='voting',
            use_tree_models=False,
            use_linear_models=True,  # 只使用线性模型
            use_svm=False
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        assert len(y_pred) == len(y_test)
    
    def test_perfect_separation(self):
        """测试完美可分的数据"""
        np.random.seed(42)
        # 创建线性可分的数据
        X = np.vstack([
            np.random.normal(0, 0.5, (100, 2)),
            np.random.normal(3, 0.5, (100, 2))
        ])
        y = np.hstack([np.zeros(100), np.ones(100)])
        
        X = pd.DataFrame(X, columns=['feature_0', 'feature_1'])
        y = pd.Series(y, dtype=int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='voting',
            use_tree_models=True,
            use_linear_models=True,
            use_svm=False
        )
        
        ensemble.fit(X_train, y_train)
        results = ensemble.evaluate(X_test, y_test)
        
        # 完美可分应该有很高的准确率
        assert results['accuracy'] > 0.9


class TestPredictionMethods:
    """测试预测方法"""
    
    def test_predict_vs_predict_proba_consistency(self, sample_data):
        """测试predict和predict_proba的一致性"""
        X_train, X_test, y_train, y_test = sample_data
        
        ensemble = DiverseModelEnsemble(
            ensemble_method='stacking',
            use_tree_models=True,
            use_linear_models=False
        )
        
        ensemble.fit(X_train, y_train)
        
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)
        
        # predict应该是predict_proba的argmax
        y_pred_from_proba = np.argmax(y_pred_proba, axis=1)
        np.testing.assert_array_equal(y_pred, y_pred_from_proba)


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])

