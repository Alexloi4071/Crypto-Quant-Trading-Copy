"""
模型集成框架 (Model Ensemble)

实现多种模型范式的集成，提高预测稳健性并降低单一模型的系统性偏差。

集成方法:
1. Voting（投票集成）
2. Stacking（堆叠集成）
3. Blending（混合集成）
4. Diversity-based Ensemble（多样性集成）

基于学术文献:
- Wolpert, D. H. (1992): "Stacked Generalization"
- Breiman, L. (1996): "Bagging Predictors"
- Dietterich, T. G. (2000): "Ensemble Methods in Machine Learning"
- Rokach, L. (2010): "Ensemble-based Classifiers"

作者: Optuna System Team
日期: 2025-10-31
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

logger = logging.getLogger(__name__)


class DiverseModelEnsemble:
    """
    多样化模型集成器
    
    集成不同范式的模型:
    - 树模型: Random Forest, LightGBM, XGBoost, CatBoost
    - 线性模型: Logistic Regression
    - 非线性模型: SVM
    
    目的: 降低单一模型的偏差，提高泛化能力
    """
    
    def __init__(self,
                 ensemble_method: str = 'stacking',
                 use_tree_models: bool = True,
                 use_linear_models: bool = True,
                 use_svm: bool = False,
                 n_cv_folds: int = 5,
                 random_state: int = 42):
        """
        初始化集成器
        
        Args:
            ensemble_method: 集成方法 ('voting', 'stacking', 'blending')
            use_tree_models: 是否使用树模型
            use_linear_models: 是否使用线性模型
            use_svm: 是否使用SVM（计算慢）
            n_cv_folds: 交叉验证折数
            random_state: 随机种子
        """
        self.ensemble_method = ensemble_method
        self.use_tree_models = use_tree_models
        self.use_linear_models = use_linear_models
        self.use_svm = use_svm
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.logger = logger
        
        # 存储模型
        self.base_models_ = None
        self.ensemble_model_ = None
        self.base_predictions_ = None
        self.diversity_scores_ = None
    
    def _create_base_models(self) -> List[Tuple[str, Any]]:
        """创建基础模型列表"""
        models = []
        
        # 树模型
        if self.use_tree_models:
            # Random Forest
            models.append((
                'rf',
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            ))
            
            # LightGBM
            if HAS_LGBM:
                models.append((
                    'lgbm',
                    LGBMClassifier(
                        n_estimators=100,
                        max_depth=8,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbose=-1
                    )
                ))
            
            # XGBoost
            if HAS_XGB:
                models.append((
                    'xgb',
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=8,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        verbosity=0
                    )
                ))
            
            # CatBoost
            if HAS_CATBOOST:
                models.append((
                    'catboost',
                    CatBoostClassifier(
                        iterations=100,
                        depth=6,
                        learning_rate=0.05,
                        random_state=self.random_state,
                        verbose=False
                    )
                ))
            
            # Gradient Boosting
            models.append((
                'gb',
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=self.random_state
                )
            ))
        
        # 线性模型
        if self.use_linear_models:
            models.append((
                'lr',
                LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            ))
        
        # SVM
        if self.use_svm:
            models.append((
                'svm',
                SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.random_state
                )
            ))
        
        self.logger.info(f"  🤖 创建了{len(models)}个基础模型: {[name for name, _ in models]}")
        
        return models
    
    def fit(self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'DiverseModelEnsemble':
        """
        训练集成模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（Blending需要）
            y_val: 验证标签（Blending需要）
            
        Returns:
            self
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 训练{self.ensemble_method.upper()}集成模型...")
        
        # 创建基础模型
        self.base_models_ = self._create_base_models()
        
        if self.ensemble_method == 'voting':
            self._fit_voting(X_train, y_train)
        
        elif self.ensemble_method == 'stacking':
            self._fit_stacking(X_train, y_train)
        
        elif self.ensemble_method == 'blending':
            if X_val is None or y_val is None:
                raise ValueError("Blending需要提供验证集")
            self._fit_blending(X_train, y_train, X_val, y_val)
        
        else:
            raise ValueError(f"Unknown ensemble_method: {self.ensemble_method}")
        
        self.logger.info("  ✅ 集成模型训练完成")
        
        return self
    
    def _fit_voting(self, X_train: pd.DataFrame, y_train: pd.Series):
        """训练投票集成"""
        self.logger.info("  📊 使用Voting集成（软投票）...")
        
        self.ensemble_model_ = VotingClassifier(
            estimators=self.base_models_,
            voting='soft',
            n_jobs=-1
        )
        
        self.ensemble_model_.fit(X_train, y_train)
    
    def _fit_stacking(self, X_train: pd.DataFrame, y_train: pd.Series):
        """训练堆叠集成"""
        self.logger.info("  📊 使用Stacking集成（元学习器: Logistic Regression）...")
        
        self.ensemble_model_ = StackingClassifier(
            estimators=self.base_models_,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
            cv=self.n_cv_folds,
            n_jobs=-1
        )
        
        self.ensemble_model_.fit(X_train, y_train)
    
    def _fit_blending(self,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series):
        """训练混合集成"""
        self.logger.info("  📊 使用Blending集成...")
        
        # 1. 在训练集上训练基础模型
        trained_models = []
        val_predictions = []
        
        for name, model in self.base_models_:
            self.logger.info(f"     训练 {name}...")
            model.fit(X_train, y_train)
            trained_models.append((name, model))
            
            # 在验证集上预测
            val_pred = model.predict_proba(X_val)
            val_predictions.append(val_pred)
        
        # 2. 使用验证集预测训练元学习器
        val_predictions_stacked = np.hstack(val_predictions)
        
        meta_learner = LogisticRegression(max_iter=1000, random_state=self.random_state)
        meta_learner.fit(val_predictions_stacked, y_val)
        
        self.ensemble_model_ = {
            'base_models': trained_models,
            'meta_learner': meta_learner
        }
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """预测类别"""
        if self.ensemble_method == 'blending':
            # Blending需要特殊处理
            base_predictions = []
            for name, model in self.ensemble_model_['base_models']:
                pred = model.predict_proba(X_test)
                base_predictions.append(pred)
            
            stacked_predictions = np.hstack(base_predictions)
            return self.ensemble_model_['meta_learner'].predict(stacked_predictions)
        
        else:
            return self.ensemble_model_.predict(X_test)
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """预测概率"""
        if self.ensemble_method == 'blending':
            base_predictions = []
            for name, model in self.ensemble_model_['base_models']:
                pred = model.predict_proba(X_test)
                base_predictions.append(pred)
            
            stacked_predictions = np.hstack(base_predictions)
            return self.ensemble_model_['meta_learner'].predict_proba(stacked_predictions)
        
        else:
            return self.ensemble_model_.predict_proba(X_test)
    
    def evaluate(self,
                X_test: pd.DataFrame,
                y_test: pd.Series,
                metrics: Optional[List[str]] = None) -> Dict:
        """
        评估集成模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            metrics: 评估指标列表
            
        Returns:
            评估结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("📊 评估集成模型性能...")
        
        # 预测
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # 计算指标
        if metrics is None:
            metrics = ['accuracy', 'f1_macro', 'roc_auc']
        
        results = {}
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_test, y_pred)
        
        if 'f1_macro' in metrics:
            results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        
        if 'roc_auc' in metrics:
            if y_pred_proba.shape[1] == 2:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        
        # 打印结果
        self.logger.info("  🎯 集成模型性能:")
        for metric, value in results.items():
            self.logger.info(f"     {metric}: {value:.4f}")
        
        # 评估基础模型（用于比较）
        results['base_models'] = self._evaluate_base_models(X_test, y_test)
        
        # 计算多样性
        results['diversity'] = self._calculate_diversity(X_test, y_test)
        
        return results
    
    def _evaluate_base_models(self,
                             X_test: pd.DataFrame,
                             y_test: pd.Series) -> Dict:
        """评估各个基础模型"""
        self.logger.info("  📊 评估基础模型...")
        
        base_results = {}
        
        if self.ensemble_method == 'blending':
            models = self.ensemble_model_['base_models']
        else:
            # Voting/Stacking的基础模型已训练
            models = [(name, estimator) for name, estimator in self.ensemble_model_.named_estimators_.items()]
        
        for name, model in models:
            y_pred = model.predict(X_test)
            
            base_results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro')
            }
            
            self.logger.info(f"     {name}: F1={base_results[name]['f1_macro']:.4f}, Acc={base_results[name]['accuracy']:.4f}")
        
        return base_results
    
    def _calculate_diversity(self,
                            X_test: pd.DataFrame,
                            y_test: pd.Series) -> Dict:
        """
        计算模型多样性
        
        使用Q统计量和不一致度量
        """
        self.logger.info("  🔍 计算模型多样性...")
        
        # 获取各个模型的预测
        if self.ensemble_method == 'blending':
            models = self.ensemble_model_['base_models']
        else:
            models = [(name, estimator) for name, estimator in self.ensemble_model_.named_estimators_.items()]
        
        predictions = {}
        for name, model in models:
            predictions[name] = model.predict(X_test)
        
        # 计算Q统计量（两两模型之间）
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        q_statistics = []
        disagreement_counts = []
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                name_i = model_names[i]
                name_j = model_names[j]
                
                pred_i = predictions[name_i]
                pred_j = predictions[name_j]
                
                # Q统计量
                n11 = np.sum((pred_i == y_test) & (pred_j == y_test))
                n00 = np.sum((pred_i != y_test) & (pred_j != y_test))
                n10 = np.sum((pred_i == y_test) & (pred_j != y_test))
                n01 = np.sum((pred_i != y_test) & (pred_j == y_test))
                
                if (n11 * n00 + n10 * n01) > 0:
                    q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                else:
                    q = 0
                
                q_statistics.append(q)
                
                # 不一致度
                disagreement = np.sum(pred_i != pred_j) / len(y_test)
                disagreement_counts.append(disagreement)
        
        avg_q = np.mean(q_statistics) if q_statistics else 0
        avg_disagreement = np.mean(disagreement_counts) if disagreement_counts else 0
        
        self.logger.info(f"     平均Q统计量: {avg_q:.4f} (越小越好，<0表示高多样性)")
        self.logger.info(f"     平均不一致度: {avg_disagreement:.4f} (越大越好)")
        
        diversity_assessment = 'high' if avg_q < 0.2 and avg_disagreement > 0.3 else 'medium' if avg_q < 0.5 else 'low'
        
        self.logger.info(f"     多样性评估: {diversity_assessment.upper()}")
        
        return {
            'avg_q_statistic': avg_q,
            'avg_disagreement': avg_disagreement,
            'diversity_assessment': diversity_assessment,
            'q_statistics': q_statistics,
            'disagreement_counts': disagreement_counts
        }
    
    def get_model_weights(self) -> Optional[Dict]:
        """获取模型权重（如果适用）"""
        if self.ensemble_method == 'stacking':
            # Stacking的元学习器权重
            if hasattr(self.ensemble_model_.final_estimator_, 'coef_'):
                weights = self.ensemble_model_.final_estimator_.coef_[0]
                model_names = [name for name, _ in self.base_models_]
                
                weight_dict = {name: float(weight) for name, weight in zip(model_names, weights)}
                
                self.logger.info("  ⚖️  Stacking模型权重:")
                for name, weight in weight_dict.items():
                    self.logger.info(f"     {name}: {weight:.4f}")
                
                return weight_dict
        
        return None


if __name__ == '__main__':
    # 简单测试
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("模型集成模块测试")
    
    # 生成数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建集成器
    ensemble = DiverseModelEnsemble(
        ensemble_method='stacking',
        use_tree_models=True,
        use_linear_models=True,
        use_svm=False
    )
    
    # 训练
    ensemble.fit(X_train, y_train)
    
    # 评估
    results = ensemble.evaluate(X_test, y_test)
    
    print(f"\n✅ 测试完成")
    print(f"   集成F1: {results['f1_macro']:.4f}")
    print(f"   多样性: {results['diversity']['diversity_assessment']}")

