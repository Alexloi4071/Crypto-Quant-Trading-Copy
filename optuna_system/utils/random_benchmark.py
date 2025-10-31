"""
随机基准测试模块 (Random Benchmark Testing)

通过对标签进行随机打乱，验证模型是否真的从数据中学习到模式。
如果打乱标签后性能依然很好，说明模型在过拟合噪声或存在数据泄漏。

基于学术文献:
- Dua, D., & Graff, C. (2017): "UCI Machine Learning Repository"
- Wolpert, D. H., & Macready, W. G. (1997): "No Free Lunch Theorems"
- López de Prado, M. (2018): "Advances in Financial Machine Learning"

作者: Optuna System Team
日期: 2025-10-31
"""

import logging
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)


class RandomBenchmarkTester:
    """
    随机基准测试器
    
    执行以下测试:
    1. 标签随机打乱测试 (Label Permutation Test)
    2. 特征随机打乱测试 (Feature Permutation Test)
    3. 与随机基线对比 (Random Baseline Comparison)
    
    判断标准:
    - 真实性能 >> 随机性能: 模型学到了真实模式 ✅
    - 真实性能 ≈ 随机性能: 模型未学到有效模式 ⚠️
    - 真实性能 < 随机性能: 可能存在Bug或数据泄漏 🚨
    """
    
    def __init__(self,
                 n_permutations: int = 10,
                 n_cv_splits: int = 5,
                 random_state: int = 42,
                 metric: str = 'f1_macro'):
        """
        初始化随机基准测试器
        
        Args:
            n_permutations: 随机打乱次数
            n_cv_splits: 交叉验证折数
            random_state: 随机种子
            metric: 评估指标 ('f1_macro', 'accuracy', 'roc_auc', etc.)
        """
        self.n_permutations = n_permutations
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.metric = metric
        self.logger = logger
        
        # 存储结果
        self.results_ = None
    
    def test_label_permutation(self,
                               model: Any,
                               X: pd.DataFrame,
                               y: pd.Series,
                               cv: Optional[Any] = None) -> Dict:
        """
        标签随机打乱测试
        
        将标签随机打乱多次，训练模型并评估性能。
        如果打乱后性能下降不明显，说明模型可能在拟合噪声。
        
        Args:
            model: 待测试的模型（需实现sklearn接口）
            X: 特征矩阵
            y: 真实标签
            cv: 交叉验证策略（None则使用默认StratifiedKFold）
            
        Returns:
            Dict包含:
                - real_score_mean: 真实标签的平均分数
                - real_score_std: 真实标签的标准差
                - permuted_scores_mean: 打乱标签的平均分数
                - permuted_scores_std: 打乱标签的标准差
                - p_value: 统计显著性p值
                - is_significant: 是否显著好于随机
        """
        self.logger.info("=" * 60)
        self.logger.info("🎲 执行标签随机打乱测试...")
        
        if cv is None:
            cv = StratifiedKFold(
                n_splits=self.n_cv_splits,
                shuffle=True,
                random_state=self.random_state
            )
        
        # 1. 在真实标签上评估
        self.logger.info("  📊 在真实标签上评估...")
        real_scores = cross_val_score(
            clone(model), X, y,
            cv=cv,
            scoring=self.metric,
            n_jobs=-1
        )
        real_score_mean = np.mean(real_scores)
        real_score_std = np.std(real_scores)
        
        self.logger.info(f"     真实分数: {real_score_mean:.4f} ± {real_score_std:.4f}")
        
        # 2. 在打乱标签上评估
        self.logger.info(f"  🔀 执行{self.n_permutations}次标签打乱测试...")
        permuted_scores = []
        
        for i in range(self.n_permutations):
            # 打乱标签
            y_permuted = y.sample(frac=1, random_state=self.random_state + i).reset_index(drop=True)
            
            # 评估
            scores = cross_val_score(
                clone(model), X, y_permuted,
                cv=cv,
                scoring=self.metric,
                n_jobs=-1
            )
            permuted_score = np.mean(scores)
            permuted_scores.append(permuted_score)
            
            self.logger.debug(f"     打乱 {i+1}: {permuted_score:.4f}")
        
        permuted_scores_mean = np.mean(permuted_scores)
        permuted_scores_std = np.std(permuted_scores)
        
        self.logger.info(f"     打乱分数: {permuted_scores_mean:.4f} ± {permuted_scores_std:.4f}")
        
        # 3. 统计显著性检验（Wilcoxon符号秩检验）
        # H0: 真实分数 = 随机分数
        # H1: 真实分数 > 随机分数
        try:
            # 扩展真实分数到相同长度以进行配对检验
            real_scores_extended = np.full(self.n_permutations, real_score_mean)
            statistic, p_value = stats.wilcoxon(
                real_scores_extended,
                permuted_scores,
                alternative='greater'
            )
        except Exception as e:
            self.logger.warning(f"⚠️ 统计检验失败: {e}")
            p_value = None
        
        # 判断是否显著
        is_significant = (p_value is not None and p_value < 0.05)
        
        # 计算效应量（Cohen's d）
        effect_size = (real_score_mean - permuted_scores_mean) / (
            np.sqrt((real_score_std**2 + permuted_scores_std**2) / 2) + 1e-10
        )
        
        result = {
            'real_score_mean': real_score_mean,
            'real_score_std': real_score_std,
            'real_scores': real_scores.tolist(),
            'permuted_scores_mean': permuted_scores_mean,
            'permuted_scores_std': permuted_scores_std,
            'permuted_scores': permuted_scores,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': effect_size,
            'improvement_pct': ((real_score_mean - permuted_scores_mean) / permuted_scores_mean * 100) if permuted_scores_mean > 0 else 0
        }
        
        self._print_permutation_summary(result)
        
        return result
    
    def test_feature_permutation(self,
                                 model: Any,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 n_features_to_shuffle: Optional[int] = None) -> Dict:
        """
        特征随机打乱测试
        
        逐个或批量打乱特征，观察性能下降情况。
        性能下降越多，说明该特征越重要。
        
        Args:
            model: 待测试的模型
            X: 特征矩阵
            y: 标签
            n_features_to_shuffle: 要打乱的特征数量（None则打乱所有）
            
        Returns:
            特征重要性字典
        """
        self.logger.info("=" * 60)
        self.logger.info("🔀 执行特征随机打乱测试...")
        
        # 基线性能（所有特征正常）
        cv = StratifiedKFold(
            n_splits=self.n_cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        baseline_scores = cross_val_score(
            clone(model), X, y,
            cv=cv,
            scoring=self.metric,
            n_jobs=-1
        )
        baseline_score = np.mean(baseline_scores)
        
        self.logger.info(f"  📊 基线分数（所有特征正常）: {baseline_score:.4f}")
        
        # 确定要测试的特征
        if n_features_to_shuffle is None:
            features_to_test = X.columns.tolist()
        else:
            # 随机选择部分特征
            features_to_test = np.random.choice(
                X.columns,
                size=min(n_features_to_shuffle, len(X.columns)),
                replace=False
            ).tolist()
        
        self.logger.info(f"  🔢 测试{len(features_to_test)}个特征...")
        
        feature_importances = {}
        
        for feature in features_to_test:
            # 复制数据并打乱该特征
            X_shuffled = X.copy()
            X_shuffled[feature] = X_shuffled[feature].sample(
                frac=1,
                random_state=self.random_state
            ).reset_index(drop=True)
            
            # 评估性能
            shuffled_scores = cross_val_score(
                clone(model), X_shuffled, y,
                cv=cv,
                scoring=self.metric,
                n_jobs=-1
            )
            shuffled_score = np.mean(shuffled_scores)
            
            # 计算性能下降
            importance = baseline_score - shuffled_score
            feature_importances[feature] = {
                'importance': importance,
                'score_after_shuffle': shuffled_score,
                'score_drop_pct': (importance / baseline_score * 100) if baseline_score > 0 else 0
            }
            
            self.logger.debug(f"     {feature}: -{importance:.4f} ({feature_importances[feature]['score_drop_pct']:.1f}%)")
        
        # 按重要性排序
        sorted_importances = dict(
            sorted(
                feature_importances.items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )
        )
        
        # 打印Top 10
        self.logger.info(f"  🔝 Top 10重要特征:")
        for i, (feat, info) in enumerate(list(sorted_importances.items())[:10]):
            self.logger.info(f"     {i+1}. {feat}: -{info['importance']:.4f} ({info['score_drop_pct']:.1f}%)")
        
        return {
            'baseline_score': baseline_score,
            'feature_importances': sorted_importances
        }
    
    def test_random_baseline(self,
                            model: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            baseline_type: str = 'stratified') -> Dict:
        """
        与随机基线对比
        
        比较模型与简单随机猜测的性能。
        
        Args:
            model: 待测试的模型
            X: 特征矩阵
            y: 标签
            baseline_type: 基线类型 ('stratified', 'uniform', 'majority')
            
        Returns:
            对比结果
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🎯 与随机基线对比 (type={baseline_type})...")
        
        cv = StratifiedKFold(
            n_splits=self.n_cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # 模型性能
        model_scores = cross_val_score(
            clone(model), X, y,
            cv=cv,
            scoring=self.metric,
            n_jobs=-1
        )
        model_score = np.mean(model_scores)
        
        # 计算随机基线
        if baseline_type == 'stratified':
            # 按类别分布随机猜测
            class_distribution = y.value_counts(normalize=True)
            if self.metric in ['f1_macro', 'accuracy']:
                baseline_score = 1.0 / len(class_distribution)  # 简化估计
            else:
                baseline_score = 0.5  # AUC的随机基线
        elif baseline_type == 'uniform':
            # 均匀随机猜测
            n_classes = y.nunique()
            baseline_score = 1.0 / n_classes
        elif baseline_type == 'majority':
            # 始终猜测多数类
            majority_class_pct = y.value_counts().max() / len(y)
            baseline_score = majority_class_pct if self.metric == 'accuracy' else 0.5
        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")
        
        # 计算改进
        improvement = model_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        
        result = {
            'model_score': model_score,
            'baseline_score': baseline_score,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'baseline_type': baseline_type
        }
        
        self._print_baseline_summary(result)
        
        return result
    
    def full_benchmark(self,
                      model: Any,
                      X: pd.DataFrame,
                      y: pd.Series) -> Dict:
        """
        执行完整的随机基准测试套件
        
        包括:
        1. 标签打乱测试
        2. 特征打乱测试（前10个特征）
        3. 随机基线对比
        
        Args:
            model: 待测试的模型
            X: 特征矩阵
            y: 标签
            
        Returns:
            完整测试结果
        """
        self.logger.info("🚀 开始完整随机基准测试套件...")
        
        results = {}
        
        # 1. 标签打乱测试
        results['label_permutation'] = self.test_label_permutation(model, X, y)
        
        # 2. 特征打乱测试
        results['feature_permutation'] = self.test_feature_permutation(
            model, X, y,
            n_features_to_shuffle=min(10, len(X.columns))
        )
        
        # 3. 随机基线对比
        results['random_baseline'] = self.test_random_baseline(model, X, y)
        
        # 综合评估
        results['overall_assessment'] = self._assess_overall(results)
        
        self.results_ = results
        
        self._print_overall_summary()
        
        return results
    
    def _assess_overall(self, results: Dict) -> str:
        """综合评估模型质量"""
        label_perm = results['label_permutation']
        baseline = results['random_baseline']
        
        # 判断标准
        is_better_than_random = label_perm['is_significant']
        improvement_vs_baseline = baseline['improvement_pct']
        
        if is_better_than_random and improvement_vs_baseline > 50:
            return 'excellent'
        elif is_better_than_random and improvement_vs_baseline > 20:
            return 'good'
        elif is_better_than_random and improvement_vs_baseline > 10:
            return 'acceptable'
        elif is_better_than_random:
            return 'marginal'
        else:
            return 'poor'
    
    def _print_permutation_summary(self, result: Dict):
        """打印标签打乱测试总结"""
        self.logger.info("=" * 60)
        self.logger.info("📊 标签打乱测试结果:")
        self.logger.info(f"  真实分数: {result['real_score_mean']:.4f} ± {result['real_score_std']:.4f}")
        self.logger.info(f"  打乱分数: {result['permuted_scores_mean']:.4f} ± {result['permuted_scores_std']:.4f}")
        self.logger.info(f"  改进: +{result['improvement_pct']:.1f}%")
        self.logger.info(f"  效应量: {result['effect_size']:.2f}")
        
        if result['p_value'] is not None:
            self.logger.info(f"  p值: {result['p_value']:.4f}")
        
        if result['is_significant']:
            self.logger.info("  ✅ 模型显著好于随机 (p < 0.05)")
        else:
            self.logger.info("  ⚠️  模型未显著好于随机 (p >= 0.05)")
            self.logger.info("     可能原因: 数据泄漏、过拟合噪声、特征无效")
        
        self.logger.info("=" * 60)
    
    def _print_baseline_summary(self, result: Dict):
        """打印基线对比总结"""
        self.logger.info("=" * 60)
        self.logger.info("📊 随机基线对比结果:")
        self.logger.info(f"  模型分数: {result['model_score']:.4f}")
        self.logger.info(f"  基线分数: {result['baseline_score']:.4f} ({result['baseline_type']})")
        self.logger.info(f"  改进: +{result['improvement_pct']:.1f}%")
        
        if result['improvement_pct'] > 50:
            self.logger.info("  ✅ 模型显著优于基线")
        elif result['improvement_pct'] > 20:
            self.logger.info("  ✅ 模型优于基线")
        elif result['improvement_pct'] > 10:
            self.logger.info("  ⚠️  模型略优于基线")
        else:
            self.logger.info("  🚨 模型仅略胜基线，需要改进")
        
        self.logger.info("=" * 60)
    
    def _print_overall_summary(self):
        """打印整体总结"""
        if self.results_ is None:
            return
        
        assessment = self.results_['overall_assessment']
        
        self.logger.info("=" * 60)
        self.logger.info("🏆 完整基准测试总体评估:")
        
        if assessment == 'excellent':
            self.logger.info("  ✅ 优秀 - 模型学到了强有力的模式")
        elif assessment == 'good':
            self.logger.info("  ✅ 良好 - 模型性能可靠")
        elif assessment == 'acceptable':
            self.logger.info("  ⚠️  可接受 - 模型有一定预测能力")
        elif assessment == 'marginal':
            self.logger.info("  ⚠️  边缘 - 模型预测能力较弱")
        else:  # poor
            self.logger.info("  🚨 较差 - 模型未学到有效模式")
            self.logger.info("     建议: 检查数据质量、特征工程、模型选择")
        
        self.logger.info("=" * 60)


if __name__ == '__main__':
    # 简单测试
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("随机基准测试模块测试")
    
    # 生成模拟数据
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # 创建模型
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    # 执行测试
    tester = RandomBenchmarkTester(n_permutations=5, n_cv_splits=3)
    results = tester.full_benchmark(model, X, y)
    
    print(f"\n✅ 测试完成")
    print(f"   总体评估: {results['overall_assessment'].upper()}")

