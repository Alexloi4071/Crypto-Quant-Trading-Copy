"""
对抗性验证模块 (Adversarial Validation)

用于检测训练集和测试集之间的分布差异，这是系统性偏差的重要指标。

基于学术文献:
- Kaufman, S., Rosset, S., & Perlich, C. (2012): "Leakage in Data Mining: 
  Formulation, Detection, and Avoidance"
- Zhuang, F. et al. (2020): "A Comprehensive Survey on Transfer Learning"

作者: Optuna System Team
日期: 2025-10-31
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AdversarialValidator:
    """
    对抗性验证器
    
    通过训练分类器区分训练集和测试集，评估数据分布差异。
    如果分类器能轻易区分两个集合（AUC远离0.5），说明存在分布偏移。
    
    学术依据:
    - AUC < 0.55: 分布相似，低风险
    - 0.55 <= AUC < 0.65: 轻度偏移，中等风险
    - 0.65 <= AUC < 0.75: 中度偏移，高风险
    - AUC >= 0.75: 严重偏移，极高风险
    """
    
    def __init__(self, 
                 model_type: str = 'rf',
                 n_cv_splits: int = 5,
                 random_state: int = 42):
        """
        初始化对抗性验证器
        
        Args:
            model_type: 分类器类型 ('rf', 'gb', 'lr')
            n_cv_splits: 交叉验证折数
            random_state: 随机种子
        """
        self.model_type = model_type
        self.n_cv_splits = n_cv_splits
        self.random_state = random_state
        self.logger = logger
        
        # 初始化分类器
        self.model = self._get_model()
        self.scaler = StandardScaler()
        
        # 存储结果
        self.results_ = None
        self.feature_importances_ = None
    
    def _get_model(self):
        """获取分类器"""
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == 'lr':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def validate(self,
                 train_features: pd.DataFrame,
                 test_features: pd.DataFrame,
                 sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        执行对抗性验证
        
        Args:
            train_features: 训练集特征
            test_features: 测试集特征
            sample_weight: 样本权重（可选）
            
        Returns:
            Dict包含:
                - train_test_auc: 训练/测试AUC
                - cv_auc_mean: 交叉验证平均AUC
                - cv_auc_std: 交叉验证AUC标准差
                - distribution_shift: 分布偏移程度
                - overfitting_risk: 过拟合风险等级
                - top_discriminative_features: 最具区分性的特征
        """
        self.logger.info("=" * 60)
        self.logger.info("🔍 执行对抗性验证...")
        
        # 数据准备
        X_train = train_features.copy()
        X_test = test_features.copy()
        
        # 确保特征一致
        common_features = list(set(X_train.columns) & set(X_test.columns))
        if len(common_features) < len(X_train.columns):
            self.logger.warning(f"⚠️ 特征不一致: train有{len(X_train.columns)}个, test有{len(X_test.columns)}个, 共同{len(common_features)}个")
        
        X_train = X_train[common_features]
        X_test = X_test[common_features]
        
        # 创建标签（0=train, 1=test）
        y_train = np.zeros(len(X_train))
        y_test = np.ones(len(X_test))
        
        # 合并数据
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_combined = np.concatenate([y_train, y_test])
        
        # 标准化（对于逻辑回归很重要）
        if self.model_type == 'lr':
            X_combined_scaled = self.scaler.fit_transform(X_combined)
            X_combined = pd.DataFrame(
                X_combined_scaled,
                columns=X_combined.columns,
                index=X_combined.index
            )
        
        # 交叉验证
        self.logger.info(f"  📊 开始{self.n_cv_splits}折交叉验证...")
        cv_aucs = []
        cv_accs = []
        
        skf = StratifiedKFold(
            n_splits=self.n_cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
            X_fold_train = X_combined.iloc[train_idx]
            y_fold_train = y_combined[train_idx]
            X_fold_val = X_combined.iloc[val_idx]
            y_fold_val = y_combined[val_idx]
            
            # 训练模型
            model = self._get_model()
            model.fit(X_fold_train, y_fold_train)
            
            # 预测
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = model.predict(X_fold_val)
            
            # 评估
            fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
            fold_acc = accuracy_score(y_fold_val, y_pred)
            
            cv_aucs.append(fold_auc)
            cv_accs.append(fold_acc)
            
            self.logger.debug(f"    Fold {fold+1}: AUC={fold_auc:.4f}, Acc={fold_acc:.4f}")
        
        cv_auc_mean = np.mean(cv_aucs)
        cv_auc_std = np.std(cv_aucs)
        
        # 在全部数据上训练（用于特征重要性）
        self.logger.info("  📈 在全部数据上训练最终模型...")
        self.model.fit(X_combined, y_combined)
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = pd.Series(
                self.model.feature_importances_,
                index=common_features
            ).sort_values(ascending=False)
            
            top_features = self.feature_importances_.head(10)
            self.logger.info(f"  🔝 Top 10区分性特征:")
            for feat, imp in top_features.items():
                self.logger.info(f"     {feat}: {imp:.4f}")
        
        # 在测试集上预测（诊断用）
        y_pred_proba = self.model.predict_proba(X_combined)[:, 1]
        train_test_auc = roc_auc_score(y_combined, y_pred_proba)
        
        # 评估分布偏移
        distribution_shift = self._assess_distribution_shift(cv_auc_mean)
        overfitting_risk = self._assess_overfitting_risk(cv_auc_mean, train_test_auc)
        
        # 存储结果
        self.results_ = {
            'train_test_auc': train_test_auc,
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_aucs': cv_aucs,
            'distribution_shift': distribution_shift,
            'overfitting_risk': overfitting_risk,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(common_features),
            'top_discriminative_features': top_features.to_dict() if hasattr(self.model, 'feature_importances_') else {}
        }
        
        # 打印总结
        self._print_summary()
        
        return self.results_
    
    def _assess_distribution_shift(self, auc: float) -> str:
        """评估分布偏移程度"""
        if auc < 0.55:
            return 'minimal'
        elif auc < 0.65:
            return 'mild'
        elif auc < 0.75:
            return 'moderate'
        else:
            return 'severe'
    
    def _assess_overfitting_risk(self, cv_auc: float, train_test_auc: float) -> str:
        """评估过拟合风险"""
        if cv_auc < 0.55:
            return 'low'
        elif cv_auc < 0.65:
            return 'medium'
        elif cv_auc < 0.75:
            return 'high'
        else:
            return 'very_high'
    
    def _print_summary(self):
        """打印验证总结"""
        if self.results_ is None:
            return
        
        r = self.results_
        
        self.logger.info("=" * 60)
        self.logger.info("📊 对抗性验证结果:")
        self.logger.info(f"  🎯 交叉验证AUC: {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f}")
        self.logger.info(f"  📈 训练/测试AUC: {r['train_test_auc']:.4f}")
        self.logger.info(f"  📏 训练样本: {r['n_train_samples']:,}")
        self.logger.info(f"  📏 测试样本: {r['n_test_samples']:,}")
        self.logger.info(f"  🔢 特征数量: {r['n_features']}")
        
        # 分布偏移评估
        shift = r['distribution_shift']
        if shift == 'minimal':
            self.logger.info(f"  ✅ 分布偏移: {shift.upper()} (AUC < 0.55)")
            self.logger.info("     低风险，训练集和测试集分布相似")
        elif shift == 'mild':
            self.logger.info(f"  ⚠️  分布偏移: {shift.upper()} (0.55 <= AUC < 0.65)")
            self.logger.info("     中等风险，存在轻微分布差异")
        elif shift == 'moderate':
            self.logger.info(f"  ⚠️  分布偏移: {shift.upper()} (0.65 <= AUC < 0.75)")
            self.logger.info("     高风险，分布差异显著")
        else:  # severe
            self.logger.info(f"  🚨 分布偏移: {shift.upper()} (AUC >= 0.75)")
            self.logger.info("     极高风险，训练集和测试集分布差异巨大")
        
        # 过拟合风险
        risk = r['overfitting_risk']
        if risk == 'low':
            self.logger.info(f"  ✅ 过拟合风险: {risk.upper()}")
        elif risk == 'medium':
            self.logger.info(f"  ⚠️  过拟合风险: {risk.upper()}")
        elif risk == 'high':
            self.logger.info(f"  🚨 过拟合风险: {risk.upper()}")
        else:  # very_high
            self.logger.info(f"  🚨 过拟合风险: {risk.upper()}")
            self.logger.info("     强烈建议重新划分训练/测试集或收集更多数据")
        
        self.logger.info("=" * 60)
    
    def get_feature_importance_report(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性报告
        
        Args:
            top_n: 返回前N个最重要的特征
            
        Returns:
            DataFrame包含特征名和重要性分数
        """
        if self.feature_importances_ is None:
            self.logger.warning("⚠️ 特征重要性未计算，请先运行validate()")
            return pd.DataFrame()
        
        report = pd.DataFrame({
            'feature': self.feature_importances_.head(top_n).index,
            'importance': self.feature_importances_.head(top_n).values
        })
        
        # 添加累积重要性
        report['cumulative_importance'] = report['importance'].cumsum() / report['importance'].sum()
        
        return report


def quick_adversarial_check(train_features: pd.DataFrame,
                            test_features: pd.DataFrame,
                            model_type: str = 'rf') -> Dict:
    """
    快速对抗性验证检查
    
    便捷函数，用于快速评估训练/测试分布差异。
    
    Args:
        train_features: 训练集特征
        test_features: 测试集特征
        model_type: 分类器类型
        
    Returns:
        验证结果字典
        
    Example:
        >>> result = quick_adversarial_check(X_train, X_test)
        >>> if result['cv_auc_mean'] > 0.65:
        >>>     print("⚠️ 警告: 存在显著的分布偏移!")
    """
    validator = AdversarialValidator(model_type=model_type)
    result = validator.validate(train_features, test_features)
    return result


def detect_covariate_shift(features_by_time: List[pd.DataFrame],
                           window_size: int = 5) -> List[Dict]:
    """
    检测时间序列中的协变量偏移（Covariate Shift）
    
    使用滑动窗口比较相邻时间段的特征分布。
    
    Args:
        features_by_time: 按时间排序的特征DataFrame列表
        window_size: 滑动窗口大小（时间段数）
        
    Returns:
        每个窗口对的验证结果列表
        
    Example:
        >>> # 假设有按月份分组的特征数据
        >>> monthly_features = [df_jan, df_feb, df_mar, ...]
        >>> shifts = detect_covariate_shift(monthly_features, window_size=3)
        >>> for i, shift in enumerate(shifts):
        >>>     if shift['cv_auc_mean'] > 0.6:
        >>>         print(f"时间段 {i} 到 {i+3} 存在分布偏移")
    """
    if len(features_by_time) < window_size + 1:
        raise ValueError(f"需要至少{window_size + 1}个时间段，但只提供了{len(features_by_time)}个")
    
    results = []
    validator = AdversarialValidator(model_type='rf', n_cv_splits=3)
    
    for i in range(len(features_by_time) - window_size):
        # 前窗口作为"训练集"
        train_window = pd.concat(
            features_by_time[i:i+window_size],
            axis=0,
            ignore_index=True
        )
        
        # 后窗口作为"测试集"
        test_window = features_by_time[i + window_size]
        
        logger.info(f"检测时间段 {i} 到 {i+window_size} 的分布偏移...")
        
        result = validator.validate(train_window, test_window)
        result['time_window_start'] = i
        result['time_window_end'] = i + window_size
        
        results.append(result)
    
    return results


if __name__ == '__main__':
    # 简单测试
    print("对抗性验证模块测试")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
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
    
    # 执行验证
    result = quick_adversarial_check(X_train, X_test)
    
    print(f"\n✅ 测试完成")
    print(f"   AUC: {result['cv_auc_mean']:.4f}")
    print(f"   分布偏移: {result['distribution_shift']}")

