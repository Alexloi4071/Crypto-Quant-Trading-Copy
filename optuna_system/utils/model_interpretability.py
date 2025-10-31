"""
模型可解释性框架

集成SHAP、LIME和排列重要性等多种可解释性方法，
用于理解模型决策过程和检测潜在偏差。

基于学术文献:
- Lundberg, S. M., & Lee, S. I. (2017): "A Unified Approach to Interpreting Model Predictions" (SHAP)
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016): "Why Should I Trust You?" (LIME)
- Breiman, L. (2001): "Random Forests" (Permutation Importance)
- Molnar, C. (2020): "Interpretable Machine Learning"

作者: Optuna System Team
日期: 2025-10-31
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("⚠️ SHAP未安装，部分功能将不可用。安装: pip install shap")

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    logger.warning("⚠️ LIME未安装，部分功能将不可用。安装: pip install lime")


class ModelInterpreter:
    """
    模型可解释性解释器
    
    支持多种解释方法:
    1. SHAP (SHapley Additive exPlanations)
    2. LIME (Local Interpretable Model-agnostic Explanations)
    3. Permutation Importance
    4. Feature Interaction Detection
    """
    
    def __init__(self,
                 model: BaseEstimator,
                 X_train: pd.DataFrame,
                 feature_names: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        初始化解释器
        
        Args:
            model: 已训练的模型
            X_train: 训练数据（用于建立基线）
            feature_names: 特征名称列表
            random_state: 随机种子
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names is not None else X_train.columns.tolist()
        self.random_state = random_state
        self.logger = logger
        
        # 存储解释结果
        self.shap_values_ = None
        self.shap_explainer_ = None
        self.lime_explainer_ = None
        self.perm_importance_ = None
    
    def explain_with_shap(self,
                         X_test: pd.DataFrame,
                         method: str = 'auto',
                         max_samples: int = 100) -> Dict:
        """
        使用SHAP解释模型预测
        
        Args:
            X_test: 测试数据
            method: SHAP方法 ('auto', 'tree', 'kernel', 'deep', 'linear')
            max_samples: 最大样本数（避免计算太慢）
            
        Returns:
            Dict包含SHAP值和总结
        """
        if not HAS_SHAP:
            self.logger.error("❌ SHAP未安装，无法使用此功能")
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info(f"🔍 使用SHAP解释模型 (method={method})...")
        
        # 限制样本数
        if len(X_test) > max_samples:
            self.logger.info(f"  📊 样本数过多，随机采样{max_samples}个")
            X_test_sample = X_test.sample(n=max_samples, random_state=self.random_state)
        else:
            X_test_sample = X_test
        
        # 选择SHAP explainer
        if method == 'auto':
            # 自动选择（基于模型类型）
            if hasattr(self.model, 'tree_'):
                method = 'tree'
            elif hasattr(self.model, 'coef_'):
                method = 'linear'
            else:
                method = 'kernel'
            self.logger.info(f"  🤖 自动选择SHAP方法: {method}")
        
        try:
            if method == 'tree':
                # TreeExplainer（适用于树模型）
                self.shap_explainer_ = shap.TreeExplainer(self.model)
                self.shap_values_ = self.shap_explainer_.shap_values(X_test_sample)
            
            elif method == 'kernel':
                # KernelExplainer（模型无关）
                self.shap_explainer_ = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(self.X_train, min(100, len(self.X_train)))
                )
                self.shap_values_ = self.shap_explainer_.shap_values(X_test_sample)
            
            elif method == 'linear':
                # LinearExplainer（线性模型）
                self.shap_explainer_ = shap.LinearExplainer(
                    self.model,
                    self.X_train
                )
                self.shap_values_ = self.shap_explainer_.shap_values(X_test_sample)
            
            else:
                raise ValueError(f"Unsupported SHAP method: {method}")
            
            # 处理多类别情况
            if isinstance(self.shap_values_, list):
                # 多类别：取第一个类别或平均
                shap_vals = np.abs(self.shap_values_[1])  # 通常关注正类
            else:
                shap_vals = np.abs(self.shap_values_)
            
            # 计算全局特征重要性
            global_importance = np.mean(shap_vals, axis=0)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': global_importance
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"  🔝 Top 10 SHAP重要特征:")
            for i, row in importance_df.head(10).iterrows():
                self.logger.info(f"     {row['feature']}: {row['importance']:.4f}")
            
            result = {
                'shap_values': self.shap_values_,
                'global_importance': importance_df,
                'method': method,
                'n_samples': len(X_test_sample)
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ SHAP解释失败: {e}")
            return {}
    
    def explain_with_lime(self,
                         X_test: pd.DataFrame,
                         instance_idx: int = 0,
                         n_features: int = 10) -> Dict:
        """
        使用LIME解释单个预测
        
        Args:
            X_test: 测试数据
            instance_idx: 要解释的实例索引
            n_features: 显示的特征数量
            
        Returns:
            LIME解释结果
        """
        if not HAS_LIME:
            self.logger.error("❌ LIME未安装，无法使用此功能")
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info(f"🔍 使用LIME解释单个预测 (instance={instance_idx})...")
        
        try:
            # 创建LIME explainer
            if self.lime_explainer_ is None:
                self.lime_explainer_ = lime_tabular.LimeTabularExplainer(
                    self.X_train.values,
                    feature_names=self.feature_names,
                    class_names=['class_0', 'class_1'],
                    mode='classification',
                    random_state=self.random_state
                )
            
            # 解释单个实例
            instance = X_test.iloc[instance_idx].values
            explanation = self.lime_explainer_.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=n_features
            )
            
            # 提取特征权重
            feature_weights = explanation.as_list()
            
            self.logger.info(f"  📊 LIME特征贡献 (Top {n_features}):")
            for feat, weight in feature_weights:
                self.logger.info(f"     {feat}: {weight:+.4f}")
            
            result = {
                'explanation': explanation,
                'feature_weights': feature_weights,
                'instance_idx': instance_idx,
                'prediction': self.model.predict_proba([instance])[0]
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ LIME解释失败: {e}")
            return {}
    
    def explain_with_permutation(self,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                n_repeats: int = 10,
                                scoring: str = 'f1_macro') -> Dict:
        """
        使用排列重要性解释模型
        
        Args:
            X_test: 测试数据
            y_test: 测试标签
            n_repeats: 排列重复次数
            scoring: 评分指标
            
        Returns:
            排列重要性结果
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🔍 使用排列重要性解释模型 (n_repeats={n_repeats})...")
        
        try:
            # 计算排列重要性
            perm_result = permutation_importance(
                self.model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=self.random_state,
                n_jobs=-1,
                scoring=scoring
            )
            
            # 整理结果
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance_mean': perm_result.importances_mean,
                'importance_std': perm_result.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            self.perm_importance_ = importance_df
            
            self.logger.info(f"  🔝 Top 10排列重要特征:")
            for i, row in importance_df.head(10).iterrows():
                self.logger.info(f"     {row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
            
            result = {
                'importance_df': importance_df,
                'raw_importances': perm_result.importances
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ 排列重要性计算失败: {e}")
            return {}
    
    def detect_feature_interactions(self,
                                    X_test: pd.DataFrame,
                                    top_n: int = 5) -> Dict:
        """
        检测特征交互（基于SHAP交互值）
        
        Args:
            X_test: 测试数据
            top_n: 返回前N对交互特征
            
        Returns:
            特征交互结果
        """
        if not HAS_SHAP:
            self.logger.error("❌ SHAP未安装，无法检测特征交互")
            return {}
        
        self.logger.info("=" * 60)
        self.logger.info("🔗 检测特征交互...")
        
        try:
            # 限制样本数（交互计算很慢）
            max_samples = min(50, len(X_test))
            X_test_sample = X_test.sample(n=max_samples, random_state=self.random_state)
            
            self.logger.info(f"  📊 使用{max_samples}个样本计算交互...")
            
            # 计算SHAP交互值
            if self.shap_explainer_ is None:
                if hasattr(self.model, 'tree_'):
                    self.shap_explainer_ = shap.TreeExplainer(self.model)
                else:
                    self.logger.warning("⚠️ 仅树模型支持快速交互计算")
                    return {}
            
            shap_interaction_values = self.shap_explainer_.shap_interaction_values(X_test_sample)
            
            # 处理多类别
            if isinstance(shap_interaction_values, list):
                shap_interaction_values = shap_interaction_values[1]
            
            # 计算平均交互强度
            n_features = len(self.feature_names)
            interaction_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        interaction_matrix[i, j] = np.abs(shap_interaction_values[:, i, j]).mean()
            
            # 找到最强的交互
            interactions = []
            for i in range(n_features):
                for j in range(i+1, n_features):
                    interactions.append({
                        'feature_1': self.feature_names[i],
                        'feature_2': self.feature_names[j],
                        'interaction_strength': interaction_matrix[i, j]
                    })
            
            interactions_df = pd.DataFrame(interactions).sort_values(
                'interaction_strength',
                ascending=False
            )
            
            self.logger.info(f"  🔝 Top {top_n}特征交互:")
            for i, row in interactions_df.head(top_n).iterrows():
                self.logger.info(f"     {row['feature_1']} × {row['feature_2']}: {row['interaction_strength']:.4f}")
            
            result = {
                'interactions_df': interactions_df,
                'interaction_matrix': interaction_matrix,
                'feature_names': self.feature_names
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"❌ 特征交互检测失败: {e}")
            return {}
    
    def comprehensive_interpretation(self,
                                    X_test: pd.DataFrame,
                                    y_test: Optional[pd.Series] = None) -> Dict:
        """
        执行全面的模型解释分析
        
        包括:
        1. SHAP全局解释
        2. 排列重要性
        3. 特征交互（可选）
        
        Args:
            X_test: 测试数据
            y_test: 测试标签（排列重要性需要）
            
        Returns:
            完整解释结果
        """
        self.logger.info("🚀 开始全面模型解释分析...")
        
        results = {}
        
        # 1. SHAP解释
        if HAS_SHAP:
            results['shap'] = self.explain_with_shap(X_test)
        
        # 2. 排列重要性
        if y_test is not None:
            results['permutation'] = self.explain_with_permutation(X_test, y_test)
        
        # 3. 特征交互（可选，计算慢）
        # results['interactions'] = self.detect_feature_interactions(X_test)
        
        # 4. 特征重要性一致性检查
        if 'shap' in results and 'permutation' in results:
            results['consistency'] = self._check_importance_consistency(
                results['shap']['global_importance'],
                results['permutation']['importance_df']
            )
        
        self._print_comprehensive_summary(results)
        
        return results
    
    def _check_importance_consistency(self,
                                     shap_importance: pd.DataFrame,
                                     perm_importance: pd.DataFrame) -> Dict:
        """检查不同方法的特征重要性一致性"""
        self.logger.info("=" * 60)
        self.logger.info("🔍 检查特征重要性一致性...")
        
        # 获取Top 10特征
        shap_top10 = set(shap_importance.head(10)['feature'])
        perm_top10 = set(perm_importance.head(10)['feature'])
        
        # 计算重叠
        overlap = shap_top10 & perm_top10
        overlap_pct = len(overlap) / 10 * 100
        
        self.logger.info(f"  📊 Top 10特征重叠: {len(overlap)}/10 ({overlap_pct:.0f}%)")
        
        if overlap_pct >= 70:
            self.logger.info("  ✅ 一致性高 - 特征重要性可靠")
            consistency_level = 'high'
        elif overlap_pct >= 50:
            self.logger.info("  ⚠️  一致性中等 - 需进一步验证")
            consistency_level = 'medium'
        else:
            self.logger.info("  🚨 一致性低 - 特征重要性不稳定")
            consistency_level = 'low'
        
        return {
            'overlap_features': list(overlap),
            'overlap_pct': overlap_pct,
            'consistency_level': consistency_level
        }
    
    def _print_comprehensive_summary(self, results: Dict):
        """打印全面解释总结"""
        self.logger.info("=" * 60)
        self.logger.info("🏆 全面模型解释总结:")
        
        if 'shap' in results and results['shap']:
            self.logger.info("  ✅ SHAP解释完成")
        
        if 'permutation' in results and results['permutation']:
            self.logger.info("  ✅ 排列重要性完成")
        
        if 'consistency' in results:
            level = results['consistency']['consistency_level']
            overlap_pct = results['consistency']['overlap_pct']
            self.logger.info(f"  📊 特征重要性一致性: {level.upper()} ({overlap_pct:.0f}%重叠)")
        
        self.logger.info("=" * 60)
    
    def plot_shap_summary(self, save_path: Optional[str] = None):
        """绘制SHAP总结图"""
        if not HAS_SHAP or self.shap_values_ is None:
            self.logger.warning("⚠️ 请先运行explain_with_shap()")
            return
        
        try:
            plt.figure(figsize=(10, 6))
            
            if isinstance(self.shap_values_, list):
                shap.summary_plot(self.shap_values_[1], self.X_train, show=False)
            else:
                shap.summary_plot(self.shap_values_, self.X_train, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"  💾 SHAP总结图已保存: {save_path}")
            
            plt.close()
        
        except Exception as e:
            self.logger.error(f"❌ 绘图失败: {e}")


if __name__ == '__main__':
    # 简单测试
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("模型可解释性模块测试")
    
    # 生成数据
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 创建解释器
    interpreter = ModelInterpreter(model, X_train)
    
    # 执行解释
    results = interpreter.comprehensive_interpretation(X_test, y_test)
    
    print(f"\n✅ 测试完成")
    if 'consistency' in results:
        print(f"   一致性: {results['consistency']['consistency_level']}")

