# -*- coding: utf-8 -*-
"""
多目标特征选择器
使用NSGA-II算法进行Pareto最优特征选择
同时优化：预测能力、特征数量、特征多样性、分组约束
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import optuna
from optuna.samplers import NSGAIISampler

warnings.filterwarnings('ignore')


class MultiObjectiveFeatureSelector:
    """
    多目标特征选择器
    
    优化目标：
    1. 最大化预测能力（F1 Macro）
    2. 最小化特征数量
    3. 最大化特征多样性（跨类别）
    4. 满足特征分组约束
    """
    
    def __init__(self, 
                 objectives: List[str] = None,
                 constraints_path: str = "optuna_system/configs/feature_group_constraints.json",
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            objectives: 优化目标列表
            constraints_path: 特征分组约束配置文件路径
            logger: 日志记录器
        """
        self.objectives = objectives or [
            'maximize_f1_macro',
            'minimize_feature_count',
            'maximize_diversity',
            'satisfy_group_constraints'
        ]
        
        self.constraints_path = Path(constraints_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # 加载分组约束
        self.group_constraints = self._load_group_constraints()
        
        # 特征分组（根据特征名前缀识别）
        self.feature_groups = {
            'native_features': ['15m_native_', '1h_native_', '4h_native_', '1D_native_'],
            'td_sequential': ['td_setup_', 'td_countdown_', 'td_combo_', 'td_9_', 'td_13_', 'td_rei_', 'td_signal', 'td_bars'],
            'wyckoff': ['wyk_', 'wyckoff_'],
            'market_structure': ['pivot_', 'fractal_', 'swing_', 'trend_structure_', 'hh_', 'hl_', 'lh_', 'll_', 'msb_', 'choch_'],
            'candlestick': ['candle_', 'doji_', 'hammer_', 'star_', 'engulf_', 'harami_', 'pattern_'],
            'composite': ['smi_', 'elder_', 'cmo_', 'kst_', 'tsi_', 'ao_', 'zscore_', 'percentile_'],
            'trend': ['hull_', 'tema_', 'dema_', 'zlema_', 'kama_', 'alma_', 'ichimoku_', 'supertrend_', 'aroon_', 'vortex_'],
            'volume': ['obv_', 'ad_line_', 'cmf_', 'vwap_', 'volume_profile_', 'mfi_', 'eom_', 'pvi_', 'nvi_'],
            'harmonic': ['gartley_', 'butterfly_', 'bat_', 'crab_', 'shark_', 'cypher_', 'abcd_', 'three_drives_'],
            'elliott': ['elliott_wave_', 'wave_count_', 'wave_type_', 'wave_extension_'],
            'gann': ['gann_angle_', 'gann_fan_', 'gann_square_', 'gann_wheel_'],
            'volatility': ['garch_', 'realized_vol_', 'parkinson_', 'gk_vol_', 'rs_vol_', 'yz_vol_'],
            'time_series': ['seasonal_', 'trend_component_', 'residual_', 'fourier_'],
            'order_flow': ['spread_est_', 'large_order_', 'aggression_', 'depth_est_', 'liquidity_']
        }
    
    def _load_group_constraints(self) -> Dict:
        """加载特征分组约束"""
        if not self.constraints_path.exists():
            self.logger.warning(f"Constraints file not found: {self.constraints_path}, using defaults")
            return self._get_default_constraints()
        
        try:
            with open(self.constraints_path, 'r', encoding='utf-8') as f:
                constraints = json.load(f)
            self.logger.info(f"Loaded feature group constraints from {self.constraints_path}")
            return constraints
        except Exception as e:
            self.logger.error(f"Failed to load constraints: {e}, using defaults")
            return self._get_default_constraints()
    
    def _get_default_constraints(self) -> Dict:
        """获取默认约束"""
        return {
            "15m": {
                "native_features": {"min_ratio": 0.35, "max_ratio": 0.50},
                "td_sequential": {"min_count": 5, "max_count": 15},
                "wyckoff": {"min_count": 5, "max_count": 15},
                "market_structure": {"min_count": 3, "max_count": 10},
                "candlestick": {"min_count": 5, "max_count": 15},
                "composite": {"min_count": 3, "max_count": 10},
                "trend": {"min_count": 5, "max_count": 15},
                "volume": {"min_count": 5, "max_count": 15},
                "harmonic": {"min_count": 1, "max_count": 5},
                "elliott": {"min_count": 1, "max_count": 5},
                "gann": {"min_count": 0, "max_count": 3},
                "volatility": {"min_count": 2, "max_count": 8},
                "time_series": {"min_count": 1, "max_count": 5},
                "order_flow": {"min_count": 2, "max_count": 8},
                "total_target": {"min": 50, "max": 100}
            },
            "1h": {
                "native_features": {"min_ratio": 0.30, "max_ratio": 0.45},
                "td_sequential": {"min_count": 5, "max_count": 12},
                "wyckoff": {"min_count": 5, "max_count": 12},
                "market_structure": {"min_count": 3, "max_count": 10},
                "candlestick": {"min_count": 3, "max_count": 12},
                "composite": {"min_count": 3, "max_count": 10},
                "trend": {"min_count": 5, "max_count": 12},
                "volume": {"min_count": 5, "max_count": 12},
                "harmonic": {"min_count": 1, "max_count": 5},
                "elliott": {"min_count": 1, "max_count": 5},
                "gann": {"min_count": 0, "max_count": 3},
                "volatility": {"min_count": 2, "max_count": 8},
                "time_series": {"min_count": 1, "max_count": 5},
                "order_flow": {"min_count": 2, "max_count": 6},
                "total_target": {"min": 45, "max": 90}
            }
        }
    
    def _identify_feature_group(self, feature_name: str) -> str:
        """识别特征所属的分组"""
        for group_name, prefixes in self.feature_groups.items():
            for prefix in prefixes:
                if feature_name.startswith(prefix):
                    return group_name
        return 'other'
    
    def _calculate_group_counts(self, selected_features: List[str]) -> Dict[str, int]:
        """计算各分组的特征数量"""
        counts = {group: 0 for group in self.feature_groups.keys()}
        counts['other'] = 0
        
        for feature in selected_features:
            group = self._identify_feature_group(feature)
            counts[group] += 1
        
        return counts
    
    def _check_group_constraints(self, selected_features: List[str], 
                                timeframe: str, total_features: int) -> Tuple[bool, float]:
        """
        检查是否满足分组约束
        
        Returns:
            (is_valid, violation_score): 是否有效，违规程度（0=完全满足，越大越违规）
        """
        if timeframe not in self.group_constraints:
            timeframe = "15m"  # 默认
        
        constraints = self.group_constraints[timeframe]
        counts = self._calculate_group_counts(selected_features)
        
        violation_score = 0.0
        
        # 检查总数约束
        total_min = constraints.get('total_target', {}).get('min', 40)
        total_max = constraints.get('total_target', {}).get('max', 100)
        
        if len(selected_features) < total_min:
            violation_score += (total_min - len(selected_features)) * 2.0
        elif len(selected_features) > total_max:
            violation_score += (len(selected_features) - total_max) * 2.0
        
        # 检查各分组约束
        for group_name, count in counts.items():
            if group_name == 'other':
                continue
            
            group_constraint = constraints.get(group_name, {})
            
            # 比例约束（针对native features）
            if 'min_ratio' in group_constraint:
                min_ratio = group_constraint['min_ratio']
                max_ratio = group_constraint['max_ratio']
                actual_ratio = count / len(selected_features) if len(selected_features) > 0 else 0
                
                if actual_ratio < min_ratio:
                    violation_score += (min_ratio - actual_ratio) * 100
                elif actual_ratio > max_ratio:
                    violation_score += (actual_ratio - max_ratio) * 100
            
            # 数量约束
            if 'min_count' in group_constraint:
                min_count = group_constraint['min_count']
                max_count = group_constraint['max_count']
                
                if count < min_count:
                    violation_score += (min_count - count) * 1.0
                elif count > max_count:
                    violation_score += (count - max_count) * 1.0
        
        is_valid = violation_score == 0.0
        return is_valid, violation_score
    
    def _calculate_diversity(self, selected_features: List[str]) -> float:
        """
        计算特征多样性
        
        多样性定义为：选中特征覆盖的分组数量 / 总分组数量
        """
        if not selected_features:
            return 0.0
        
        selected_groups = set()
        for feature in selected_features:
            group = self._identify_feature_group(feature)
            selected_groups.add(group)
        
        # 归一化到[0, 1]
        diversity = len(selected_groups) / len(self.feature_groups)
        return diversity
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       timeframe: str = "15m",
                       n_trials: int = 100,
                       cv_folds: int = 3) -> Tuple[List[str], Dict]:
        """
        多目标特征选择
        
        Args:
            X: 特征矩阵
            y: 标签
            timeframe: 时框
            n_trials: 优化trial数
            cv_folds: 交叉验证折数
            
        Returns:
            (selected_features, metrics): 选中的特征列表和评估指标
        """
        self.logger.info(f"Starting multi-objective feature selection for {timeframe}")
        self.logger.info(f"Total features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        all_features = X.columns.tolist()
        
        # 定义优化目标函数
        def objective(trial: optuna.Trial) -> Tuple[float, float, float, float]:
            # 1. 选择特征子集（使用二进制编码）
            selected_mask = []
            for i, feature in enumerate(all_features):
                # 使用试探性选择，更智能地采样
                group = self._identify_feature_group(feature)
                
                # 不同分组使用不同的选择概率（使用先验概率权重）
                # Optuna不支持hint参数，我们使用suggest_categorical的随机性
                is_selected = trial.suggest_categorical(f'feature_{i}', [True, False])
                selected_mask.append(is_selected)
            
            selected_features = [f for f, s in zip(all_features, selected_mask) if s]
            
            if len(selected_features) == 0:
                return 0.0, 999.0, 0.0, 999.0
            
            # 2. 计算目标值
            
            # 目标1：预测能力（F1 Macro）- 最大化
            X_selected = X[selected_features]
            try:
                # 使用轻量级模型快速评估
                model = RandomForestClassifier(
                    n_estimators=50, 
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
                
                cv = TimeSeriesSplit(n_splits=cv_folds)
                scores = cross_val_score(model, X_selected, y, cv=cv, 
                                        scoring='f1_macro', n_jobs=-1)
                f1_macro = float(scores.mean())
            except Exception as e:
                self.logger.warning(f"Cross-validation failed: {e}")
                f1_macro = 0.0
            
            # 目标2：特征数量 - 最小化
            feature_count = len(selected_features)
            
            # 目标3：特征多样性 - 最大化
            diversity = self._calculate_diversity(selected_features)
            
            # 目标4：分组约束违规度 - 最小化
            is_valid, violation_score = self._check_group_constraints(
                selected_features, timeframe, len(all_features)
            )
            
            # 返回4个目标（NSGA-II会自动处理多目标）
            # 注意：optuna的NSGA-II是最小化所有目标，所以需要取负值来最大化
            return (
                -f1_macro,           # 最大化F1（取负变成最小化）
                feature_count,       # 最小化特征数
                -diversity,          # 最大化多样性（取负）
                violation_score      # 最小化违规度
            )
        
        # 使用NSGA-II采样器
        sampler = NSGAIISampler(population_size=50)
        
        study = optuna.create_study(
            directions=['minimize', 'minimize', 'minimize', 'minimize'],
            sampler=sampler
        )
        
        self.logger.info(f"Running {n_trials} trials with NSGA-II...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 获取Pareto前沿
        pareto_trials = [t for t in study.best_trials]
        
        self.logger.info(f"Found {len(pareto_trials)} Pareto optimal solutions")
        
        # 从Pareto前沿中选择最佳解
        # 策略：优先满足约束，然后平衡F1和特征数量
        best_trial = None
        best_score = -np.inf
        
        for trial in pareto_trials:
            f1_neg, feat_count, div_neg, violation = trial.values
            
            # 只考虑满足约束的解（violation接近0）
            if violation > 10:  # 允许少量违规
                continue
            
            # 综合评分：F1权重60%，特征数惩罚20%，多样性20%
            f1 = -f1_neg
            diversity = -div_neg
            
            # 特征数标准化（假设理想范围50-80）
            feat_penalty = abs(feat_count - 65) / 65.0
            
            score = f1 * 0.6 - feat_penalty * 0.2 + diversity * 0.2
            
            if score > best_score:
                best_score = score
                best_trial = trial
        
        # 如果没有找到满足约束的解，选择违规最小的
        if best_trial is None:
            best_trial = min(pareto_trials, key=lambda t: t.values[3])
            self.logger.warning("No solution fully satisfies constraints, selecting least-violating one")
        
        # 提取选中的特征
        selected_features = []
        for i, feature in enumerate(all_features):
            if best_trial.params[f'feature_{i}']:
                selected_features.append(feature)
        
        # 计算最终指标
        f1_neg, feat_count, div_neg, violation = best_trial.values
        
        metrics = {
            'f1_macro': -f1_neg,
            'feature_count': feat_count,
            'diversity': -div_neg,
            'constraint_violation': violation,
            'group_counts': self._calculate_group_counts(selected_features),
            'pareto_front_size': len(pareto_trials),
            'selected_solution_rank': pareto_trials.index(best_trial) + 1
        }
        
        self.logger.info(f"Selected {len(selected_features)} features with F1={metrics['f1_macro']:.4f}")
        self.logger.info(f"Feature group distribution: {metrics['group_counts']}")
        
        return selected_features, metrics
    
    def save_constraints_template(self, output_path: str):
        """保存约束模板配置文件"""
        template = self._get_default_constraints()
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        self.logger.info(f"Saved constraints template to {output_path}")


def main():
    """测试多目标特征选择器"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # 创建模拟特征名
    feature_names = []
    feature_names += [f'15m_native_feature_{i}' for i in range(30)]
    feature_names += [f'td_setup_{i}' for i in range(10)]
    feature_names += [f'wyk_spring_{i}' for i in range(10)]
    feature_names += [f'pivot_point_{i}' for i in range(10)]
    feature_names += [f'candle_pattern_{i}' for i in range(10)]
    feature_names += [f'smi_{i}' for i in range(10)]
    feature_names += [f'hull_ma_{i}' for i in range(10)]
    feature_names += [f'obv_{i}' for i in range(10)]
    
    X = pd.DataFrame(
        np.random.randn(n_samples, len(feature_names)),
        columns=feature_names
    )
    
    y = pd.Series(np.random.randint(0, 3, n_samples))
    
    # 测试选择器
    selector = MultiObjectiveFeatureSelector()
    
    print("Testing multi-objective feature selection...")
    selected_features, metrics = selector.select_features(X, y, timeframe='15m', n_trials=50)
    
    print(f"\nSelected {len(selected_features)} features:")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"Diversity: {metrics['diversity']:.4f}")
    print(f"Constraint violation: {metrics['constraint_violation']:.2f}")
    print(f"\nGroup distribution:")
    for group, count in metrics['group_counts'].items():
        if count > 0:
            print(f"  {group}: {count}")


if __name__ == '__main__':
    main()

