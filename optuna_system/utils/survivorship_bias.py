"""
生存者偏差校正模块

基于学术文献:
- Brown & Goetzmann (1995): Performance Persistence
- Elton et al. (1996): Survivorship Bias in Mutual Fund Performance  
- Efron & Tibshirani (1993): An Introduction to the Bootstrap

作者: Optuna System Team
日期: 2025-10-31
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class SurvivorshipBiasCorrector:
    """
    生存者偏差校正器
    
    使用Bootstrap方法校正仅包含幸存资产的回测结果
    """
    
    def __init__(self,
                 failure_db_path: str = "data/raw/failure_events.json"):
        """
        初始化校正器
        
        Args:
            failure_db_path: 失败事件数据库路径
        """
        self.logger = logger  # 先初始化logger
        self.failure_db_path = Path(failure_db_path)
        self.failure_events = self._load_failure_events()
        
    def _load_failure_events(self) -> List[Dict]:
        """加载失败事件数据库"""
        try:
            with open(self.failure_db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('events', [])
        except FileNotFoundError:
            self.logger.warning(f"失败事件数据库未找到: {self.failure_db_path}")
            return []
    
    def calculate_bias(self,
                      strategy_returns: pd.Series,
                      method: str = 'bootstrap',
                      n_bootstrap: int = 1000) -> Dict:
        """
        计算生存者偏差
        
        Args:
            strategy_returns: 策略收益率序列
            method: 校正方法 ('bootstrap', 'analytical')
            n_bootstrap: Bootstrap迭代次数
            
        Returns:
            Dict包含:
                - raw_sharpe: 原始夏普比率
                - corrected_sharpe: 校正后夏普比率
                - sharpe_bias: 夏普偏差
                - raw_return: 原始年化收益
                - corrected_return: 校正后年化收益
                - return_bias: 收益偏差
                - ci_lower: 95%置信区间下界
                - ci_upper: 95%置信区间上界
        """
        if strategy_returns.empty:
            return self._empty_result()
        
        # 计算原始指标
        raw_sharpe = self._calculate_sharpe(strategy_returns)
        raw_return = strategy_returns.mean() * 252  # 年化
        
        if method == 'bootstrap':
            result = self._bootstrap_correction(strategy_returns, n_bootstrap)
        elif method == 'analytical':
            result = self._analytical_correction(strategy_returns)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result.update({
            'raw_sharpe': raw_sharpe,
            'raw_return': raw_return,
        })
        
        self.logger.info(f"📊 生存者偏差校正:")
        self.logger.info(f"   原始Sharpe: {raw_sharpe:.3f}")
        self.logger.info(f"   校正Sharpe: {result['corrected_sharpe']:.3f}")
        self.logger.info(f"   偏差: {result['sharpe_bias']:.3f} ({result['sharpe_bias']/raw_sharpe*100:.1f}%)")
        
        return result
    
    def _bootstrap_correction(self,
                             returns: pd.Series,
                             n_iterations: int = 1000) -> Dict:
        """
        Bootstrap校正方法
        
        基于Efron & Tibshirani (1993)
        """
        self.logger.info(f"🔄 执行Bootstrap校正 (n={n_iterations})...")
        
        sharpes = []
        annual_returns = []
        
        for i in range(n_iterations):
            # 重采样，模拟包含失败案例
            sampled_returns = self._resample_with_failures(returns)
            
            # 计算指标
            sharpe = self._calculate_sharpe(sampled_returns)
            annual_ret = sampled_returns.mean() * 252
            
            sharpes.append(sharpe)
            annual_returns.append(annual_ret)
            
            if (i + 1) % 200 == 0:
                self.logger.debug(f"   完成 {i+1}/{n_iterations} 次迭代")
        
        # 计算统计量
        sharpes = np.array(sharpes)
        annual_returns = np.array(annual_returns)
        
        corrected_sharpe = np.mean(sharpes)
        corrected_return = np.mean(annual_returns)
        
        # 95%置信区间
        ci_lower_sharpe = np.percentile(sharpes, 2.5)
        ci_upper_sharpe = np.percentile(sharpes, 97.5)
        
        ci_lower_return = np.percentile(annual_returns, 2.5)
        ci_upper_return = np.percentile(annual_returns, 97.5)
        
        return {
            'corrected_sharpe': corrected_sharpe,
            'sharpe_bias': self._calculate_sharpe(returns) - corrected_sharpe,
            'corrected_return': corrected_return,
            'return_bias': (returns.mean() * 252) - corrected_return,
            'ci_lower_sharpe': ci_lower_sharpe,
            'ci_upper_sharpe': ci_upper_sharpe,
            'ci_lower_return': ci_lower_return,
            'ci_upper_return': ci_upper_return,
            'bootstrap_samples': n_iterations
        }
    
    def _resample_with_failures(self, returns: pd.Series) -> pd.Series:
        """
        重采样，注入失败案例
        
        模拟策略在包含失败资产的宇宙中的表现
        """
        n = len(returns)
        
        # 随机选择是否注入失败事件
        # 基于历史失败率（约10-15%的币种最终失败）
        if np.random.rand() < 0.12 and self.failure_events:
            # 选择一个失败事件
            event = np.random.choice(self.failure_events)
            
            # 在随机位置注入失败
            inject_idx = np.random.randint(n // 4, 3 * n // 4)  # 中间段
            crash_duration = event.get('time_to_crash_days', 5)
            crash_pct = event.get('drawdown_pct', -95) / 100
            
            # 创建副本
            modified_returns = returns.copy()
            
            # 注入崩盘收益
            for i in range(inject_idx, min(inject_idx + crash_duration, n)):
                # 分布式崩盘（不是单日）
                modified_returns.iloc[i] = crash_pct / crash_duration
            
            return modified_returns
        else:
            # 正常Bootstrap重采样
            return returns.sample(n=n, replace=True).reset_index(drop=True)
    
    def _analytical_correction(self, returns: pd.Series) -> Dict:
        """
        解析校正方法
        
        基于Elton et al. (1996)的研究
        假设偏差为年化收益的2-3%，Sharpe的15-20%
        """
        raw_sharpe = self._calculate_sharpe(returns)
        raw_return = returns.mean() * 252
        
        # 保守估计偏差
        sharpe_bias_pct = 0.18  # 18% Sharpe高估
        return_bias_annual = 0.025  # 2.5% 年化收益高估
        
        corrected_sharpe = raw_sharpe * (1 - sharpe_bias_pct)
        corrected_return = raw_return - return_bias_annual
        
        return {
            'corrected_sharpe': corrected_sharpe,
            'sharpe_bias': raw_sharpe - corrected_sharpe,
            'corrected_return': corrected_return,
            'return_bias': return_bias_annual,
            'ci_lower_sharpe': corrected_sharpe * 0.9,
            'ci_upper_sharpe': corrected_sharpe * 1.1,
            'ci_lower_return': corrected_return - 0.01,
            'ci_upper_return': corrected_return + 0.01
        }
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算年化夏普比率"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        std = excess_returns.std()
        
        # 零标准差或接近零
        if std == 0 or std < 1e-10:
            return 0.0
        
        sharpe = (excess_returns.mean() / std) * np.sqrt(252)
        return sharpe
    
    def _empty_result(self) -> Dict:
        """空结果"""
        return {
            'raw_sharpe': 0.0,
            'corrected_sharpe': 0.0,
            'sharpe_bias': 0.0,
            'raw_return': 0.0,
            'corrected_return': 0.0,
            'return_bias': 0.0,
            'ci_lower_sharpe': 0.0,
            'ci_upper_sharpe': 0.0
        }
    
    def get_failure_statistics(self) -> Dict:
        """获取失败事件统计"""
        if not self.failure_events:
            return {}
        
        drawdowns = [e.get('drawdown_pct', 0) for e in self.failure_events]
        crash_days = [e.get('time_to_crash_days', 0) for e in self.failure_events]
        
        return {
            'total_events': len(self.failure_events),
            'avg_drawdown': np.mean(drawdowns),
            'median_drawdown': np.median(drawdowns),
            'avg_crash_days': np.mean(crash_days),
            'recovery_rate': sum(1 for e in self.failure_events if e.get('recovery', False)) / len(self.failure_events)
        }


class FailureEventDatabase:
    """失败事件数据库"""
    
    def __init__(self, db_path: str = "data/raw/failure_events.json"):
        self.db_path = Path(db_path)
        self.events = self._load_events()
        self.logger = logger
    
    def _load_events(self) -> List[Dict]:
        """加载事件"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('events', [])
        except FileNotFoundError:
            self.logger.warning(f"数据库未找到: {self.db_path}")
            return []
    
    def get_event_by_symbol(self, symbol: str) -> Optional[Dict]:
        """根据交易对获取事件"""
        for event in self.events:
            if event.get('symbol') == symbol:
                return event
        return None
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """根据类型获取事件"""
        return [e for e in self.events if e.get('event_type') == event_type]
    
    def get_similar_events(self,
                          features: Dict,
                          k: int = 5) -> List[Dict]:
        """
        找到相似的历史事件
        
        使用简单的欧氏距离
        """
        if not self.events:
            return []
        
        similarities = []
        
        for event in self.events:
            event_features = event.get('pre_crash_features', {})
            
            # 计算特征相似度
            distance = self._calculate_feature_distance(features, event_features)
            similarities.append((event, distance))
        
        # 排序并返回最相似的k个
        similarities.sort(key=lambda x: x[1])
        return [e[0] for e in similarities[:k]]
    
    def _calculate_feature_distance(self,
                                    features1: Dict,
                                    features2: Dict) -> float:
        """计算特征欧氏距离"""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return float('inf')
        
        distance = 0.0
        for key in common_keys:
            v1 = features1[key]
            v2 = features2[key]
            distance += (v1 - v2) ** 2
        
        return np.sqrt(distance / len(common_keys))
    
    def calculate_failure_probability(self,
                                     current_features: Dict) -> float:
        """
        计算当前特征下的失败概率
        
        简单方法：基于相似历史事件的失败率
        """
        similar_events = self.get_similar_events(current_features, k=5)
        
        if not similar_events:
            return 0.05  # 默认5%失败率
        
        # 计算相似事件的失败率
        failed_count = sum(1 for e in similar_events if not e.get('recovery', False))
        
        return failed_count / len(similar_events)


def apply_survivorship_correction(backtest_results: Dict,
                                  method: str = 'bootstrap',
                                  n_bootstrap: int = 1000) -> Dict:
    """
    便捷函数：应用生存者偏差校正到回测结果
    
    Args:
        backtest_results: 包含returns_series的回测结果字典
        method: 校正方法
        n_bootstrap: Bootstrap次数
        
    Returns:
        增强的结果字典，包含校正后的指标
    """
    corrector = SurvivorshipBiasCorrector()
    
    returns = backtest_results.get('returns_series', pd.Series())
    
    if returns.empty:
        logger.warning("未找到收益率序列，跳过生存者偏差校正")
        return backtest_results
    
    correction = corrector.calculate_bias(returns, method=method, n_bootstrap=n_bootstrap)
    
    # 添加校正结果
    backtest_results['survivorship_correction'] = correction
    backtest_results['corrected_sharpe'] = correction['corrected_sharpe']
    backtest_results['corrected_annual_return'] = correction['corrected_return']
    
    return backtest_results

