# -*- coding: utf-8 -*-
"""
Financial-First Objective Functions

修复问题5：优化目标错配
- 当前问题：ML指标90%，金融指标10%（完全错误！）
- 修复方案：Financial-First，金融指标98%，ML指标2%

学术依据：
- Modern Portfolio Theory
- López de Prado "Advances in Financial ML"
- Quantified Strategies: Sharpe Ratio优先
- Performance Metrics & Risk Optimization (QuantInsti)

Reference:
[1] https://www.quantifiedstrategies.com/sharpe-ratio/
[2] https://blog.quantinsti.com/performance-metrics-risk-metrics-optimization/
[3] https://www.daytrading.com/objective-functions
[4] https://www.mezzi.com/blog/ai-portfolio-optimization-for-risk-management
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging


class FinancialMetricsCalculator:
    """
    计算标准金融指标（风险调整后收益）
    
    实现的指标：
    - Sharpe Ratio: 风险调整后收益（主要指标）
    - Sortino Ratio: 下行风险调整后收益
    - Calmar Ratio: 最大回撤调整后收益
    - Max Drawdown: 最大回撤
    - Win Rate: 胜率
    - Profit Factor: 盈亏比
    """
    
    def __init__(self, risk_free_rate: float = 0.02, logger=None):
        """
        Args:
            risk_free_rate: 无风险利率（年化），默认2%
            logger: 日志记录器
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        计算Sharpe Ratio（风险调整后收益）
        
        公式: Sharpe = (E[R] - Rf) / Std[R]
        
        Args:
            returns: 收益率序列（如daily returns）
            annualize: 是否年化（假设252个交易日）
            
        Returns:
            float: Sharpe Ratio
            
        学术依据：
        - William F. Sharpe (1966) "Mutual Fund Performance"
        - 15分钟频率，年化因子 = sqrt(252 * 24 * 4) ≈ 98
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        sharpe = (mean_return - self.risk_free_rate / 252) / std_return
        
        if annualize:
            # 15分钟数据：252天 * 24小时 * 4个15分钟 = 24,192个周期
            # 年化因子 = sqrt(24192) ≈ 155.6
            # 但更保守的做法是假设每天只交易8小时（32个15分钟周期）
            # 年化因子 = sqrt(252 * 32) ≈ 89.6
            annualization_factor = np.sqrt(252 * 32)  # 假设每天8小时交易
            sharpe *= annualization_factor
        
        return sharpe
    
    def calculate_sortino_ratio(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        计算Sortino Ratio（下行风险调整后收益）
        
        公式: Sortino = (E[R] - Rf) / DownsideStd[R]
        
        与Sharpe的区别：
        - Sharpe使用全部波动率
        - Sortino只使用下行波动率（更符合投资者心理）
        
        Args:
            returns: 收益率序列
            annualize: 是否年化
            
        Returns:
            float: Sortino Ratio
            
        学术依据：
        - Frank A. Sortino & Robert van der Meer (1991)
        - "Downside Risk" - Journal of Portfolio Management
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        
        # 下行收益（负收益）
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            downside_std = returns.std()  # 如果没有负收益，退化为Sharpe
        else:
            downside_std = downside_returns.std()
        
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        
        sortino = (mean_return - self.risk_free_rate / 252) / downside_std
        
        if annualize:
            annualization_factor = np.sqrt(252 * 32)
            sortino *= annualization_factor
        
        return sortino
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算最大回撤（Maximum Drawdown）
        
        定义：从峰值到谷底的最大跌幅
        公式: MaxDD = max(1 - current_value / peak_value)
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 最大回撤（0-1之间，如0.25表示25%回撤）
            
        学术依据：
        - Magdon-Ismail & Atiya (2004) "Maximum Drawdown"
        - Risk Management标准指标
        """
        if len(returns) < 2:
            return 0.0
        
        # 累积收益
        cumulative = (1 + returns).cumprod()
        
        # 滚动最大值
        running_max = cumulative.expanding().max()
        
        # 回撤序列
        drawdown = (cumulative - running_max) / running_max
        
        # 最大回撤（绝对值）
        max_dd = abs(drawdown.min())
        
        return max_dd
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """
        计算Calmar Ratio（最大回撤调整后收益）
        
        公式: Calmar = Annualized Return / Max Drawdown
        
        特点：
        - 衡量承担回撤风险所获得的收益
        - 比Sharpe更关注极端风险
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: Calmar Ratio
            
        学术依据：
        - Terry W. Young (1991)
        - "Calmar Ratio: A Smoother Tool"
        """
        if len(returns) < 2:
            return 0.0
        
        # 年化收益
        total_return = (1 + returns).prod() - 1
        periods = len(returns)
        # 假设15分钟数据，每年252*32个周期
        annualized_return = (1 + total_return) ** (252 * 32 / periods) - 1
        
        # 最大回撤
        max_dd = self.calculate_max_drawdown(returns)
        
        if max_dd == 0 or np.isnan(max_dd):
            return 0.0
        
        calmar = annualized_return / max_dd
        
        return calmar
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        计算胜率（Win Rate）
        
        定义：盈利交易占总交易的比例
        公式: Win Rate = 盈利次数 / 总交易次数
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 胜率（0-1之间）
        """
        if len(returns) == 0:
            return 0.0
        
        # 非零收益（实际交易）
        trades = returns[returns != 0]
        if len(trades) == 0:
            return 0.0
        
        # 盈利交易
        winning_trades = trades[trades > 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        return win_rate
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        计算盈亏比（Profit Factor）
        
        定义：总盈利 / 总亏损
        公式: Profit Factor = Sum(Positive Returns) / |Sum(Negative Returns)|
        
        特点：
        - >1表示盈利，<1表示亏损
        - >2表示优秀策略
        
        Args:
            returns: 收益率序列
            
        Returns:
            float: 盈亏比
        """
        if len(returns) == 0:
            return 1.0
        
        # 总盈利
        gross_profit = returns[returns > 0].sum()
        
        # 总亏损（绝对值）
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0
        
        profit_factor = gross_profit / gross_loss
        
        return profit_factor
    
    def calculate_all_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算所有金融指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            Dict: 包含所有指标的字典
        """
        metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns),
            'win_rate': self.calculate_win_rate(returns),
            'profit_factor': self.calculate_profit_factor(returns),
            'total_return': (1 + returns).prod() - 1,
            'volatility': returns.std() * np.sqrt(252 * 32),  # 年化波动率
        }
        
        return metrics


class FinancialFirstObjective:
    """
    Financial-First目标函数（修复问题5）
    
    核心设计原则：
    1. 金融指标主导（98%权重）
    2. ML指标辅助（2%权重）
    3. Sharpe Ratio为主要目标（40%权重）
    4. 风险控制为次要目标（20%权重）
    
    权重设计（基于学术研究）：
    - Sharpe Ratio: 40%  （从8%提升 +400%）
    - Max Drawdown: 20%  （新增）
    - Calmar Ratio: 15%  （新增）
    - Sortino Ratio: 10%  （新增）
    - Win Rate: 8%       （从2%提升）
    - Profit Factor: 5%  （新增）
    - F1 Macro: 2%       （从40%降低 -95%）
    
    学术依据：
    [1] López de Prado (2018) "Advances in Financial ML"
    [2] Sharpe, W.F. (1966) "Mutual Fund Performance"
    [3] QuantInsti: Performance & Risk Optimization
    [4] DayTrading.com: Objective Functions for Trading
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 weights: Optional[Dict[str, float]] = None,
                 logger=None):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
            weights: 自定义权重（如果为None，使用默认权重）
            logger: 日志记录器
        """
        self.risk_free_rate = risk_free_rate
        self.metrics_calculator = FinancialMetricsCalculator(risk_free_rate, logger)
        self.logger = logger or logging.getLogger(__name__)
        
        # 默认权重（Financial-First设计）
        if weights is None:
            self.weights = {
                'sharpe_ratio': 0.40,      # Sharpe为主要目标（从8%→40%）
                'max_drawdown': 0.20,      # 风险控制第二重要（新增）
                'calmar_ratio': 0.15,      # 回撤调整收益（新增）
                'sortino_ratio': 0.10,     # 下行风险调整（新增）
                'win_rate': 0.08,          # 胜率（从2%→8%）
                'profit_factor': 0.05,     # 盈亏比（新增）
                'f1_macro': 0.02           # 预测指标仅辅助（从40%→2%）
            }
        else:
            self.weights = weights
        
        # 验证权重和为1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"权重和不为1.0: {total_weight:.4f}，将进行归一化")
            # 归一化
            for key in self.weights:
                self.weights[key] /= total_weight
        
        self.logger.info(f"Financial-First目标函数初始化完成")
        self.logger.info(f"权重设计: {self.weights}")
    
    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        归一化指标到[0, 1]范围
        
        归一化方法：
        - Sharpe, Sortino, Calmar: sigmoid(x/2) - 因为好的值在0-3范围
        - Max Drawdown: 1 - min(x, 1) - 因为要最小化
        - Win Rate: 已经在[0, 1]
        - Profit Factor: sigmoid((x-1)/2) - 因为中心在1，好的值在1-5
        - F1: 已经在[0, 1]
        
        Args:
            metrics: 原始指标字典
            
        Returns:
            Dict: 归一化后的指标字典
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        normalized = {}
        
        # Sharpe Ratio (好的值在0-3，优秀>2)
        sharpe = metrics.get('sharpe_ratio', 0)
        normalized['sharpe'] = sigmoid(sharpe / 2)  # 归一化到[0, 1]
        
        # Sortino Ratio (类似Sharpe)
        sortino = metrics.get('sortino_ratio', 0)
        normalized['sortino'] = sigmoid(sortino / 2)
        
        # Calmar Ratio (好的值在0-5，优秀>2)
        calmar = metrics.get('calmar_ratio', 0)
        normalized['calmar'] = sigmoid(calmar / 2)
        
        # Max Drawdown (0-1，越小越好)
        max_dd = metrics.get('max_drawdown', 0)
        normalized['max_dd'] = min(max_dd, 1.0)  # 限制在[0, 1]，直接使用
        
        # Win Rate (0-1，已经归一化)
        win_rate = metrics.get('win_rate', 0)
        normalized['win_rate'] = min(max(win_rate, 0), 1)
        
        # Profit Factor (>1盈利，>2优秀)
        profit_factor = metrics.get('profit_factor', 1)
        normalized['profit_factor'] = sigmoid((profit_factor - 1) / 2)
        
        # F1 Macro (0-1，已经归一化)
        f1 = metrics.get('f1_macro', 0)
        normalized['f1_macro'] = min(max(f1, 0), 1)
        
        return normalized
    
    def calculate_score(self, 
                       returns: pd.Series, 
                       ml_metrics: Optional[Dict[str, float]] = None) -> float:
        """
        计算Financial-First综合得分
        
        公式：
        Score = Σ(weight_i * normalized_metric_i)
        
        其中：
        - 金融指标占98%
        - ML指标占2%
        
        Args:
            returns: 收益率序列
            ml_metrics: ML指标字典（如{'f1_macro': 0.75}）
            
        Returns:
            float: 综合得分（0-1之间）
        """
        # 计算所有金融指标
        financial_metrics = self.metrics_calculator.calculate_all_metrics(returns)
        
        # 合并ML指标
        if ml_metrics is None:
            ml_metrics = {}
        all_metrics = {**financial_metrics, **ml_metrics}
        
        # 归一化
        normalized = self._normalize_metrics(all_metrics)
        
        # 加权求和（Financial-First权重）
        score = (
            normalized['sharpe'] * self.weights['sharpe_ratio'] +
            (1 - normalized['max_dd']) * self.weights['max_drawdown'] +  # MaxDD要最小化，所以用1-x
            normalized['calmar'] * self.weights['calmar_ratio'] +
            normalized['sortino'] * self.weights['sortino_ratio'] +
            normalized['win_rate'] * self.weights['win_rate'] +
            normalized['profit_factor'] * self.weights['profit_factor'] +
            normalized['f1_macro'] * self.weights['f1_macro']
        )
        
        # 日志记录（调试用）
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Financial-First Score: {score:.4f}")
            self.logger.debug(f"  Sharpe: {financial_metrics['sharpe_ratio']:.4f} (norm: {normalized['sharpe']:.4f})")
            self.logger.debug(f"  MaxDD: {financial_metrics['max_drawdown']:.4f} (norm: {normalized['max_dd']:.4f})")
            self.logger.debug(f"  Calmar: {financial_metrics['calmar_ratio']:.4f} (norm: {normalized['calmar']:.4f})")
            self.logger.debug(f"  Sortino: {financial_metrics['sortino_ratio']:.4f} (norm: {normalized['sortino']:.4f})")
            self.logger.debug(f"  F1: {ml_metrics.get('f1_macro', 0):.4f} (norm: {normalized['f1_macro']:.4f})")
        
        return score
    
    def get_component_scores(self, 
                            returns: pd.Series, 
                            ml_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        获取各个组成部分的得分（用于调试和分析）
        
        Args:
            returns: 收益率序列
            ml_metrics: ML指标字典
            
        Returns:
            Dict: 包含各组成部分得分的字典
        """
        # 计算所有指标
        financial_metrics = self.metrics_calculator.calculate_all_metrics(returns)
        if ml_metrics is None:
            ml_metrics = {}
        all_metrics = {**financial_metrics, **ml_metrics}
        
        # 归一化
        normalized = self._normalize_metrics(all_metrics)
        
        # 计算各组成部分的贡献
        components = {
            'sharpe_component': normalized['sharpe'] * self.weights['sharpe_ratio'],
            'maxdd_component': (1 - normalized['max_dd']) * self.weights['max_drawdown'],
            'calmar_component': normalized['calmar'] * self.weights['calmar_ratio'],
            'sortino_component': normalized['sortino'] * self.weights['sortino_ratio'],
            'winrate_component': normalized['win_rate'] * self.weights['win_rate'],
            'profit_factor_component': normalized['profit_factor'] * self.weights['profit_factor'],
            'f1_component': normalized['f1_macro'] * self.weights['f1_macro'],
        }
        
        # 总分
        components['total_score'] = sum(components.values())
        
        # 金融vs ML贡献
        components['financial_contribution'] = sum([
            components['sharpe_component'],
            components['maxdd_component'],
            components['calmar_component'],
            components['sortino_component'],
            components['winrate_component'],
            components['profit_factor_component']
        ])
        components['ml_contribution'] = components['f1_component']
        
        return components


def create_default_financial_objective(risk_free_rate: float = 0.02) -> FinancialFirstObjective:
    """
    创建默认的Financial-First目标函数
    
    Args:
        risk_free_rate: 无风险利率
        
    Returns:
        FinancialFirstObjective: 目标函数实例
    """
    return FinancialFirstObjective(risk_free_rate=risk_free_rate)


def compare_old_vs_new_objectives(returns: pd.Series, 
                                  ml_metrics: Dict[str, float],
                                  logger=None) -> Dict[str, Any]:
    """
    对比旧目标函数vs新目标函数的得分差异
    
    Args:
        returns: 收益率序列
        ml_metrics: ML指标字典
        logger: 日志记录器
        
    Returns:
        Dict: 对比结果
    """
    logger = logger or logging.getLogger(__name__)
    
    # 旧目标函数（ML主导）
    metrics_calc = FinancialMetricsCalculator()
    financial_metrics = metrics_calc.calculate_all_metrics(returns)
    
    # 假设的旧得分计算方式（从optuna_model.py）
    # 需要归一化Sharpe到[0, 1]
    sharpe = financial_metrics['sharpe_ratio']
    normalized_sharpe = 1 / (1 + np.exp(-sharpe / 2))
    
    old_score = (
        ml_metrics.get('f1_weighted', ml_metrics.get('f1_macro', 0)) * 0.40 +
        ml_metrics.get('f1_macro', 0) * 0.25 +
        ml_metrics.get('accuracy', 0) * 0.10 +
        ml_metrics.get('auc_macro', 0) * 0.15 +
        normalized_sharpe * 0.08 +
        financial_metrics['win_rate'] * 0.02
    )
    
    # 新目标函数（Financial-First）
    new_objective = FinancialFirstObjective()
    new_score = new_objective.calculate_score(returns, ml_metrics)
    
    # 对比
    comparison = {
        'old_score': old_score,
        'new_score': new_score,
        'score_change': new_score - old_score,
        'score_change_pct': (new_score - old_score) / old_score * 100 if old_score > 0 else 0,
        'old_financial_weight': 0.10,  # Sharpe 8% + Win Rate 2%
        'new_financial_weight': 0.98,  # 除F1外都是金融指标
        'old_ml_weight': 0.90,
        'new_ml_weight': 0.02,
        'component_scores': new_objective.get_component_scores(returns, ml_metrics)
    }
    
    logger.info("=" * 60)
    logger.info("目标函数对比（旧 vs 新）")
    logger.info("=" * 60)
    logger.info(f"旧得分（ML主导90%）: {old_score:.4f}")
    logger.info(f"新得分（Financial-First 98%）: {new_score:.4f}")
    logger.info(f"得分变化: {comparison['score_change']:+.4f} ({comparison['score_change_pct']:+.2f}%)")
    logger.info(f"金融权重: {comparison['old_financial_weight']:.0%} → {comparison['new_financial_weight']:.0%}")
    logger.info(f"ML权重: {comparison['old_ml_weight']:.0%} → {comparison['new_ml_weight']:.0%}")
    logger.info("=" * 60)
    
    return comparison

