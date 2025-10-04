"""
Risk Attribution
风险归因分析系统，基于因子模型和组合分解方法
分析投资组合风险的来源和贡献度，支持多种风险归因模型
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path
from scipy import stats, linalg
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

class RiskFactorType(Enum):
    """风险因子类型"""
    MARKET = "market"                    # 市场因子
    SIZE = "size"                       # 市值因子
    VALUE = "value"                     # 价值因子
    MOMENTUM = "momentum"               # 动量因子
    QUALITY = "quality"                 # 质量因子
    VOLATILITY = "volatility"           # 波动率因子
    LIQUIDITY = "liquidity"             # 流动性因子
    SECTOR = "sector"                   # 行业因子
    COUNTRY = "country"                 # 国家因子
    CURRENCY = "currency"               # 货币因子
    INTEREST_RATE = "interest_rate"     # 利率因子
    CREDIT = "credit"                   # 信用因子
    COMMODITY = "commodity"             # 商品因子
    CUSTOM = "custom"                   # 自定义因子

class RiskMeasure(Enum):
    """风险度量类型"""
    VARIANCE = "variance"               # 方差
    VOLATILITY = "volatility"           # 波动率
    VAR = "var"                        # 风险价值
    CVAR = "cvar"                      # 条件风险价值
    TRACKING_ERROR = "tracking_error"   # 跟踪误差
    BETA = "beta"                      # 贝塔系数
    CORRELATION = "correlation"         # 相关性

@dataclass

class RiskFactor:
    """风险因子"""
    factor_id: str
    factor_name: str
    factor_type: RiskFactorType

    # 因子数据
    factor_returns: pd.Series
    factor_loadings: pd.Series = field(default_factory=pd.Series)

    # 因子特征
    factor_volatility: float = 0.0
    factor_sharpe: float = 0.0
    factor_skewness: float = 0.0
    factor_kurtosis: float = 0.0

    # 因子描述
    description: str = ""

    def calculate_statistics(self):
        """计算因子统计特征"""
        if len(self.factor_returns) > 0:
            self.factor_volatility = self.factor_returns.std() * np.sqrt(252)
            self.factor_sharpe = self.factor_returns.mean() / self.factor_returns.std() * np.sqrt(252)
            self.factor_skewness = self.factor_returns.skew()
            self.factor_kurtosis = self.factor_returns.kurtosis()

    def to_dict(self) -> dict:
        return {
            'factor_id': self.factor_id,
            'factor_name': self.factor_name,
            'factor_type': self.factor_type.value,
            'factor_volatility': self.factor_volatility,
            'factor_sharpe': self.factor_sharpe,
            'factor_skewness': self.factor_skewness,
            'factor_kurtosis': self.factor_kurtosis,
            'description': self.description,
            'data_points': len(self.factor_returns)
        }

@dataclass

class RiskContribution:
    """风险贡献"""
    asset_id: str
    factor_id: str

    # 风险贡献度
    absolute_contribution: float  # 绝对贡献
    relative_contribution: float  # 相对贡献（百分比）
    marginal_contribution: float  # 边际贡献

    # 因子暴露
    factor_exposure: float
    factor_return: float

    # 统计信息
    t_statistic: float = 0.0
    p_value: float = 1.0
    r_squared: float = 0.0

    def to_dict(self) -> dict:
        return {
            'asset_id': self.asset_id,
            'factor_id': self.factor_id,
            'absolute_contribution': self.absolute_contribution,
            'relative_contribution': self.relative_contribution,
            'marginal_contribution': self.marginal_contribution,
            'factor_exposure': self.factor_exposure,
            'factor_return': self.factor_return,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'r_squared': self.r_squared
        }

@dataclass

class PortfolioRiskAttribution:
    """投资组合风险归因结果"""
    portfolio_id: str
    attribution_date: datetime

    # 组合总风险
    total_risk: float
    active_risk: float = 0.0  # 相对基准的主动风险

    # 因子风险贡献
    factor_contributions: List[RiskContribution] = field(default_factory=list)

    # 特异风险
    specific_risk: float = 0.0
    specific_risk_contribution: float = 0.0

    # 风险分解
    systematic_risk: float = 0.0  # 系统性风险
    idiosyncratic_risk: float = 0.0  # 特异风险

    # 各因子类型的风险贡献
    risk_by_factor_type: Dict[str, float] = field(default_factory=dict)

    # 资产层面的风险贡献
    risk_by_asset: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'portfolio_id': self.portfolio_id,
            'attribution_date': self.attribution_date.isoformat(),
            'total_risk': self.total_risk,
            'active_risk': self.active_risk,
            'systematic_risk': self.systematic_risk,
            'specific_risk': self.specific_risk,
            'specific_risk_contribution': self.specific_risk_contribution,
            'risk_by_factor_type': self.risk_by_factor_type,
            'risk_by_asset': self.risk_by_asset,
            'factor_contributions': [fc.to_dict() for fc in self.factor_contributions]
        }

class FactorModel:
    """因子模型基类"""

    def __init__(self, name: str):
        self.name = name
        self.factors = {}
        self.factor_loadings = pd.DataFrame()
        self.factor_returns = pd.DataFrame()
        self.residual_returns = pd.DataFrame()

    def fit(self, asset_returns: pd.DataFrame, factor_data: pd.DataFrame = None):
        """拟合因子模型"""
        raise NotImplementedError

    def calculate_risk_attribution(self, weights: pd.Series) -> PortfolioRiskAttribution:
        """计算风险归因"""
        raise NotImplementedError

class FamaFrenchModel(FactorModel):
    """Fama-French三因子模型"""

    def __init__(self):
        super().__init__("Fama-French 3-Factor")

    def fit(self, asset_returns: pd.DataFrame, factor_data: pd.DataFrame = None):
        """拟合FF三因子模型"""
        # 如果没有提供因子数据，构建简单的代理因子
        if factor_data is None:
            factor_data = self._construct_proxy_factors(asset_returns)

        # 确保有必要的因子
        required_factors = ['MKT', 'SMB', 'HML']
        for factor in required_factors:
            if factor not in factor_data.columns:
                logger.warning(f"缺少因子 {factor}，使用默认值")
                factor_data[factor] = np.random.normal(0, 0.1, len(factor_data))

        self.factor_returns = factor_data[required_factors]

        # 对每个资产进行回归
        loadings = []
        residuals = []

        common_index = asset_returns.index.intersection(factor_data.index)
        asset_returns_aligned = asset_returns.loc[common_index]
        factor_returns_aligned = self.factor_returns.loc[common_index]

        for asset in asset_returns.columns:
            # 多元线性回归
            y = asset_returns_aligned[asset].dropna()
            X = factor_returns_aligned.loc[y.index]

            if len(y) > 10 and len(X) > 10:  # 确保有足够的数据
                reg = LinearRegression().fit(X, y)

                # 保存因子载荷
                asset_loadings = pd.Series(reg.coef_, index=required_factors, name=asset)
                loadings.append(asset_loadings)

                # 计算残差
                predictions = reg.predict(X)
                residual = y - predictions
                residuals.append(pd.Series(residual, name=asset))

        if loadings:
            self.factor_loadings = pd.DataFrame(loadings)
            self.factor_loadings = self.factor_loadings.T

        if residuals:
            # 对齐残差数据
            max_length = max(len(r) for r in residuals)
            aligned_residuals = []

            for residual in residuals:
                if len(residual) < max_length:
                    # 填充缺失值
                    residual = residual.reindex(residuals[0].index).fillna(0)
                aligned_residuals.append(residual)

            self.residual_returns = pd.DataFrame(aligned_residuals).T

        # 创建因子对象
        for factor_name in required_factors:
            factor_returns_series = self.factor_returns[factor_name]

            factor = RiskFactor(
                factor_id=factor_name,
                factor_name=self._get_factor_description(factor_name),
                factor_type=self._get_factor_type(factor_name),
                factor_returns=factor_returns_series,
                description=self._get_factor_description(factor_name)
            )
            factor.calculate_statistics()
            self.factors[factor_name] = factor

        logger.info(f"FF三因子模型拟合完成，{len(self.factor_loadings.columns)} 个资产")

    def _construct_proxy_factors(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """构建代理因子"""
        # 市场因子：所有资产的等权重平均收益
        mkt_factor = asset_returns.mean(axis=1)

        # SMB因子：简单的规模代理（使用收益率波动性）
        volatilities = asset_returns.rolling(window=60).std()
        small_assets = volatilities.gt(volatilities.median(axis=1), axis=0)
        big_assets = ~small_assets

        smb_factor = (asset_returns.where(small_assets).mean(axis=1) -
                     asset_returns.where(big_assets).mean(axis=1))

        # HML因子：简单的价值代理（使用动量反转）
        returns_momentum = asset_returns.rolling(window=252).mean()
        value_assets = returns_momentum.lt(returns_momentum.median(axis=1), axis=0)
        growth_assets = ~value_assets

        hml_factor = (asset_returns.where(value_assets).mean(axis=1) -
                     asset_returns.where(growth_assets).mean(axis=1))

        factor_data = pd.DataFrame({
            'MKT': mkt_factor,
            'SMB': smb_factor.fillna(0),
            'HML': hml_factor.fillna(0)
        })

        return factor_data

    def _get_factor_type(self, factor_name: str) -> RiskFactorType:
        """获取因子类型"""
        factor_type_map = {
            'MKT': RiskFactorType.MARKET,
            'SMB': RiskFactorType.SIZE,
            'HML': RiskFactorType.VALUE
        }
        return factor_type_map.get(factor_name, RiskFactorType.CUSTOM)

    def _get_factor_description(self, factor_name: str) -> str:
        """获取因子描述"""
        descriptions = {
            'MKT': '市场因子 - 整体市场风险',
            'SMB': '市值因子 - 小市值相对大市值的超额收益',
            'HML': '价值因子 - 高账面市值比相对低账面市值比的超额收益'
        }
        return descriptions.get(factor_name, f'自定义因子: {factor_name}')

    def calculate_risk_attribution(self, weights: pd.Series,
                                 benchmark_weights: pd.Series = None) -> PortfolioRiskAttribution:
        """计算FF三因子风险归因"""

        if self.factor_loadings.empty:
            raise ValueError("模型未拟合，请先调用fit方法")

        # 对齐权重和因子载荷
        common_assets = weights.index.intersection(self.factor_loadings.columns)
        if len(common_assets) == 0:
            raise ValueError("权重和因子载荷没有共同资产")

        weights_aligned = weights.loc[common_assets]
        loadings_aligned = self.factor_loadings[common_assets]

        # 计算组合的因子暴露
        portfolio_exposures = loadings_aligned.dot(weights_aligned)

        # 计算因子协方差矩阵
        factor_cov = self.factor_returns.cov() * 252  # 年化

        # 计算特异风险协方差矩阵
        if not self.residual_returns.empty:
            specific_cov = self.residual_returns.cov() * 252
            # 只保留对角线元素（假设特异风险之间不相关）
            specific_cov = pd.DataFrame(
                np.diag(np.diag(specific_cov.values)),
                index=specific_cov.index,
                columns=specific_cov.columns
            )
        else:
            # 如果没有残差数据，使用简单估计
            specific_cov = pd.DataFrame(
                np.eye(len(common_assets)) * 0.05,  # 假设5%的特异波动率
                index=common_assets,
                columns=common_assets
            )

        # 确保特异风险矩阵与权重对齐
        specific_cov_aligned = specific_cov.reindex(
            index=common_assets, columns=common_assets
        ).fillna(0)

        # 计算总组合风险
        # 因子风险贡献
        factor_risk_contrib = portfolio_exposures.T @ factor_cov @ portfolio_exposures

        # 特异风险贡献
        specific_risk_contrib = weights_aligned.T @ specific_cov_aligned @ weights_aligned

        # 总风险
        total_variance = factor_risk_contrib + specific_risk_contrib
        total_risk = np.sqrt(total_variance)

        # 计算各因子的风险贡献
        factor_contributions = []

        for factor_name in self.factors.keys():
            if factor_name in portfolio_exposures.index:
                factor_exposure = portfolio_exposures[factor_name]
                factor_variance = factor_cov.loc[factor_name, factor_name]

                # 绝对风险贡献
                abs_contrib = (factor_exposure ** 2) * factor_variance

                # 相对风险贡献
                rel_contrib = abs_contrib / total_variance * 100

                # 边际风险贡献
                marginal_contrib = 2 * factor_exposure * factor_variance / (2 * total_risk)

                contribution = RiskContribution(
                    asset_id="Portfolio",
                    factor_id=factor_name,
                    absolute_contribution=abs_contrib,
                    relative_contribution=rel_contrib,
                    marginal_contribution=marginal_contrib,
                    factor_exposure=factor_exposure,
                    factor_return=self.factors[factor_name].factor_returns.mean() * 252
                )

                factor_contributions.append(contribution)

        # 计算各因子类型的风险贡献
        risk_by_factor_type = {}
        for contrib in factor_contributions:
            factor_type = self.factors[contrib.factor_id].factor_type.value
            risk_by_factor_type[factor_type] = risk_by_factor_type.get(factor_type, 0) + contrib.relative_contribution

        # 计算各资产的风险贡献
        risk_by_asset = {}
        for asset in common_assets:
            asset_weight = weights_aligned[asset]

            # 因子风险贡献
            asset_loadings = loadings_aligned[asset]
            asset_factor_risk = asset_loadings.T @ factor_cov @ portfolio_exposures * asset_weight

            # 特异风险贡献
            asset_specific_risk = specific_cov_aligned.loc[asset, asset] * (asset_weight ** 2)

            # 总贡献
            asset_total_contrib = (asset_factor_risk + asset_specific_risk) / total_variance * 100
            risk_by_asset[asset] = asset_total_contrib

        # 创建风险归因结果
        attribution = PortfolioRiskAttribution(
            portfolio_id="portfolio",
            attribution_date=datetime.now(),
            total_risk=total_risk,
            systematic_risk=np.sqrt(factor_risk_contrib),
            specific_risk=np.sqrt(specific_risk_contrib),
            specific_risk_contribution=specific_risk_contrib / total_variance * 100,
            factor_contributions=factor_contributions,
            risk_by_factor_type=risk_by_factor_type,
            risk_by_asset=risk_by_asset
        )

        return attribution

class PCAFactorModel(FactorModel):
    """主成分分析因子模型"""

    def __init__(self, n_components: int = 5):
        super().__init__("PCA Factor Model")
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()

    def fit(self, asset_returns: pd.DataFrame, factor_data: pd.DataFrame = None):
        """拟合PCA因子模型"""
        # 标准化数据
        returns_scaled = self.scaler.fit_transform(asset_returns.dropna())

        # 进行主成分分析
        self.pca = PCA(n_components=self.n_components)
        factor_scores = self.pca.fit_transform(returns_scaled)

        # 创建因子收益时间序列
        factor_names = [f'PC{i+1}' for i in range(self.n_components)]
        self.factor_returns = pd.DataFrame(
            factor_scores,
            index=asset_returns.dropna().index,
            columns=factor_names
        )

        # 因子载荷就是主成分
        self.factor_loadings = pd.DataFrame(
            self.pca.components_,
            index=factor_names,
            columns=asset_returns.columns
        )

        # 计算残差
        reconstructed = self.pca.inverse_transform(factor_scores)
        reconstructed_df = pd.DataFrame(
            self.scaler.inverse_transform(reconstructed),
            index=asset_returns.dropna().index,
            columns=asset_returns.columns
        )

        self.residual_returns = asset_returns.dropna() - reconstructed_df

        # 创建因子对象
        for i, factor_name in enumerate(factor_names):
            factor = RiskFactor(
                factor_id=factor_name,
                factor_name=f'主成分{i+1}',
                factor_type=RiskFactorType.CUSTOM,
                factor_returns=self.factor_returns[factor_name],
                description=f'第{i+1}主成分，解释方差比例: {self.pca.explained_variance_ratio_[i]:.3f}'
            )
            factor.calculate_statistics()
            self.factors[factor_name] = factor

        logger.info(f"PCA因子模型拟合完成，{self.n_components}个主成分，"
                   f"累计解释方差: {self.pca.explained_variance_ratio_.sum():.3f}")

    def calculate_risk_attribution(self, weights: pd.Series) -> PortfolioRiskAttribution:
        """计算PCA风险归因"""
        if self.pca is None:
            raise ValueError("模型未拟合，请先调用fit方法")

        # 计算组合的因子暴露
        portfolio_exposures = self.factor_loadings.dot(weights)

        # 计算因子协方差矩阵
        factor_cov = self.factor_returns.cov() * 252

        # 计算特异风险
        residual_cov = self.residual_returns.cov() * 252
        residual_cov_diag = pd.DataFrame(
            np.diag(np.diag(residual_cov.values)),
            index=residual_cov.index,
            columns=residual_cov.columns
        )

        # 总风险计算
        factor_risk = portfolio_exposures.T @ factor_cov @ portfolio_exposures
        specific_risk = weights.T @ residual_cov_diag @ weights
        total_variance = factor_risk + specific_risk
        total_risk = np.sqrt(total_variance)

        # 因子贡献
        factor_contributions = []
        for factor_name in self.factors.keys():
            exposure = portfolio_exposures[factor_name]
            variance = factor_cov.loc[factor_name, factor_name]

            abs_contrib = (exposure ** 2) * variance
            rel_contrib = abs_contrib / total_variance * 100

            contribution = RiskContribution(
                asset_id="Portfolio",
                factor_id=factor_name,
                absolute_contribution=abs_contrib,
                relative_contribution=rel_contrib,
                marginal_contribution=2 * exposure * variance / (2 * total_risk),
                factor_exposure=exposure,
                factor_return=self.factors[factor_name].factor_returns.mean() * 252
            )
            factor_contributions.append(contribution)

        attribution = PortfolioRiskAttribution(
            portfolio_id="portfolio",
            attribution_date=datetime.now(),
            total_risk=total_risk,
            systematic_risk=np.sqrt(factor_risk),
            specific_risk=np.sqrt(specific_risk),
            specific_risk_contribution=specific_risk / total_variance * 100,
            factor_contributions=factor_contributions
        )

        return attribution

class RiskAttributionEngine:
    """风险归因引擎"""

    def __init__(self):
        self.models = {}
        self.attribution_history = []

        # 注册默认模型
        self._register_default_models()

        # 统计信息
        self.stats = {
            'total_attributions': 0,
            'models_used': defaultdict(int),
            'average_processing_time': 0.0
        }

        logger.info("风险归因引擎初始化完成")

    def _register_default_models(self):
        """注册默认模型"""
        self.models['fama_french'] = FamaFrenchModel()
        self.models['pca'] = PCAFactorModel(n_components=5)

    def register_model(self, model_name: str, model: FactorModel):
        """注册新模型"""
        self.models[model_name] = model
        logger.info(f"注册风险模型: {model_name}")

    def fit_model(self, model_name: str, asset_returns: pd.DataFrame,
                 factor_data: pd.DataFrame = None):
        """拟合指定模型"""
        if model_name not in self.models:
            raise ValueError(f"未找到模型: {model_name}")

        import time
        start_time = time.time()

        try:
            self.models[model_name].fit(asset_returns, factor_data)

            fit_time = time.time() - start_time
            logger.info(f"模型 {model_name} 拟合完成，耗时 {fit_time:.2f}s")

            return True

        except Exception as e:
            logger.error(f"模型 {model_name} 拟合失败: {e}")
            return False

    def calculate_attribution(self, model_name: str, weights: pd.Series,
                            benchmark_weights: pd.Series = None) -> PortfolioRiskAttribution:
        """计算风险归因"""
        if model_name not in self.models:
            raise ValueError(f"未找到模型: {model_name}")

        import time
        start_time = time.time()

        try:
            attribution = self.models[model_name].calculate_risk_attribution(
                weights, benchmark_weights
            )

            # 记录统计信息
            processing_time = time.time() - start_time
            self.stats['total_attributions'] += 1
            self.stats['models_used'][model_name] += 1

            # 更新平均处理时间
            total = self.stats['total_attributions']
            current_avg = self.stats['average_processing_time']
            self.stats['average_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )

            # 记录历史
            self.attribution_history.append({
                'timestamp': datetime.now(),
                'model_name': model_name,
                'processing_time': processing_time,
                'total_risk': attribution.total_risk,
                'systematic_risk': attribution.systematic_risk,
                'specific_risk': attribution.specific_risk
            })

            logger.debug(f"风险归因完成: {model_name}, 总风险: {attribution.total_risk:.4f}")

            return attribution

        except Exception as e:
            logger.error(f"风险归因计算失败 {model_name}: {e}")
            raise

    def compare_models(self, asset_returns: pd.DataFrame, weights: pd.Series,
                      models: List[str] = None) -> Dict[str, Any]:
        """比较不同模型的归因结果"""
        if models is None:
            models = list(self.models.keys())

        comparison_results = {}

        for model_name in models:
            try:
                # 拟合模型
                self.fit_model(model_name, asset_returns)

                # 计算归因
                attribution = self.calculate_attribution(model_name, weights)

                comparison_results[model_name] = {
                    'total_risk': attribution.total_risk,
                    'systematic_risk': attribution.systematic_risk,
                    'specific_risk': attribution.specific_risk,
                    'factor_count': len(attribution.factor_contributions),
                    'top_risk_factors': sorted(
                        attribution.factor_contributions,
                        key=lambda x: abs(x.relative_contribution),
                        reverse=True
                    )[:3]
                }

            except Exception as e:
                logger.error(f"模型 {model_name} 比较失败: {e}")
                comparison_results[model_name] = {'error': str(e)}

        return comparison_results

    def get_factor_analysis(self, model_name: str) -> Dict[str, Any]:
        """获取因子分析"""
        if model_name not in self.models:
            raise ValueError(f"未找到模型: {model_name}")

        model = self.models[model_name]

        if not model.factors:
            return {'message': '模型未拟合或没有因子数据'}

        factor_analysis = {}

        for factor_id, factor in model.factors.items():
            factor_analysis[factor_id] = factor.to_dict()

        # 因子相关性分析
        if not model.factor_returns.empty:
            factor_corr = model.factor_returns.corr()

            factor_analysis['correlation_matrix'] = factor_corr.to_dict()

            # 找出高相关性的因子对
            high_corr_pairs = []
            for i in range(len(factor_corr.columns)):
                for j in range(i+1, len(factor_corr.columns)):
                    corr_value = factor_corr.iloc[i, j]
                    if abs(corr_value) > 0.7:  # 高相关阈值
                        high_corr_pairs.append({
                            'factor1': factor_corr.columns[i],
                            'factor2': factor_corr.columns[j],
                            'correlation': corr_value
                        })

            factor_analysis['high_correlation_pairs'] = high_corr_pairs

        return factor_analysis

    def generate_risk_report(self, model_name: str, weights: pd.Series,
                           benchmark_weights: pd.Series = None) -> Dict[str, Any]:
        """生成风险报告"""
        # 计算归因
        attribution = self.calculate_attribution(model_name, weights, benchmark_weights)

        # 基础信息
        report = {
            'report_date': attribution.attribution_date.isoformat(),
            'model_used': model_name,
            'portfolio_summary': {
                'total_risk': attribution.total_risk,
                'systematic_risk': attribution.systematic_risk,
                'specific_risk': attribution.specific_risk,
                'risk_decomposition': {
                    'systematic_pct': (attribution.systematic_risk ** 2) / (attribution.total_risk ** 2) * 100,
                    'specific_pct': attribution.specific_risk_contribution
                }
            }
        }

        # 因子风险贡献排序
        factor_contributions = sorted(
            attribution.factor_contributions,
            key=lambda x: abs(x.relative_contribution),
            reverse=True
        )

        report['top_risk_factors'] = [
            {
                'factor_name': fc.factor_id,
                'contribution_pct': fc.relative_contribution,
                'exposure': fc.factor_exposure,
                'significance': 'High' if abs(fc.relative_contribution) > 10 else
                              'Medium' if abs(fc.relative_contribution) > 5 else 'Low'
            }
            for fc in factor_contributions[:10]
        ]

        # 因子类型风险分析
        if attribution.risk_by_factor_type:
            report['risk_by_factor_type'] = attribution.risk_by_factor_type

        # 资产风险贡献
        if attribution.risk_by_asset:
            top_risk_assets = sorted(
                attribution.risk_by_asset.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

            report['top_risk_assets'] = [
                {'asset': asset, 'contribution_pct': contrib}
                for asset, contrib in top_risk_assets
            ]

        # 风险建议
        recommendations = []

        # 基于因子暴露的建议
        for fc in factor_contributions[:3]:
            if abs(fc.relative_contribution) > 15:
                if fc.relative_contribution > 0:
                    recommendations.append(f"考虑降低对{fc.factor_id}因子的暴露")
                else:
                    recommendations.append(f"可以考虑增加对{fc.factor_id}因子的暴露")

        # 基于特异风险的建议
        if attribution.specific_risk_contribution > 30:
            recommendations.append("特异风险过高，建议增加分散化")

        report['recommendations'] = recommendations

        return report

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return {
            'stats': self.stats,
            'registered_models': list(self.models.keys()),
            'attribution_history_size': len(self.attribution_history),
            'recent_attributions': self.attribution_history[-10:] if self.attribution_history else []
        }

class RiskAttributionSystem:
    """风险归因系统主类"""

    def __init__(self):
        self.attribution_engine = RiskAttributionEngine()

        # 数据缓存
        self.asset_returns_cache = {}
        self.factor_data_cache = {}

        # 系统统计
        self.system_stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("风险归因系统初始化完成")

    def analyze_portfolio_risk(self, portfolio_id: str, asset_returns: pd.DataFrame,
                             weights: pd.Series, model_name: str = 'fama_french',
                             benchmark_weights: pd.Series = None,
                             factor_data: pd.DataFrame = None) -> Dict[str, Any]:
        """分析投资组合风险"""

        try:
            # 拟合模型
            success = self.attribution_engine.fit_model(model_name, asset_returns, factor_data)
            if not success:
                return {'error': f'模型 {model_name} 拟合失败'}

            # 计算风险归因
            attribution = self.attribution_engine.calculate_attribution(
                model_name, weights, benchmark_weights
            )

            # 生成详细报告
            risk_report = self.attribution_engine.generate_risk_report(
                model_name, weights, benchmark_weights
            )

            # 因子分析
            factor_analysis = self.attribution_engine.get_factor_analysis(model_name)

            # 更新系统统计
            self.system_stats['total_analyses'] += 1

            result = {
                'portfolio_id': portfolio_id,
                'analysis_completed': True,
                'attribution': attribution.to_dict(),
                'risk_report': risk_report,
                'factor_analysis': factor_analysis,
                'model_info': {
                    'model_name': model_name,
                    'model_description': self.attribution_engine.models[model_name].name
                }
            }

            logger.info(f"投资组合风险分析完成: {portfolio_id}")
            return result

        except Exception as e:
            logger.error(f"投资组合风险分析失败 {portfolio_id}: {e}")
            return {
                'portfolio_id': portfolio_id,
                'analysis_completed': False,
                'error': str(e)
            }

    def compare_portfolio_models(self, portfolio_id: str, asset_returns: pd.DataFrame,
                               weights: pd.Series, models: List[str] = None) -> Dict[str, Any]:
        """比较不同模型的分析结果"""

        comparison_results = self.attribution_engine.compare_models(
            asset_returns, weights, models
        )

        # 分析比较结果
        model_comparison = {}

        for model_name, results in comparison_results.items():
            if 'error' not in results:
                model_comparison[model_name] = {
                    'total_risk': results['total_risk'],
                    'systematic_risk_ratio': (results['systematic_risk'] ** 2) / (results['total_risk'] ** 2),
                    'factor_count': results['factor_count'],
                    'model_complexity': 'High' if results['factor_count'] > 10 else
                                      'Medium' if results['factor_count'] > 5 else 'Low'
                }

        return {
            'portfolio_id': portfolio_id,
            'model_comparison': model_comparison,
            'detailed_results': comparison_results,
            'recommendation': self._recommend_best_model(comparison_results)
        }

    def _recommend_best_model(self, comparison_results: Dict[str, Any]) -> str:
        """推荐最佳模型"""
        valid_models = {k: v for k, v in comparison_results.items() if 'error' not in v}

        if not valid_models:
            return "无有效模型"

        # 简单的模型选择逻辑
        if 'fama_french' in valid_models and len(valid_models) > 1:
            return "fama_french - 经典三因子模型，解释性强"
        elif 'pca' in valid_models:
            return "pca - 主成分分析模型，数据驱动"
        else:
            return f"{list(valid_models.keys())[0]} - 可用的模型"

    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        engine_stats = self.attribution_engine.get_engine_stats()

        return {
            'system_stats': self.system_stats,
            'engine_stats': engine_stats['stats'],
            'available_models': engine_stats['registered_models'],
            'cache_stats': {
                'returns_cache_size': len(self.asset_returns_cache),
                'factor_cache_size': len(self.factor_data_cache)
            }
        }

# 全局实例
_risk_attribution_system_instance = None

def get_risk_attribution_system() -> RiskAttributionSystem:
    """获取风险归因系统实例"""
    global _risk_attribution_system_instance
    if _risk_attribution_system_instance is None:
        _risk_attribution_system_instance = RiskAttributionSystem()
    return _risk_attribution_system_instance
