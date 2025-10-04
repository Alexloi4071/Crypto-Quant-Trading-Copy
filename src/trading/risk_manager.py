"""
Risk Manager Module
Comprehensive risk management for trading operations
Includes portfolio risk, position risk, and system-wide risk controls
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger, get_trading_logger
from src.utils.helpers import timing_decorator, calculate_sharpe_ratio, calculate_max_drawdown

logger = setup_logger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskAlert(Enum):
    """Risk alert types"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    DAILY_LOSS = "daily_loss"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    SYSTEM_ERROR = "system_error"

@dataclass

class RiskMetric:
    """Risk metric data structure"""
    name: str
    value: float
    threshold: float
    risk_level: RiskLevel
    description: str
    timestamp: datetime

    def is_breached(self) -> bool:
        """Check if risk threshold is breached"""
        return self.value > self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'risk_level': self.risk_level.value,
            'description': self.description,
            'breached': self.is_breached(),
            'timestamp': self.timestamp.isoformat()
        }

class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self):
        # Risk thresholds (can be configured)
        self.risk_thresholds = {
            'max_position_size_pct': config.get('MAX_POSITION_SIZE', 0.05),      # 5%
            'max_portfolio_exposure_pct': config.get('MAX_TOTAL_EXPOSURE', 0.8), # 80%
            'max_daily_loss_pct': config.get('MAX_DAILY_LOSS', 0.02),            # 2%
            'max_drawdown_pct': config.get('MAX_DRAWDOWN', 0.15),                # 15%
            'max_volatility': config.get('MAX_VOLATILITY', 0.05),                # 5% daily
            'max_correlation': config.get('MAX_CORRELATION', 0.7),               # 70%
            'max_consecutive_losses': config.get('MAX_CONSECUTIVE_LOSSES', 5),   # 5 trades
            'min_sharpe_ratio': config.get('MIN_SHARPE_RATIO', -0.5),           # -0.5
            'max_var_95': config.get('MAX_VAR_95', 0.03),                       # 3% VaR
            'max_positions': config.get('MAX_POSITIONS', 10)                     # 10 positions
        }

        # Risk monitoring
        self.risk_metrics: Dict[str, RiskMetric] = {}
        self.risk_alerts: List[Dict[str, Any]] = []
        self.position_correlations: Dict[Tuple[str, str], float] = {}

        # Performance tracking
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []

        # Risk controls
        self.trading_enabled = True
        self.emergency_stop = False
        self.position_limits_active = True

        logger.info("Risk manager initialized")

    @timing_decorator

    def assess_portfolio_risk(self, portfolio_value: float,
                            positions: Dict[str, Any],
                            daily_return: float = None) -> Dict[str, Any]:
        """Comprehensive portfolio risk assessment"""
        try:
            logger.debug("Assessing portfolio risk")

            risk_assessment = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'risk_metrics': {},
                'alerts': [],
                'overall_risk_level': RiskLevel.LOW,
                'recommendations': []
            }

            # Update performance tracking
            self.portfolio_values.append(portfolio_value)
            if daily_return is not None:
                self.daily_returns.append(daily_return)

            # Keep only recent data (last 252 days for annual calculations)
            if len(self.portfolio_values) > 252:
                self.portfolio_values = self.portfolio_values[-252:]
            if len(self.daily_returns) > 252:
                self.daily_returns = self.daily_returns[-252:]

            # 1. Position Size Risk
            position_risk = self._assess_position_size_risk(positions, portfolio_value)
            risk_assessment['risk_metrics']['position_size'] = position_risk

            # 2. Portfolio Exposure Risk
            exposure_risk = self._assess_exposure_risk(positions, portfolio_value)
            risk_assessment['risk_metrics']['exposure'] = exposure_risk

            # 3. Drawdown Risk
            drawdown_risk = self._assess_drawdown_risk()
            risk_assessment['risk_metrics']['drawdown'] = drawdown_risk

            # 4. Volatility Risk
            volatility_risk = self._assess_volatility_risk()
            risk_assessment['risk_metrics']['volatility'] = volatility_risk

            # 5. Correlation Risk
            correlation_risk = self._assess_correlation_risk(positions)
            risk_assessment['risk_metrics']['correlation'] = correlation_risk

            # 6. Daily Loss Risk
            daily_loss_risk = self._assess_daily_loss_risk(daily_return)
            risk_assessment['risk_metrics']['daily_loss'] = daily_loss_risk

            # 7. Performance Risk
            performance_risk = self._assess_performance_risk()
            risk_assessment['risk_metrics']['performance'] = performance_risk

            # Collect all alerts
            all_metrics = risk_assessment['risk_metrics']
            for metric_name, metric in all_metrics.items():
                if metric and metric.is_breached():
                    alert = {
                        'type': metric_name,
                        'level': metric.risk_level.value,
                        'description': metric.description,
                        'value': metric.value,
                        'threshold': metric.threshold,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    risk_assessment['alerts'].append(alert)
                    self.risk_alerts.append(alert)

            # Determine overall risk level
            overall_risk = self._calculate_overall_risk_level(all_metrics)
            risk_assessment['overall_risk_level'] = overall_risk

            # Generate recommendations
            recommendations = self._generate_risk_recommendations(all_metrics)
            risk_assessment['recommendations'] = recommendations

            # Update risk controls
            self._update_risk_controls(overall_risk, risk_assessment['alerts'])

            # Keep only recent alerts (last 100)
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-100:]

            logger.debug(f"Risk assessment completed. Overall risk: {overall_risk.value}")
            return risk_assessment

        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return {'error': str(e), 'overall_risk_level': RiskLevel.CRITICAL}

    def _assess_position_size_risk(self, positions: Dict[str, Any],
                                 portfolio_value: float) -> RiskMetric:
        """Assess individual position size risk"""
        try:
            if not positions or portfolio_value <= 0:
                return RiskMetric(
                    name="position_size",
                    value=0.0,
                    threshold=self.risk_thresholds['max_position_size_pct'],
                    risk_level=RiskLevel.LOW,
                    description="No positions to assess",
                    timestamp=datetime.now()
                )

            # Find largest position as percentage of portfolio
            max_position_pct = 0.0
            largest_symbol = ""

            for symbol, position in positions.items():
                if 'quantity' in position and 'current_price' in position:
                    position_value = abs(position['quantity'] * position['current_price'])
                    position_pct = position_value / portfolio_value

                    if position_pct > max_position_pct:
                        max_position_pct = position_pct
                        largest_symbol = symbol

            # Determine risk level
            threshold = self.risk_thresholds['max_position_size_pct']

            if max_position_pct > threshold * 1.5:
                risk_level = RiskLevel.CRITICAL
            elif max_position_pct > threshold:
                risk_level = RiskLevel.HIGH
            elif max_position_pct > threshold * 0.8:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            description = f"Largest position ({largest_symbol}): {max_position_pct:.1%} of portfolio"

            return RiskMetric(
                name="position_size",
                value=max_position_pct,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Position size risk assessment failed: {e}")
            return RiskMetric(
                name="position_size",
                value=1.0,  # Assume high risk on error
                threshold=self.risk_thresholds['max_position_size_pct'],
                risk_level=RiskLevel.CRITICAL,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _assess_exposure_risk(self, positions: Dict[str, Any],
                            portfolio_value: float) -> RiskMetric:
        """Assess total portfolio exposure risk"""
        try:
            if not positions or portfolio_value <= 0:
                return RiskMetric(
                    name="exposure",
                    value=0.0,
                    threshold=self.risk_thresholds['max_portfolio_exposure_pct'],
                    risk_level=RiskLevel.LOW,
                    description="No exposure",
                    timestamp=datetime.now()
                )

            # Calculate total exposure
            total_exposure = 0.0

            for position in positions.values():
                if 'quantity' in position and 'current_price' in position:
                    position_value = abs(position['quantity'] * position['current_price'])
                    total_exposure += position_value

            exposure_pct = total_exposure / portfolio_value
            threshold = self.risk_thresholds['max_portfolio_exposure_pct']

            # Determine risk level
            if exposure_pct > threshold * 1.2:
                risk_level = RiskLevel.CRITICAL
            elif exposure_pct > threshold:
                risk_level = RiskLevel.HIGH
            elif exposure_pct > threshold * 0.9:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            description = f"Total portfolio exposure: {exposure_pct:.1%}"

            return RiskMetric(
                name="exposure",
                value=exposure_pct,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Exposure risk assessment failed: {e}")
            return RiskMetric(
                name="exposure",
                value=1.0,
                threshold=self.risk_thresholds['max_portfolio_exposure_pct'],
                risk_level=RiskLevel.CRITICAL,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _assess_drawdown_risk(self) -> RiskMetric:
        """Assess maximum drawdown risk"""
        try:
            if len(self.portfolio_values) < 2:
                return RiskMetric(
                    name="drawdown",
                    value=0.0,
                    threshold=self.risk_thresholds['max_drawdown_pct'],
                    risk_level=RiskLevel.LOW,
                    description="Insufficient data for drawdown calculation",
                    timestamp=datetime.now()
                )

            # Calculate current drawdown
            portfolio_series = pd.Series(self.portfolio_values)
            max_drawdown, _, _ = calculate_max_drawdown(portfolio_series)
            max_drawdown = abs(max_drawdown)  # Ensure positive

            threshold = self.risk_thresholds['max_drawdown_pct']

            # Determine risk level
            if max_drawdown > threshold * 1.5:
                risk_level = RiskLevel.CRITICAL
            elif max_drawdown > threshold:
                risk_level = RiskLevel.HIGH
            elif max_drawdown > threshold * 0.7:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            description = f"Maximum drawdown: {max_drawdown:.1%}"

            return RiskMetric(
                name="drawdown",
                value=max_drawdown,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Drawdown risk assessment failed: {e}")
            return RiskMetric(
                name="drawdown",
                value=0.5,  # Assume moderate risk
                threshold=self.risk_thresholds['max_drawdown_pct'],
                risk_level=RiskLevel.HIGH,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _assess_volatility_risk(self) -> RiskMetric:
        """Assess portfolio volatility risk"""
        try:
            if len(self.daily_returns) < 10:
                return RiskMetric(
                    name="volatility",
                    value=0.0,
                    threshold=self.risk_thresholds['max_volatility'],
                    risk_level=RiskLevel.LOW,
                    description="Insufficient data for volatility calculation",
                    timestamp=datetime.now()
                )

            # Calculate rolling volatility (last 30 days)
            recent_returns = self.daily_returns[-30:]
            volatility = np.std(recent_returns)

            threshold = self.risk_thresholds['max_volatility']

            # Determine risk level
            if volatility > threshold * 2:
                risk_level = RiskLevel.CRITICAL
            elif volatility > threshold:
                risk_level = RiskLevel.HIGH
            elif volatility > threshold * 0.7:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            description = f"30-day volatility: {volatility:.2%}"

            return RiskMetric(
                name="volatility",
                value=volatility,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Volatility risk assessment failed: {e}")
            return RiskMetric(
                name="volatility",
                value=0.1,
                threshold=self.risk_thresholds['max_volatility'],
                risk_level=RiskLevel.HIGH,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _assess_correlation_risk(self, positions: Dict[str, Any]) -> RiskMetric:
        """Assess position correlation risk"""
        try:
            if len(positions) < 2:
                return RiskMetric(
                    name="correlation",
                    value=0.0,
                    threshold=self.risk_thresholds['max_correlation'],
                    risk_level=RiskLevel.LOW,
                    description="Less than 2 positions for correlation analysis",
                    timestamp=datetime.now()
                )

            # This is a simplified correlation assessment
            # In practice, you'd calculate actual price correlations
            symbols = list(positions.keys())

            # Check for crypto-crypto correlations (simplified)
            crypto_symbols = 0
            for symbol in symbols:
                if 'USDT' in symbol or 'USD' in symbol or 'BTC' in symbol or 'ETH' in symbol:
                    crypto_symbols += 1

            # If most positions are crypto, assume high correlation
            correlation_ratio = crypto_symbols / len(symbols)

            threshold = self.risk_thresholds['max_correlation']

            # Determine risk level based on correlation proxy
            if correlation_ratio > 0.9:
                risk_level = RiskLevel.HIGH
                avg_correlation = 0.8  # Assume high correlation
            elif correlation_ratio > 0.7:
                risk_level = RiskLevel.MEDIUM
                avg_correlation = 0.6
            else:
                risk_level = RiskLevel.LOW
                avg_correlation = 0.3

            description = f"Estimated average correlation: {avg_correlation:.1%}"

            return RiskMetric(
                name="correlation",
                value=avg_correlation,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Correlation risk assessment failed: {e}")
            return RiskMetric(
                name="correlation",
                value=0.5,
                threshold=self.risk_thresholds['max_correlation'],
                risk_level=RiskLevel.MEDIUM,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _assess_daily_loss_risk(self, daily_return: float = None) -> RiskMetric:
        """Assess daily loss risk"""
        try:
            if daily_return is None:
                if len(self.daily_returns) == 0:
                    return RiskMetric(
                        name="daily_loss",
                        value=0.0,
                        threshold=self.risk_thresholds['max_daily_loss_pct'],
                        risk_level=RiskLevel.LOW,
                        description="No daily return data",
                        timestamp=datetime.now()
                    )
                daily_return = self.daily_returns[-1]

            # Use absolute value for comparison (losses are negative)
            daily_loss = abs(min(0, daily_return))  # Only consider losses
            threshold = self.risk_thresholds['max_daily_loss_pct']

            # Determine risk level
            if daily_loss > threshold * 2:
                risk_level = RiskLevel.CRITICAL
            elif daily_loss > threshold:
                risk_level = RiskLevel.HIGH
            elif daily_loss > threshold * 0.5:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            if daily_return >= 0:
                description = f"Daily gain: {daily_return:.2%}"
            else:
                description = f"Daily loss: {daily_loss:.2%}"

            return RiskMetric(
                name="daily_loss",
                value=daily_loss,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Daily loss risk assessment failed: {e}")
            return RiskMetric(
                name="daily_loss",
                value=0.0,
                threshold=self.risk_thresholds['max_daily_loss_pct'],
                risk_level=RiskLevel.LOW,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _assess_performance_risk(self) -> RiskMetric:
        """Assess overall performance risk"""
        try:
            if len(self.daily_returns) < 30:
                return RiskMetric(
                    name="performance",
                    value=0.0,
                    threshold=abs(self.risk_thresholds['min_sharpe_ratio']),
                    risk_level=RiskLevel.LOW,
                    description="Insufficient data for performance analysis",
                    timestamp=datetime.now()
                )

            # Calculate Sharpe ratio
            returns_series = pd.Series(self.daily_returns)
            sharpe_ratio = calculate_sharpe_ratio(returns_series)

            # Convert to risk metric (negative Sharpe is risk)
            performance_risk = max(0, abs(min(0, sharpe_ratio)))  # Risk when Sharpe < 0
            threshold = abs(self.risk_thresholds['min_sharpe_ratio'])

            # Determine risk level
            if sharpe_ratio < -1.0:
                risk_level = RiskLevel.CRITICAL
            elif sharpe_ratio < -0.5:
                risk_level = RiskLevel.HIGH
            elif sharpe_ratio < 0:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            description = f"Sharpe ratio: {sharpe_ratio:.2f}"

            return RiskMetric(
                name="performance",
                value=performance_risk,
                threshold=threshold,
                risk_level=risk_level,
                description=description,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Performance risk assessment failed: {e}")
            return RiskMetric(
                name="performance",
                value=0.5,
                threshold=abs(self.risk_thresholds['min_sharpe_ratio']),
                risk_level=RiskLevel.MEDIUM,
                description=f"Assessment error: {e}",
                timestamp=datetime.now()
            )

    def _calculate_overall_risk_level(self, metrics: Dict[str, RiskMetric]) -> RiskLevel:
        """Calculate overall risk level from individual metrics"""
        try:
            risk_scores = []

            for metric in metrics.values():
                if metric:
                    if metric.risk_level == RiskLevel.CRITICAL:
                        risk_scores.append(4)
                    elif metric.risk_level == RiskLevel.HIGH:
                        risk_scores.append(3)
                    elif metric.risk_level == RiskLevel.MEDIUM:
                        risk_scores.append(2)
                    else:
                        risk_scores.append(1)

            if not risk_scores:
                return RiskLevel.LOW

            avg_score = np.mean(risk_scores)
            max_score = max(risk_scores)

            # Overall risk is influenced by both average and maximum
            # If any metric is critical, overall is at least high
            if max_score >= 4 or avg_score >= 3.5:
                return RiskLevel.CRITICAL
            elif max_score >= 3 or avg_score >= 2.5:
                return RiskLevel.HIGH
            elif avg_score >= 1.5:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception as e:
            logger.error(f"Overall risk calculation failed: {e}")
            return RiskLevel.HIGH

    def _generate_risk_recommendations(self, metrics: Dict[str, RiskMetric]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []

        try:
            for metric_name, metric in metrics.items():
                if metric and metric.is_breached():
                    if metric_name == "position_size":
                        recommendations.append("Reduce individual position sizes")
                    elif metric_name == "exposure":
                        recommendations.append("Reduce total portfolio exposure")
                    elif metric_name == "drawdown":
                        recommendations.append("Consider stopping trading until drawdown recovers")
                    elif metric_name == "volatility":
                        recommendations.append("Reduce position sizes due to high volatility")
                    elif metric_name == "correlation":
                        recommendations.append("Diversify positions across different asset classes")
                    elif metric_name == "daily_loss":
                        recommendations.append("Review and tighten stop-loss rules")
                    elif metric_name == "performance":
                        recommendations.append("Review trading strategy performance")

            # General recommendations based on overall situation
            if len([m for m in metrics.values() if m and m.risk_level == RiskLevel.CRITICAL]) > 0:
                recommendations.append("CRITICAL: Consider emergency stop of all trading")

            if not recommendations:
                recommendations.append("Risk levels within acceptable limits")

        except Exception as e:
            logger.error(f"Risk recommendations generation failed: {e}")
            recommendations.append("Unable to generate recommendations due to error")

        return recommendations

    def _update_risk_controls(self, overall_risk: RiskLevel, alerts: List[Dict[str, Any]]):
        """Update risk controls based on assessment"""
        try:
            # Emergency stop conditions
            critical_alerts = [a for a in alerts if a['level'] == 'critical']

            if len(critical_alerts) >= 2:
                self.emergency_stop = True
                self.trading_enabled = False
                logger.critical("EMERGENCY STOP: Multiple critical risk alerts triggered")
            elif overall_risk == RiskLevel.CRITICAL:
                self.trading_enabled = False
                logger.warning("Trading disabled due to critical risk level")
            elif overall_risk == RiskLevel.HIGH:
                self.position_limits_active = True
                logger.warning("Position limits activated due to high risk")
            else:
                # Reset controls if risk is manageable
                if not self.emergency_stop:
                    self.trading_enabled = True
                    self.position_limits_active = False

        except Exception as e:
            logger.error(f"Risk controls update failed: {e}")

    def check_trading_allowed(self, symbol: str = None,
                            position_size: float = None) -> Tuple[bool, str]:
        """Check if trading is allowed given current risk conditions"""
        try:
            if self.emergency_stop:
                return False, "Trading stopped: Emergency stop activated"

            if not self.trading_enabled:
                return False, "Trading disabled due to high risk conditions"

            # Additional checks can be added here based on specific requirements
            # For example, symbol-specific limits, time-based restrictions, etc.

            return True, "Trading allowed"

        except Exception as e:
            logger.error(f"Trading allowance check failed: {e}")
            return False, f"Check failed: {e}"

    def add_trade_result(self, trade: Dict[str, Any]):
        """Add trade result for risk tracking"""
        try:
            trade_data = {
                'timestamp': datetime.now(),
                'symbol': trade.get('symbol', 'Unknown'),
                'pnl': trade.get('pnl', 0.0),
                'pnl_pct': trade.get('pnl_pct', 0.0),
                'duration_hours': trade.get('duration_hours', 0),
                'side': trade.get('side', 'unknown')
            }

            self.trade_history.append(trade_data)

            # Keep only recent trades (last 1000)
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]

        except Exception as e:
            logger.error(f"Failed to add trade result: {e}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        try:
            recent_alerts = [a for a in self.risk_alerts[-20:]]  # Last 20 alerts

            # Calculate recent performance
            recent_returns = self.daily_returns[-30:] if len(self.daily_returns) >= 30 else self.daily_returns
            recent_performance = {
                'avg_daily_return': np.mean(recent_returns) if recent_returns else 0,
                'volatility': np.std(recent_returns) if len(recent_returns) > 1 else 0,
                'sharpe_ratio': calculate_sharpe_ratio(pd.Series(recent_returns)) if recent_returns else 0
            }

            # Trading statistics
            recent_trades = self.trade_history[-50:] if len(self.trade_history) >= 50 else self.trade_history
            winning_trades = [t for t in recent_trades if t['pnl'] > 0]

            trade_stats = {
                'total_trades': len(recent_trades),
                'winning_trades': len(winning_trades),
                'win_rate': len(winning_trades) / len(recent_trades) if recent_trades else 0,
                'avg_pnl': np.mean([t['pnl'] for t in recent_trades]) if recent_trades else 0
            }

            return {
                'timestamp': datetime.now().isoformat(),
                'trading_enabled': self.trading_enabled,
                'emergency_stop': self.emergency_stop,
                'position_limits_active': self.position_limits_active,
                'recent_alerts': recent_alerts,
                'recent_performance': recent_performance,
                'trade_statistics': trade_stats,
                'risk_thresholds': self.risk_thresholds
            }

        except Exception as e:
            logger.error(f"Risk summary generation failed: {e}")
            return {'error': str(e)}

    def reset_emergency_stop(self, confirmation: bool = False):
        """Reset emergency stop (use with caution)"""
        if not confirmation:
            logger.warning("Emergency stop reset requires confirmation=True")
            return False

        try:
            self.emergency_stop = False
            self.trading_enabled = True
            self.position_limits_active = False

            logger.warning("Emergency stop has been manually reset")
            return True

        except Exception as e:
            logger.error(f"Failed to reset emergency stop: {e}")
            return False

    def update_risk_thresholds(self, new_thresholds: Dict[str, float]):
        """Update risk thresholds"""
        try:
            for key, value in new_thresholds.items():
                if key in self.risk_thresholds:
                    old_value = self.risk_thresholds[key]
                    self.risk_thresholds[key] = value
                    logger.info(f"Risk threshold updated: {key} {old_value} -> {value}")
                else:
                    logger.warning(f"Unknown risk threshold: {key}")

        except Exception as e:
            logger.error(f"Risk threshold update failed: {e}")

# Convenience functions

def create_risk_manager() -> RiskManager:
    """Create a risk manager with default settings"""
    return RiskManager()

def assess_position_risk(position_size_pct: float, portfolio_exposure_pct: float,
                        correlation: float = 0.5) -> Dict[str, Any]:
    """Quick position risk assessment"""
    risk_manager = RiskManager()

    # Create mock position data for assessment
    mock_positions = {
        'TEST': {
            'quantity': 1.0,
            'current_price': position_size_pct * 10000  # Assume $10k portfolio
        }
    }

    assessment = risk_manager.assess_portfolio_risk(10000, mock_positions)
    return assessment

# Usage example
if __name__ == "__main__":
    # Create risk manager
    risk_manager = RiskManager()

    # Simulate some portfolio data
    mock_positions = {
        'BTCUSDT': {
            'quantity': 0.1,
            'current_price': 45000,
            'side': 'long'
        },
        'ETHUSDT': {
            'quantity': 2.0,
            'current_price': 3000,
            'side': 'long'
        }
    }

    # Assess risk
    assessment = risk_manager.assess_portfolio_risk(50000, mock_positions, 0.02)

    print("Risk Assessment Results:")
    print(f"Overall Risk Level: {assessment['overall_risk_level'].value}")
    print(f"Alerts: {len(assessment['alerts'])}")
    print(f"Recommendations: {len(assessment['recommendations'])}")

    # Check if trading is allowed
    allowed, reason = risk_manager.check_trading_allowed()
    print(f"Trading Allowed: {allowed} - {reason}")

    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print(f"Emergency Stop: {summary['emergency_stop']}")
    print(f"Trading Enabled: {summary['trading_enabled']}")

    print("Risk manager test completed!")
