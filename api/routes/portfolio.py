"""
Portfolio API Routes
组合管理API路由，集成现有的PositionManager和RiskManager
提供完整的组合查询、分析和管理功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional, List
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入依赖和配置
from api.dependencies import (
    get_trading_system, get_trading_system_optional, get_data_manager,
    get_pagination, get_query_params, get_current_user, create_response, create_error_response
)

# 导入现有系统组件
from src.trading.trading_system import TradingSystem
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter()

@router.get("/summary")

async def get_portfolio_summary(
    request: Request,
    include_history: bool = Query(False, description="是否包含历史数据"),
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取组合概览信息
    集成现有的TradingSystem.get_portfolio_summary()
    """
    try:
        if trading_system is None:
            # 如果交易系统未启动，返回模拟数据
            summary = {
                "portfolio_value": 10000.0,
                "initial_capital": 10000.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "daily_pnl": 0.0,
                "daily_return_pct": 0.0,
                "cash": 10000.0,
                "positions": [],
                "risk_metrics": {
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "volatility": 0.0
                },
                "status": "not_trading"
            }
        else:
            # 使用现有系统获取真实数据
            summary = trading_system.get_portfolio_summary()

            # 如果需要包含历史数据
            if include_history:
                # 获取最近30天的组合价值历史
                history = trading_system.get_portfolio_history(days=30)
                summary["history"] = history

            summary["status"] = "active"

        # 添加额外的分析数据
        summary["last_updated"] = datetime.now().isoformat() + "Z"
        summary["currency"] = "USD"

        return await create_response(
            data=summary,
            message="组合概览获取成功"
        )

    except Exception as e:
        logger.error(f"获取组合概览失败: {e}")
        return await create_error_response(
            message="获取组合概览失败",
            code="PORTFOLIO_SUMMARY_ERROR",
            details=str(e)
        )

@router.get("/positions")

async def get_positions(
    request: Request,
    symbol: Optional[str] = Query(None, description="筛选特定交易对"),
    status: Optional[str] = Query(None, description="筛选仓位状态: open, closed, all"),
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    pagination = Depends(get_pagination),
    current_user: dict = Depends(get_current_user)
):
    """
    获取持仓信息
    集成现有的PositionManager功能
    """
    try:
        if trading_system is None:
            positions = []
        else:
            # 使用现有的PositionManager获取持仓
            position_manager = trading_system.position_manager
            if position_manager:
                all_positions = position_manager.get_all_positions()

                # 应用筛选条件
                positions = []
                for pos_id, position in all_positions.items():
                    # 符号筛选
                    if symbol and position.get('symbol', '').upper() != symbol.upper():
                        continue

                    # 状态筛选
                    pos_status = position.get('status', 'open')
                    if status and status != 'all' and pos_status != status:
                        continue

                    # 添加额外信息
                    position_info = {
                        "id": pos_id,
                        "symbol": position.get('symbol'),
                        "side": position.get('side'),
                        "size": position.get('size'),
                        "entry_price": position.get('entry_price'),
                        "current_price": position.get('current_price'),
                        "pnl": position.get('pnl', 0),
                        "pnl_pct": position.get('pnl_pct', 0),
                        "unrealized_pnl": position.get('unrealized_pnl', 0),
                        "stop_loss": position.get('stop_loss'),
                        "take_profit": position.get('take_profit'),
                        "opened_at": position.get('opened_at'),
                        "updated_at": position.get('updated_at'),
                        "status": pos_status,
                        "risk_level": position.get('risk_level', 'medium')
                    }
                    positions.append(position_info)
            else:
                positions = []

        # 应用分页
        total_count = len(positions)
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_positions = positions[start_idx:end_idx]

        # 计算汇总信息
        summary = {
            "total_positions": total_count,
            "open_positions": len([p for p in positions if p.get('status') == 'open']),
            "closed_positions": len([p for p in positions if p.get('status') == 'closed']),
            "total_pnl": sum(p.get('pnl', 0) for p in positions),
            "total_unrealized_pnl": sum(p.get('unrealized_pnl', 0) for p in positions if p.get('status') == 'open')
        }

        return await create_response(
            data={
                "positions": paginated_positions,
                "summary": summary,
                "pagination": {
                    "page": pagination.page,
                    "size": pagination.size,
                    "total": total_count,
                    "pages": (total_count + pagination.size - 1) // pagination.size
                }
            },
            message="持仓信息获取成功"
        )

    except Exception as e:
        logger.error(f"获取持仓信息失败: {e}")
        return await create_error_response(
            message="获取持仓信息失败",
            code="POSITIONS_ERROR",
            details=str(e)
        )

@router.get("/positions/{position_id}")

async def get_position_detail(
    position_id: str,
    request: Request,
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取特定持仓的详细信息
    """
    try:
        if trading_system is None or trading_system.position_manager is None:
            raise HTTPException(
                status_code=404,
                detail="交易系统未启动或持仓不存在"
            )

        # 从PositionManager获取持仓详情
        position = trading_system.position_manager.get_position(position_id)
        if not position:
            raise HTTPException(
                status_code=404,
                detail=f"持仓 {position_id} 不存在"
            )

        # 获取持仓的详细历史和分析
        position_detail = {
            "id": position_id,
            "symbol": position.get('symbol'),
            "side": position.get('side'),
            "size": position.get('size'),
            "entry_price": position.get('entry_price'),
            "current_price": position.get('current_price'),
            "pnl": position.get('pnl', 0),
            "pnl_pct": position.get('pnl_pct', 0),
            "unrealized_pnl": position.get('unrealized_pnl', 0),
            "stop_loss": position.get('stop_loss'),
            "take_profit": position.get('take_profit'),
            "opened_at": position.get('opened_at'),
            "updated_at": position.get('updated_at'),
            "status": position.get('status', 'open'),

            # 额外分析信息
            "entry_signal": position.get('entry_signal'),
            "strategy": position.get('strategy'),
            "risk_level": position.get('risk_level', 'medium'),
            "max_profit": position.get('max_profit', 0),
            "max_loss": position.get('max_loss', 0),
            "holding_duration": position.get('holding_duration'),

            # 实时市场信息
            "market_data": {
                "current_price": position.get('current_price'),
                "price_change_24h": position.get('price_change_24h', 0),
                "volume_24h": position.get('volume_24h', 0)
            }
        }

        # 如果持仓仍然开启，计算实时风险指标
        if position.get('status') == 'open' and trading_system.risk_manager:
            risk_assessment = trading_system.risk_manager.check_position_risk(position)
            position_detail["risk_assessment"] = risk_assessment

        return await create_response(
            data=position_detail,
            message="持仓详情获取成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取持仓详情失败: {e}")
        return await create_error_response(
            message="获取持仓详情失败",
            code="POSITION_DETAIL_ERROR",
            details=str(e)
        )

@router.post("/positions/{position_id}/close")

async def close_position(
    position_id: str,
    request: Request,
    reason: Optional[str] = Query("Manual close", description="平仓原因"),
    trading_system: TradingSystem = Depends(get_trading_system),
    current_user: dict = Depends(get_current_user)
):
    """
    手动平仓
    集成现有的PositionManager.close_position()
    """
    try:
        if trading_system.position_manager is None:
            raise HTTPException(
                status_code=503,
                detail="持仓管理器不可用"
            )

        # 检查持仓是否存在
        position = trading_system.position_manager.get_position(position_id)
        if not position:
            raise HTTPException(
                status_code=404,
                detail=f"持仓 {position_id} 不存在"
            )

        if position.get('status') != 'open':
            raise HTTPException(
                status_code=400,
                detail="只能平仓开启状态的持仓"
            )

        # 调用现有的平仓功能
        result = await trading_system.position_manager.close_position(
            position_id,
            reason=reason
        )

        if result.get('success'):
            return await create_response(
                data={
                    "position_id": position_id,
                    "status": "closed",
                    "close_price": result.get('close_price'),
                    "pnl": result.get('pnl'),
                    "reason": reason,
                    "closed_at": datetime.now().isoformat() + "Z"
                },
                message="持仓平仓成功"
            )
        else:
            return await create_error_response(
                message="持仓平仓失败",
                code="CLOSE_POSITION_ERROR",
                details=result.get('error', '未知错误')
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"平仓失败: {e}")
        return await create_error_response(
            message="平仓操作失败",
            code="CLOSE_POSITION_ERROR",
            details=str(e)
        )

@router.get("/performance")

async def get_performance_metrics(
    request: Request,
    period: str = Query("30d", description="分析周期: 7d, 30d, 90d, 1y"),
    benchmark: Optional[str] = Query(None, description="基准对比: BTCUSDT, SPY等"),
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取组合表现指标
    集成现有的PerformanceAnalyzer
    """
    try:
        # 解析周期
        period_days = {
            '7d': 7,
            '30d': 30,
            '90d': 90,
            '1y': 365
        }.get(period, 30)

        if trading_system is None:
            # 返回模拟性能数据
            performance = {
                "period": period,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "sortino_ratio": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "average_trade": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "status": "not_trading"
            }
        else:
            # 使用现有的PerformanceAnalyzer
            analyzer = PerformanceAnalyzer()

            # 获取组合历史数据
            portfolio_history = trading_system.get_portfolio_history(days=period_days)

            if portfolio_history and len(portfolio_history) > 1:
                # 计算性能指标
                performance = analyzer.calculate_performance_metrics(
                    portfolio_values=portfolio_history['values'],
                    dates=portfolio_history['dates'],
                    initial_capital=trading_system.config.initial_capital
                )

                # 获取交易统计
                trade_stats = analyzer.calculate_trade_statistics(
                    trading_system.get_trade_history(days=period_days)
                )
                performance.update(trade_stats)

                # 如果有基准，计算相对表现
                if benchmark:
                    try:
                        benchmark_performance = analyzer.calculate_benchmark_performance(
                            benchmark, period_days
                        )
                        performance["benchmark"] = {
                            "symbol": benchmark,
                            "return": benchmark_performance.get('total_return', 0),
                            "volatility": benchmark_performance.get('volatility', 0),
                            "alpha": performance.get('total_return', 0) - benchmark_performance.get('total_return', 0),
                            "beta": benchmark_performance.get('beta', 1.0)
                        }
                    except:
                        logger.warning(f"无法获取基准 {benchmark} 的数据")

            else:
                performance = {
                    "period": period,
                    "total_return": 0.0,
                    "message": "历史数据不足，无法计算性能指标"
                }

            performance["status"] = "calculated"

        performance["period"] = period
        performance["calculation_date"] = datetime.now().isoformat() + "Z"

        return await create_response(
            data=performance,
            message="性能指标计算完成"
        )

    except Exception as e:
        logger.error(f"计算性能指标失败: {e}")
        return await create_error_response(
            message="性能指标计算失败",
            code="PERFORMANCE_ERROR",
            details=str(e)
        )

@router.get("/risk-assessment")

async def get_risk_assessment(
    request: Request,
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取组合风险评估
    集成现有的RiskManager
    """
    try:
        if trading_system is None or trading_system.risk_manager is None:
            # 返回默认风险评估
            risk_assessment = {
                "overall_risk_level": "low",
                "risk_score": 0,
                "risk_metrics": {
                    "portfolio_volatility": 0.0,
                    "var_95": 0.0,
                    "max_drawdown": 0.0,
                    "position_concentration": 0.0
                },
                "recommendations": ["系统未启动，无法评估风险"],
                "status": "not_available"
            }
        else:
            # 使用现有的RiskManager
            portfolio_summary = trading_system.get_portfolio_summary()

            # 计算组合风险
            risk_assessment = trading_system.risk_manager.assess_portfolio_risk(
                portfolio_value=portfolio_summary.get('total_value', 0),
                positions=portfolio_summary.get('positions', []),
                daily_return=portfolio_summary.get('daily_return_pct', 0) / 100
            )

            # 检查各种风险限制
            risk_limits = trading_system.risk_manager.check_all_risk_limits(
                portfolio_summary
            )
            risk_assessment["risk_limits"] = risk_limits

            # 生成风险建议
            recommendations = []

            if risk_assessment.get('overall_risk_level') == 'high':
                recommendations.append("组合风险较高，建议减少仓位或分散投资")

            if risk_limits.get('daily_loss_limit_exceeded'):
                recommendations.append("已触及日损失限制，建议暂停交易")

            if risk_limits.get('concentration_risk') == 'high':
                recommendations.append("持仓过于集中，建议分散投资")

            if not recommendations:
                recommendations.append("当前风险水平可控，继续监控")

            risk_assessment["recommendations"] = recommendations
            risk_assessment["status"] = "assessed"

        risk_assessment["assessment_time"] = datetime.now().isoformat() + "Z"

        return await create_response(
            data=risk_assessment,
            message="风险评估完成"
        )

    except Exception as e:
        logger.error(f"风险评估失败: {e}")
        return await create_error_response(
            message="风险评估失败",
            code="RISK_ASSESSMENT_ERROR",
            details=str(e)
        )

@router.get("/history")

async def get_portfolio_history(
    request: Request,
    days: int = Query(30, description="历史天数"),
    granularity: str = Query("1h", description="数据粒度: 5m, 15m, 1h, 1d"),
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取组合历史数据
    """
    try:
        if trading_system is None:
            # 返回模拟历史数据
            history = {
                "values": [10000.0],
                "dates": [datetime.now().isoformat() + "Z"],
                "returns": [0.0],
                "drawdowns": [0.0]
            }
        else:
            # 获取真实历史数据
            history = trading_system.get_portfolio_history(
                days=days,
                granularity=granularity
            )

        # 计算额外统计信息
        if len(history.get('values', [])) > 1:
            values = history['values']
            stats = {
                "start_value": values[0],
                "end_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "total_return": (values[-1] - values[0]) / values[0] * 100,
                "volatility": float(pd.Series(history.get('returns', [])).std() * (365**0.5) * 100) if history.get('returns') else 0,
                "max_drawdown": max(history.get('drawdowns', [0])) * 100
            }
        else:
            stats = {
                "start_value": 0,
                "end_value": 0,
                "total_return": 0,
                "volatility": 0,
                "max_drawdown": 0
            }

        result = {
            "history": history,
            "statistics": stats,
            "period": {
                "days": days,
                "granularity": granularity,
                "data_points": len(history.get('values', []))
            }
        }

        return await create_response(
            data=result,
            message="组合历史数据获取成功"
        )

    except Exception as e:
        logger.error(f"获取组合历史失败: {e}")
        return await create_error_response(
            message="获取组合历史失败",
            code="PORTFOLIO_HISTORY_ERROR",
            details=str(e)
        )
