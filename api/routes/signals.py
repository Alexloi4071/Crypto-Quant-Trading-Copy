"""
Signals API Routes
交易信号API路由，集成现有的SignalGenerator和TechnicalAnalysis
提供信号查询、分析和历史记录功能
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
from src.trading.signal_generator import SignalGenerator, SignalType, SignalSource
from src.analysis.wyckoff_analyzer import WyckoffAnalyzer
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter()

@router.get("/current")
async def get_current_signals(
    request: Request,
    symbols: Optional[str] = Query(None, description="筛选交易对，逗号分隔"),
    timeframes: Optional[str] = Query(None, description="筛选时间框架，逗号分隔"),
    min_confidence: float = Query(0.5, description="最低信心度筛选"),
    sources: Optional[str] = Query(None, description="信号源筛选"),
    trading_system = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取当前有效交易信号
    集成现有的SignalGenerator功能
    """
    try:
        signals = []

        if trading_system and trading_system.signal_generator:
            # 解析筛选条件
            symbol_list = [s.strip().upper() for s in symbols.split(',')] if symbols else None
            timeframe_list = [t.strip() for t in timeframes.split(',')] if timeframes else None
            source_list = [s.strip() for s in sources.split(',')] if sources else None

            # 获取当前信号
            current_signals = await trading_system.signal_generator.get_current_signals(
                symbols=symbol_list,
                timeframes=timeframe_list,
                min_confidence=min_confidence,
                sources=source_list
            )

            # 格式化信号数据
            for signal in current_signals:
                signal_data = {
                    "id": getattr(signal, 'id', None),
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                    "confidence": round(signal.confidence, 4),
                    "price": signal.price,
                    "source": signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
                    "timestamp": signal.timestamp.isoformat() + "Z" if signal.timestamp else None,

                    # 信号强度分类
                    "strength": (
                        "very_strong" if signal.confidence >= 0.9 else
                        "strong" if signal.confidence >= 0.8 else
                        "medium" if signal.confidence >= 0.6 else
                        "weak"
                    ),

                    # 信号方向
                    "direction": (
                        "bullish" if "buy" in str(signal.signal_type).lower() else
                        "bearish" if "sell" in str(signal.signal_type).lower() else
                        "neutral"
                    ),

                    # 元数据
                    "metadata": getattr(signal, 'metadata', {}),
                    "valid_until": getattr(signal, 'valid_until', None),
                    "generated_by": getattr(signal, 'generated_by', 'system')
                }
                signals.append(signal_data)

        # 信号统计
        stats = {
            "total_signals": len(signals),
            "by_strength": {
                "very_strong": len([s for s in signals if s.get('strength') == 'very_strong']),
                "strong": len([s for s in signals if s.get('strength') == 'strong']),
                "medium": len([s for s in signals if s.get('strength') == 'medium']),
                "weak": len([s for s in signals if s.get('strength') == 'weak'])
            },
            "by_direction": {
                "bullish": len([s for s in signals if s.get('direction') == 'bullish']),
                "bearish": len([s for s in signals if s.get('direction') == 'bearish']),
                "neutral": len([s for s in signals if s.get('direction') == 'neutral'])
            },
            "by_source": {},
            "average_confidence": round(sum(s.get('confidence', 0) for s in signals) / len(signals), 4) if signals else 0
        }

        # 按信号源统计
        for signal in signals:
            source = signal.get('source', 'unknown')
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

        return await create_response(
            data={
                "signals": signals,
                "statistics": stats,
                "filters": {
                    "symbols": symbol_list,
                    "timeframes": timeframe_list,
                    "min_confidence": min_confidence,
                    "sources": source_list
                },
                "timestamp": datetime.now().isoformat() + "Z"
            },
            message="当前信号获取成功"
        )

    except Exception as e:
        logger.error(f"获取当前信号失败: {e}")
        return await create_error_response(
            message="获取当前信号失败",
            code="CURRENT_SIGNALS_ERROR",
            details=str(e)
        )

@router.get("/history")
async def get_signal_history(
    request: Request,
    symbol: Optional[str] = Query(None, description="筛选交易对"),
    timeframe: Optional[str] = Query(None, description="筛选时间框架"),
    days: int = Query(7, description="历史天数"),
    min_confidence: float = Query(0.0, description="最低信心度"),
    signal_type: Optional[str] = Query(None, description="信号类型筛选"),
    trading_system = Depends(get_trading_system_optional),
    pagination = Depends(get_pagination),
    current_user: dict = Depends(get_current_user)
):
    """
    获取历史交易信号
    """
    try:
        signals = []

        if trading_system and trading_system.signal_generator:
            # 获取历史信号
            start_date = datetime.now() - timedelta(days=days)

            signal_history = await trading_system.signal_generator.get_signal_history(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                min_confidence=min_confidence,
                signal_type=signal_type
            )

            # 格式化历史信号数据
            for signal_record in signal_history:
                signal_data = {
                    "id": signal_record.get('id'),
                    "symbol": signal_record.get('symbol'),
                    "timeframe": signal_record.get('timeframe'),
                    "signal_type": signal_record.get('signal_type'),
                    "confidence": signal_record.get('confidence'),
                    "price": signal_record.get('price'),
                    "source": signal_record.get('source'),
                    "timestamp": signal_record.get('timestamp'),

                    # 信号结果分析
                    "outcome": signal_record.get('outcome'),  # 'profitable', 'loss', 'pending'
                    "pnl": signal_record.get('pnl', 0),
                    "accuracy": signal_record.get('accuracy'),
                    "execution_status": signal_record.get('execution_status'),  # 'executed', 'ignored', 'failed'

                    # 技术指标快照
                    "technical_snapshot": signal_record.get('technical_snapshot', {}),
                    "market_conditions": signal_record.get('market_conditions', {}),

                    "notes": signal_record.get('notes', '')
                }
                signals.append(signal_data)

        # 应用分页
        total_count = len(signals)
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_signals = signals[start_idx:end_idx]

        # 计算历史统计
        if signals:
            executed_signals = [s for s in signals if s.get('execution_status') == 'executed']
            profitable_signals = [s for s in executed_signals if s.get('pnl', 0) > 0]

            stats = {
                "total_signals": total_count,
                "executed_signals": len(executed_signals),
                "profitable_signals": len(profitable_signals),
                "success_rate": len(profitable_signals) / len(executed_signals) * 100 if executed_signals else 0,
                "total_pnl": sum(s.get('pnl', 0) for s in executed_signals),
                "average_pnl": sum(s.get('pnl', 0) for s in executed_signals) / len(executed_signals) if executed_signals else 0,
                "best_signal_pnl": max((s.get('pnl', 0) for s in executed_signals), default=0),
                "worst_signal_pnl": min((s.get('pnl', 0) for s in executed_signals), default=0)
            }
        else:
            stats = {}

        return await create_response(
            data={
                "signals": paginated_signals,
                "statistics": stats,
                "pagination": {
                    "page": pagination.page,
                    "size": pagination.size,
                    "total": total_count,
                    "pages": (total_count + pagination.size - 1) // pagination.size
                },
                "filters": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "days": days,
                    "min_confidence": min_confidence,
                    "signal_type": signal_type
                }
            },
            message="历史信号获取成功"
        )

    except Exception as e:
        logger.error(f"获取历史信号失败: {e}")
        return await create_error_response(
            message="获取历史信号失败",
            code="SIGNAL_HISTORY_ERROR",
            details=str(e)
        )

@router.get("/analysis/{symbol}")
async def get_signal_analysis(
    symbol: str,
    request: Request,
    timeframe: str = Query("1h", description="分析时间框架"),
    include_wyckoff: bool = Query(True, description="包含Wyckoff分析"),
    include_technical: bool = Query(True, description="包含技术分析"),
    trading_system = Depends(get_trading_system_optional),
    data_manager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    获取特定交易对的信号分析
    集成现有的WyckoffAnalyzer和技术分析组件
    """
    try:
        symbol = symbol.upper()
        analysis_result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now().isoformat() + "Z"
        }

        # 获取市场数据
        try:
            market_data = await data_manager.fetch_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=200  # 获取足够的历史数据进行分析
            )

            if market_data is None or len(market_data) == 0:
                return await create_error_response(
                    message=f"无法获取 {symbol} 的市场数据",
                    code="NO_MARKET_DATA"
                )

            current_price = float(market_data['close'].iloc[-1])
            analysis_result["current_price"] = current_price

        except Exception as e:
            logger.error(f"获取市场数据失败: {e}")
            return await create_error_response(
                message="获取市场数据失败",
                code="MARKET_DATA_ERROR",
                details=str(e)
            )

        # Wyckoff分析
        if include_wyckoff:
            try:
                wyckoff_analyzer = WyckoffAnalyzer()
                wyckoff_analysis = await wyckoff_analyzer.analyze_market_structure(
                    symbol=symbol,
                    timeframe=timeframe,
                    ohlcv_data=market_data
                )

                analysis_result["wyckoff_analysis"] = {
                    "current_phase": wyckoff_analysis.get('current_phase'),
                    "phase_confidence": wyckoff_analysis.get('phase_confidence'),
                    "supply_demand_balance": wyckoff_analysis.get('supply_demand_balance'),
                    "volume_analysis": wyckoff_analysis.get('volume_analysis'),
                    "key_levels": wyckoff_analysis.get('key_levels', {}),
                    "recommendations": wyckoff_analysis.get('recommendations', [])
                }

            except Exception as e:
                logger.warning(f"Wyckoff分析失败: {e}")
                analysis_result["wyckoff_analysis"] = {"error": "Wyckoff分析暂不可用"}

        # 技术分析
        if include_technical:
            try:
                if trading_system and trading_system.signal_generator:
                    technical_analysis = await trading_system.signal_generator.get_technical_analysis(
                        symbol=symbol,
                        timeframe=timeframe,
                        market_data=market_data
                    )

                    analysis_result["technical_analysis"] = {
                        "indicators": technical_analysis.get('indicators', {}),
                        "trend_analysis": technical_analysis.get('trend_analysis', {}),
                        "momentum_indicators": technical_analysis.get('momentum_indicators', {}),
                        "volatility_indicators": technical_analysis.get('volatility_indicators', {}),
                        "volume_indicators": technical_analysis.get('volume_indicators', {}),
                        "support_resistance": technical_analysis.get('support_resistance', {}),
                        "overall_sentiment": technical_analysis.get('overall_sentiment')
                    }
                else:
                    analysis_result["technical_analysis"] = {"error": "技术分析服务不可用"}

            except Exception as e:
                logger.warning(f"技术分析失败: {e}")
                analysis_result["technical_analysis"] = {"error": "技术分析暂不可用"}

        # 获取最新信号
        if trading_system and trading_system.signal_generator:
            try:
                latest_signals = trading_system.signal_generator.get_latest_signals(
                    symbols=[symbol],
                    limit=5
                )

                recent_signals = []
                for signal in latest_signals:
                    signal_data = {
                        "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                        "confidence": signal.confidence,
                        "source": signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
                        "timestamp": signal.timestamp.isoformat() + "Z" if signal.timestamp else None,
                        "price": signal.price
                    }
                    recent_signals.append(signal_data)

                analysis_result["recent_signals"] = recent_signals

            except Exception as e:
                logger.warning(f"获取最新信号失败: {e}")
                analysis_result["recent_signals"] = []

        # 综合评估
        overall_score = 0
        sentiment_factors = []

        # Wyckoff评估
        if analysis_result.get("wyckoff_analysis") and not analysis_result["wyckoff_analysis"].get("error"):
            wyckoff_phase = analysis_result["wyckoff_analysis"].get("current_phase", "")
            if "accumulation" in wyckoff_phase.lower() or "markup" in wyckoff_phase.lower():
                overall_score += 30
                sentiment_factors.append("Wyckoff: Bullish phase")
            elif "distribution" in wyckoff_phase.lower() or "markdown" in wyckoff_phase.lower():
                overall_score -= 30
                sentiment_factors.append("Wyckoff: Bearish phase")

        # 技术分析评估
        if analysis_result.get("technical_analysis") and not analysis_result["technical_analysis"].get("error"):
            technical_sentiment = analysis_result["technical_analysis"].get("overall_sentiment")
            if technical_sentiment == "bullish":
                overall_score += 25
                sentiment_factors.append("Technical: Bullish")
            elif technical_sentiment == "bearish":
                overall_score -= 25
                sentiment_factors.append("Technical: Bearish")

        # 最新信号评估
        if analysis_result.get("recent_signals"):
            strong_buy_signals = len([s for s in analysis_result["recent_signals"]
                                   if "buy" in s.get("signal_type", "").lower() and s.get("confidence", 0) > 0.7])
            strong_sell_signals = len([s for s in analysis_result["recent_signals"]
                                    if "sell" in s.get("signal_type", "").lower() and s.get("confidence", 0) > 0.7])

            if strong_buy_signals > strong_sell_signals:
                overall_score += 20
                sentiment_factors.append(f"Recent signals: {strong_buy_signals} strong buy signals")
            elif strong_sell_signals > strong_buy_signals:
                overall_score -= 20
                sentiment_factors.append(f"Recent signals: {strong_sell_signals} strong sell signals")

        # 综合评分
        analysis_result["overall_assessment"] = {
            "score": max(-100, min(100, overall_score)),  # 限制在-100到100之间
            "sentiment": (
                "very_bullish" if overall_score > 60 else
                "bullish" if overall_score > 20 else
                "neutral" if overall_score > -20 else
                "bearish" if overall_score > -60 else
                "very_bearish"
            ),
            "confidence": min(abs(overall_score) / 100, 1.0),
            "factors": sentiment_factors,
            "recommendation": (
                "强烈看多" if overall_score > 60 else
                "看多" if overall_score > 20 else
                "中性观望" if overall_score > -20 else
                "看空" if overall_score > -60 else
                "强烈看空"
            )
        }

        return await create_response(
            data=analysis_result,
            message="信号分析完成"
        )

    except Exception as e:
        logger.error(f"信号分析失败: {e}")
        return await create_error_response(
            message="信号分析失败",
            code="SIGNAL_ANALYSIS_ERROR",
            details=str(e)
        )

@router.get("/performance")
async def get_signal_performance(
    request: Request,
    symbol: Optional[str] = Query(None, description="筛选交易对"),
    days: int = Query(30, description="分析周期天数"),
    source: Optional[str] = Query(None, description="信号源筛选"),
    trading_system = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号表现分析
    """
    try:
        if not trading_system or not trading_system.signal_generator:
            return await create_error_response(
                message="信号生成器不可用",
                code="SIGNAL_GENERATOR_UNAVAILABLE"
            )

        # 获取信号表现数据
        start_date = datetime.now() - timedelta(days=days)
        performance_data = await trading_system.signal_generator.analyze_signal_performance(
            symbol=symbol,
            start_date=start_date,
            source=source
        )

        # 格式化表现数据
        performance_summary = {
            "period": {
                "days": days,
                "start_date": start_date.isoformat() + "Z",
                "end_date": datetime.now().isoformat() + "Z"
            },

            "overall_performance": {
                "total_signals": performance_data.get('total_signals', 0),
                "executed_signals": performance_data.get('executed_signals', 0),
                "success_rate": performance_data.get('success_rate', 0),
                "average_return": performance_data.get('average_return', 0),
                "total_return": performance_data.get('total_return', 0),
                "sharpe_ratio": performance_data.get('sharpe_ratio', 0),
                "win_rate": performance_data.get('win_rate', 0)
            },

            "by_signal_type": performance_data.get('by_signal_type', {}),
            "by_source": performance_data.get('by_source', {}),
            "by_confidence_range": performance_data.get('by_confidence_range', {}),

            "best_performing": {
                "signals": performance_data.get('best_signals', []),
                "avg_return": performance_data.get('best_avg_return', 0)
            },

            "worst_performing": {
                "signals": performance_data.get('worst_signals', []),
                "avg_return": performance_data.get('worst_avg_return', 0)
            },

            "recommendations": performance_data.get('recommendations', [])
        }

        return await create_response(
            data=performance_summary,
            message="信号表现分析完成"
        )

    except Exception as e:
        logger.error(f"信号表现分析失败: {e}")
        return await create_error_response(
            message="信号表现分析失败",
            code="SIGNAL_PERFORMANCE_ERROR",
            details=str(e)
        )

@router.get("/alerts")
async def get_signal_alerts(
    request: Request,
    active_only: bool = Query(True, description="只显示活跃告警"),
    trading_system = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取信号相关告警
    """
    try:
        alerts = []

        if trading_system and trading_system.alerting_system:
            # 获取信号相关的告警
            signal_alerts = trading_system.alerting_system.get_alerts(
                category="signals",
                active_only=active_only
            )

            for alert in signal_alerts:
                alert_data = {
                    "id": alert.get('id'),
                    "type": alert.get('type'),
                    "severity": alert.get('severity'),
                    "message": alert.get('message'),
                    "symbol": alert.get('symbol'),
                    "trigger_condition": alert.get('trigger_condition'),
                    "current_value": alert.get('current_value'),
                    "threshold": alert.get('threshold'),
                    "created_at": alert.get('created_at'),
                    "updated_at": alert.get('updated_at'),
                    "status": alert.get('status'),
                    "acknowledged": alert.get('acknowledged', False)
                }
                alerts.append(alert_data)

        # 告警统计
        stats = {
            "total_alerts": len(alerts),
            "by_severity": {},
            "by_status": {},
            "unacknowledged": len([a for a in alerts if not a.get('acknowledged')])
        }

        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            status = alert.get('status', 'unknown')

            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        return await create_response(
            data={
                "alerts": alerts,
                "statistics": stats,
                "filters": {
                    "active_only": active_only
                }
            },
            message="信号告警获取成功"
        )

    except Exception as e:
        logger.error(f"获取信号告警失败: {e}")
        return await create_error_response(
            message="获取信号告警失败",
            code="SIGNAL_ALERTS_ERROR",
            details=str(e)
        )
