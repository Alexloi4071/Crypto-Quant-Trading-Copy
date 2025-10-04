"""
Trading API Routes
交易执行API路由，集成现有的TradingSystem和SignalGenerator
提供完整的交易执行、信号管理和策略控制功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入依赖和配置
from api.dependencies import (
    get_trading_system, get_trading_system_optional, get_data_manager,
    get_current_user, create_response, create_error_response, require_permission
)

# 导入现有系统组件
from src.trading.trading_system import TradingSystem, TradingMode
from src.trading.signal_generator import SignalGenerator, SignalType, TradingSignal
from src.trading.position_manager import PositionSide
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter()

# Pydantic模型定义

class ManualTradeRequest(BaseModel):
    """手动交易请求"""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: Optional[float] = None
    order_type: str = "market"  # 'market' or 'limit'
    price: Optional[float] = None
    stop_loss_pct: Optional[float] = 0.02
    take_profit_pct: Optional[float] = 0.04
    notes: Optional[str] = None

class TradingControlRequest(BaseModel):
    """交易控制请求"""
    action: str  # 'start', 'stop', 'pause', 'resume'
    symbols: Optional[List[str]] = None
    reason: Optional[str] = None

class StrategyUpdateRequest(BaseModel):
    """策略更新请求"""
    strategy_name: str
    parameters: Dict[str, Any]
    symbols: Optional[List[str]] = None

@router.get("/status")

async def get_trading_status(
    request: Request,
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取交易系统状态
    集成现有的TradingSystem状态信息
    """
    try:
        if trading_system is None:
            status = {
                "system_status": "not_started",
                "trading_mode": "none",
                "active_symbols": [],
                "strategies_running": 0,
                "auto_trading": False,
                "last_signal_time": None,
                "last_trade_time": None,
                "uptime_seconds": 0
            }
        else:
            # 获取系统状态
            system_status = trading_system.get_system_status()

            status = {
                "system_status": trading_system.status.value,
                "trading_mode": trading_system.config.trading_mode.value,
                "active_symbols": trading_system.config.symbols,
                "strategies_running": len(getattr(trading_system, 'active_strategies', [])),
                "auto_trading": trading_system.running,
                "last_signal_time": system_status.get('last_signal_time'),
                "last_trade_time": system_status.get('last_trade_time'),
                "uptime_seconds": system_status.get('uptime_seconds', 0),

                # 交易统计
                "daily_trades": system_status.get('daily_trades', 0),
                "daily_pnl": system_status.get('daily_pnl', 0),
                "signals_generated": system_status.get('signals_generated', 0),
                "positions_open": len(system_status.get('open_positions', [])),

                # 配置信息
                "config": {
                    "initial_capital": trading_system.config.initial_capital,
                    "max_positions": trading_system.config.max_concurrent_positions,
                    "update_interval": trading_system.config.update_interval,
                    "risk_management": trading_system.config.enable_risk_management
                }
            }

        status["timestamp"] = datetime.now().isoformat() + "Z"

        return await create_response(
            data=status,
            message="交易状态获取成功"
        )

    except Exception as e:
        logger.error(f"获取交易状态失败: {e}")
        return await create_error_response(
            message="获取交易状态失败",
            code="TRADING_STATUS_ERROR",
            details=str(e)
        )

@router.post("/control")

async def control_trading(
    request: Request,
    control_request: TradingControlRequest,
    trading_system: TradingSystem = Depends(get_trading_system),
    current_user: dict = Depends(require_permission("trade"))
):
    """
    控制交易系统
    集成现有的TradingSystem控制功能
    """
    try:
        action = control_request.action.lower()

        if action == "start":
            if trading_system.running:
                return await create_error_response(
                    message="交易系统已经在运行中",
                    code="ALREADY_RUNNING"
                )

            # 启动交易系统
            await trading_system.start()
            message = "交易系统启动成功"

        elif action == "stop":
            if not trading_system.running:
                return await create_error_response(
                    message="交易系统未在运行",
                    code="NOT_RUNNING"
                )

            # 停止交易系统
            await trading_system.stop()
            message = "交易系统停止成功"

        elif action == "pause":
            if not trading_system.running:
                return await create_error_response(
                    message="交易系统未在运行",
                    code="NOT_RUNNING"
                )

            # 暂停交易
            await trading_system.pause_trading()
            message = "交易系统暂停成功"

        elif action == "resume":
            # 恢复交易
            await trading_system.resume_trading()
            message = "交易系统恢复成功"

        elif action == "emergency_stop":
            # 紧急停止
            reason = control_request.reason or "Manual emergency stop"
            await trading_system.emergency_stop(reason)
            message = "紧急停止执行成功"

        else:
            return await create_error_response(
                message=f"不支持的操作: {action}",
                code="INVALID_ACTION"
            )

        # 记录操作日志
        logger.info(f"交易控制操作: {action}, 用户: {current_user.get('user_id')}, 原因: {control_request.reason}")

        return await create_response(
            data={
                "action": action,
                "status": trading_system.status.value,
                "timestamp": datetime.now().isoformat() + "Z",
                "reason": control_request.reason
            },
            message=message
        )

    except Exception as e:
        logger.error(f"交易控制失败: {e}")
        return await create_error_response(
            message=f"交易控制操作失败",
            code="TRADING_CONTROL_ERROR",
            details=str(e)
        )

@router.post("/manual-trade")

async def execute_manual_trade(
    request: Request,
    trade_request: ManualTradeRequest,
    trading_system: TradingSystem = Depends(get_trading_system),
    current_user: dict = Depends(require_permission("trade"))
):
    """
    执行手动交易
    集成现有的PositionManager.open_position()
    """
    try:
        if trading_system.position_manager is None:
            raise HTTPException(
                status_code=503,
                detail="持仓管理器不可用"
            )

        # 验证交易参数
        if trade_request.symbol.upper() not in trading_system.config.symbols:
            return await create_error_response(
                message=f"交易对 {trade_request.symbol} 不在允许列表中",
                code="INVALID_SYMBOL"
            )

        if trade_request.side.lower() not in ['buy', 'sell']:
            return await create_error_response(
                message="交易方向必须是 'buy' 或 'sell'",
                code="INVALID_SIDE"
            )

        # 获取当前价格
        current_price = await trading_system.get_current_price(trade_request.symbol)
        if not current_price:
            return await create_error_response(
                message=f"无法获取 {trade_request.symbol} 的当前价格",
                code="PRICE_ERROR"
            )

        # 转换交易方向
        position_side = PositionSide.LONG if trade_request.side.lower() == 'buy' else PositionSide.SHORT

        # 计算交易数量
        if trade_request.size is None:
            # 使用默认仓位大小
            trade_size = trading_system.position_manager.calculate_default_position_size(
                trade_request.symbol,
                current_price
            )
        else:
            trade_size = trade_request.size

        # 执行交易
        result = await trading_system.position_manager.open_position(
            symbol=trade_request.symbol,
            side=position_side,
            size=trade_size,
            signal_confidence=0.8,  # 手动交易默认信心度
            current_price=current_price,
            stop_loss_pct=trade_request.stop_loss_pct,
            take_profit_pct=trade_request.take_profit_pct,
            notes=trade_request.notes or f"Manual trade by {current_user.get('user_id')}"
        )

        if result.get('success'):
            # 记录交易日志
            logger.info(f"手动交易执行成功: {trade_request.symbol} {trade_request.side} {trade_size}")

            return await create_response(
                data={
                    "trade_id": result.get('position_id'),
                    "symbol": trade_request.symbol,
                    "side": trade_request.side,
                    "size": trade_size,
                    "price": result.get('fill_price', current_price),
                    "order_type": trade_request.order_type,
                    "stop_loss": result.get('stop_loss'),
                    "take_profit": result.get('take_profit'),
                    "timestamp": datetime.now().isoformat() + "Z",
                    "status": "executed"
                },
                message="手动交易执行成功"
            )
        else:
            return await create_error_response(
                message="手动交易执行失败",
                code="TRADE_EXECUTION_ERROR",
                details=result.get('error', '未知错误')
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"手动交易失败: {e}")
        return await create_error_response(
            message="手动交易执行失败",
            code="MANUAL_TRADE_ERROR",
            details=str(e)
        )

@router.get("/signals/latest")

async def get_latest_signals(
    request: Request,
    symbols: Optional[str] = Query(None, description="筛选交易对，逗号分隔"),
    min_confidence: float = Query(0.5, description="最低信心度筛选"),
    limit: int = Query(10, description="返回数量限制"),
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取最新交易信号
    集成现有的SignalGenerator功能
    """
    try:
        if trading_system is None or trading_system.signal_generator is None:
            signals = []
        else:
            # 解析筛选符号
            filter_symbols = None
            if symbols:
                filter_symbols = [s.strip().upper() for s in symbols.split(',')]

            # 获取最新信号
            latest_signals = trading_system.signal_generator.get_latest_signals(
                symbols=filter_symbols,
                min_confidence=min_confidence,
                limit=limit
            )

            # 格式化信号数据
            signals = []
            for signal in latest_signals:
                signal_data = {
                    "id": getattr(signal, 'id', None),
                    "symbol": signal.symbol,
                    "timeframe": signal.timeframe,
                    "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                    "confidence": signal.confidence,
                    "price": signal.price,
                    "source": signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
                    "timestamp": signal.timestamp.isoformat() + "Z" if signal.timestamp else None,

                    # 额外信息
                    "strength": "strong" if signal.confidence > 0.8 else "medium" if signal.confidence > 0.6 else "weak",
                    "metadata": getattr(signal, 'metadata', {}),
                    "valid_until": getattr(signal, 'valid_until', None)
                }
                signals.append(signal_data)

        # 添加信号统计
        signal_stats = {
            "total_signals": len(signals),
            "strong_signals": len([s for s in signals if s.get('strength') == 'strong']),
            "buy_signals": len([s for s in signals if 'buy' in s.get('signal_type', '').lower()]),
            "sell_signals": len([s for s in signals if 'sell' in s.get('signal_type', '').lower()]),
            "average_confidence": sum(s.get('confidence', 0) for s in signals) / len(signals) if signals else 0
        }

        return await create_response(
            data={
                "signals": signals,
                "statistics": signal_stats,
                "filters": {
                    "symbols": symbols,
                    "min_confidence": min_confidence,
                    "limit": limit
                }
            },
            message="最新信号获取成功"
        )

    except Exception as e:
        logger.error(f"获取最新信号失败: {e}")
        return await create_error_response(
            message="获取最新信号失败",
            code="SIGNALS_ERROR",
            details=str(e)
        )

@router.post("/signals/manual")

async def create_manual_signal(
    request: Request,
    signal_data: Dict[str, Any] = Body(...),
    trading_system: TradingSystem = Depends(get_trading_system),
    current_user: dict = Depends(require_permission("trade"))
):
    """
    创建手动交易信号
    """
    try:
        # 验证信号数据
        required_fields = ['symbol', 'signal_type', 'confidence']
        for field in required_fields:
            if field not in signal_data:
                return await create_error_response(
                    message=f"缺少必需字段: {field}",
                    code="MISSING_FIELD"
                )

        # 获取当前价格
        current_price = await trading_system.get_current_price(signal_data['symbol'])

        # 创建手动信号
        from src.trading.signal_generator import TradingSignal, SignalSource, SignalType

        manual_signal = TradingSignal(
            symbol=signal_data['symbol'].upper(),
            timeframe=signal_data.get('timeframe', '1h'),
            signal_type=SignalType[signal_data['signal_type'].upper()],
            confidence=float(signal_data['confidence']),
            price=current_price,
            source=SignalSource.MANUAL,
            timestamp=datetime.now(),
            metadata={
                'created_by': current_user.get('user_id'),
                'reason': signal_data.get('reason', 'Manual signal'),
                'notes': signal_data.get('notes', ''),
                'manual': True
            }
        )

        # 发送信号到交易系统
        if trading_system.signal_generator:
            signal_sent = await trading_system.signal_generator.process_manual_signal(manual_signal)

            if signal_sent:
                return await create_response(
                    data={
                        "signal_id": getattr(manual_signal, 'id', None),
                        "symbol": manual_signal.symbol,
                        "signal_type": manual_signal.signal_type.value,
                        "confidence": manual_signal.confidence,
                        "price": manual_signal.price,
                        "timestamp": manual_signal.timestamp.isoformat() + "Z",
                        "status": "created"
                    },
                    message="手动信号创建成功"
                )
            else:
                return await create_error_response(
                    message="信号创建失败",
                    code="SIGNAL_CREATION_ERROR"
                )
        else:
            return await create_error_response(
                message="信号生成器不可用",
                code="SIGNAL_GENERATOR_ERROR"
            )

    except Exception as e:
        logger.error(f"创建手动信号失败: {e}")
        return await create_error_response(
            message="创建手动信号失败",
            code="MANUAL_SIGNAL_ERROR",
            details=str(e)
        )

@router.get("/strategies")

async def get_active_strategies(
    request: Request,
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    current_user: dict = Depends(get_current_user)
):
    """
    获取活跃策略信息
    """
    try:
        if trading_system is None:
            strategies = []
        else:
            # 获取活跃策略列表
            active_strategies = getattr(trading_system, 'active_strategies', [])

            strategies = []
            for strategy in active_strategies:
                strategy_info = {
                    "name": getattr(strategy, 'name', 'Unknown'),
                    "type": getattr(strategy, 'strategy_type', 'Unknown'),
                    "symbols": getattr(strategy, 'symbols', []),
                    "status": getattr(strategy, 'status', 'active'),
                    "performance": getattr(strategy, 'performance_summary', {}),
                    "parameters": getattr(strategy, 'parameters', {}),
                    "last_signal": getattr(strategy, 'last_signal_time', None),
                    "trades_count": getattr(strategy, 'trades_count', 0)
                }
                strategies.append(strategy_info)

        return await create_response(
            data={
                "strategies": strategies,
                "total_strategies": len(strategies),
                "active_count": len([s for s in strategies if s.get('status') == 'active'])
            },
            message="策略信息获取成功"
        )

    except Exception as e:
        logger.error(f"获取策略信息失败: {e}")
        return await create_error_response(
            message="获取策略信息失败",
            code="STRATEGIES_ERROR",
            details=str(e)
        )

@router.put("/strategies/{strategy_name}")

async def update_strategy(
    strategy_name: str,
    request: Request,
    update_request: StrategyUpdateRequest,
    trading_system: TradingSystem = Depends(get_trading_system),
    current_user: dict = Depends(require_permission("admin"))
):
    """
    更新策略参数
    """
    try:
        # 查找策略
        strategy = trading_system.get_strategy(strategy_name)
        if not strategy:
            return await create_error_response(
                message=f"策略 {strategy_name} 不存在",
                code="STRATEGY_NOT_FOUND"
            )

        # 更新策略参数
        updated = await trading_system.update_strategy_parameters(
            strategy_name,
            update_request.parameters,
            update_request.symbols
        )

        if updated:
            logger.info(f"策略 {strategy_name} 参数已更新，用户: {current_user.get('user_id')}")

            return await create_response(
                data={
                    "strategy_name": strategy_name,
                    "updated_parameters": update_request.parameters,
                    "symbols": update_request.symbols,
                    "timestamp": datetime.now().isoformat() + "Z"
                },
                message="策略参数更新成功"
            )
        else:
            return await create_error_response(
                message="策略参数更新失败",
                code="STRATEGY_UPDATE_ERROR"
            )

    except Exception as e:
        logger.error(f"更新策略失败: {e}")
        return await create_error_response(
            message="策略参数更新失败",
            code="STRATEGY_UPDATE_ERROR",
            details=str(e)
        )

@router.get("/trades/history")

async def get_trade_history(
    request: Request,
    days: int = Query(30, description="历史天数"),
    symbol: Optional[str] = Query(None, description="筛选交易对"),
    status: Optional[str] = Query(None, description="筛选状态"),
    limit: int = Query(100, description="返回数量限制"),
    trading_system: TradingSystem = Depends(get_trading_system_optional),
    pagination = Depends(get_pagination),
    current_user: dict = Depends(get_current_user)
):
    """
    获取交易历史
    """
    try:
        if trading_system is None:
            trades = []
        else:
            # 获取交易历史
            trade_history = trading_system.get_trade_history(
                days=days,
                symbol=symbol,
                status=status,
                limit=limit
            )

            trades = []
            for trade in trade_history:
                trade_data = {
                    "id": trade.get('id'),
                    "symbol": trade.get('symbol'),
                    "side": trade.get('side'),
                    "size": trade.get('size'),
                    "entry_price": trade.get('entry_price'),
                    "exit_price": trade.get('exit_price'),
                    "pnl": trade.get('pnl'),
                    "pnl_pct": trade.get('pnl_pct'),
                    "duration": trade.get('duration'),
                    "strategy": trade.get('strategy'),
                    "signal_confidence": trade.get('signal_confidence'),
                    "opened_at": trade.get('opened_at'),
                    "closed_at": trade.get('closed_at'),
                    "status": trade.get('status'),
                    "notes": trade.get('notes')
                }
                trades.append(trade_data)

        # 应用分页
        total_count = len(trades)
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_trades = trades[start_idx:end_idx]

        # 计算统计信息
        if trades:
            stats = {
                "total_trades": total_count,
                "winning_trades": len([t for t in trades if t.get('pnl', 0) > 0]),
                "losing_trades": len([t for t in trades if t.get('pnl', 0) < 0]),
                "total_pnl": sum(t.get('pnl', 0) for t in trades),
                "average_pnl": sum(t.get('pnl', 0) for t in trades) / len(trades),
                "win_rate": len([t for t in trades if t.get('pnl', 0) > 0]) / len(trades) * 100,
                "best_trade": max(trades, key=lambda x: x.get('pnl', 0)).get('pnl', 0) if trades else 0,
                "worst_trade": min(trades, key=lambda x: x.get('pnl', 0)).get('pnl', 0) if trades else 0
            }
        else:
            stats = {}

        return await create_response(
            data={
                "trades": paginated_trades,
                "statistics": stats,
                "pagination": {
                    "page": pagination.page,
                    "size": pagination.size,
                    "total": total_count,
                    "pages": (total_count + pagination.size - 1) // pagination.size
                },
                "filters": {
                    "days": days,
                    "symbol": symbol,
                    "status": status
                }
            },
            message="交易历史获取成功"
        )

    except Exception as e:
        logger.error(f"获取交易历史失败: {e}")
        return await create_error_response(
            message="获取交易历史失败",
            code="TRADE_HISTORY_ERROR",
            details=str(e)
        )
