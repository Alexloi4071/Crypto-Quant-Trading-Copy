"""
Data API Routes
数据查询API路由，集成现有的DataManager和数据处理功能
提供市场数据、特征数据和历史数据查询功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional, List
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入依赖和配置
from api.dependencies import (
    get_data_manager, get_trading_system_optional, get_pagination,
    get_query_params, get_current_user, create_response, create_error_response
)

# 导入现有系统组件
from src.data.data_manager import DataManager
from src.features.feature_engineering import FeatureEngineer
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter()

@router.get("/symbols")

async def get_available_symbols(
    request: Request,
    exchange: Optional[str] = Query(None, description="筛选交易所"),
    active_only: bool = Query(True, description="只显示活跃交易对"),
    data_manager: DataManager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    获取可用交易对列表
    集成现有的DataManager.get_available_symbols()
    """
    try:
        # 获取可用交易对
        available_symbols = await data_manager.get_available_symbols(
            exchange=exchange,
            active_only=active_only
        )

        # 格式化交易对信息
        symbols_info = []
        for symbol_data in available_symbols:
            if isinstance(symbol_data, str):
                # 如果只返回symbol字符串
                symbol_info = {
                    "symbol": symbol_data,
                    "exchange": exchange or "binance",
                    "status": "active" if active_only else "unknown",
                    "last_update": None
                }
            else:
                # 如果返回详细信息
                symbol_info = {
                    "symbol": symbol_data.get('symbol'),
                    "exchange": symbol_data.get('exchange', exchange or "binance"),
                    "base_asset": symbol_data.get('base_asset'),
                    "quote_asset": symbol_data.get('quote_asset'),
                    "status": symbol_data.get('status', 'active'),
                    "price_precision": symbol_data.get('price_precision'),
                    "quantity_precision": symbol_data.get('quantity_precision'),
                    "min_quantity": symbol_data.get('min_quantity'),
                    "last_update": symbol_data.get('last_update')
                }
            symbols_info.append(symbol_info)

        # 统计信息
        stats = {
            "total_symbols": len(symbols_info),
            "by_exchange": {},
            "by_status": {}
        }

        for symbol in symbols_info:
            exchange_name = symbol.get('exchange', 'unknown')
            status = symbol.get('status', 'unknown')

            stats["by_exchange"][exchange_name] = stats["by_exchange"].get(exchange_name, 0) + 1
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        return await create_response(
            data={
                "symbols": symbols_info,
                "statistics": stats,
                "filters": {
                    "exchange": exchange,
                    "active_only": active_only
                }
            },
            message="可用交易对获取成功"
        )

    except Exception as e:
        logger.error(f"获取可用交易对失败: {e}")
        return await create_error_response(
            message="获取可用交易对失败",
            code="SYMBOLS_ERROR",
            details=str(e)
        )

@router.get("/ohlcv/{symbol}")

async def get_ohlcv_data(
    symbol: str,
    request: Request,
    timeframe: str = Query("1h", description="时间框架"),
    limit: int = Query(100, description="数据条数限制"),
    start_date: Optional[str] = Query(None, description="开始日期 YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="结束日期 YYYY-MM-DD"),
    data_manager: DataManager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    获取OHLCV市场数据
    集成现有的DataManager.fetch_ohlcv_data()
    """
    try:
        symbol = symbol.upper()

        # 解析日期参数
        start_datetime = None
        end_datetime = None

        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                return await create_error_response(
                    message="开始日期格式错误，请使用 YYYY-MM-DD 格式",
                    code="INVALID_START_DATE"
                )

        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                return await create_error_response(
                    message="结束日期格式错误，请使用 YYYY-MM-DD 格式",
                    code="INVALID_END_DATE"
                )

        # 获取OHLCV数据
        ohlcv_data = await data_manager.fetch_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=min(limit, 1000),  # 限制最大数据量
            start_date=start_datetime,
            end_date=end_datetime
        )

        if ohlcv_data is None or len(ohlcv_data) == 0:
            return await create_error_response(
                message=f"未找到 {symbol} 的数据",
                code="NO_DATA_FOUND"
            )

        # 格式化数据
        formatted_data = []
        for index, row in ohlcv_data.iterrows():
            data_point = {
                "timestamp": row.get('timestamp', index).isoformat() + "Z" if pd.notnull(row.get('timestamp', index)) else None,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            }
            formatted_data.append(data_point)

        # 计算统计信息
        stats = {
            "count": len(formatted_data),
            "timeframe": timeframe,
            "period": {
                "start": formatted_data[0]["timestamp"] if formatted_data else None,
                "end": formatted_data[-1]["timestamp"] if formatted_data else None
            },
            "price_summary": {
                "current_price": float(ohlcv_data['close'].iloc[-1]) if len(ohlcv_data) > 0 else None,
                "high_24h": float(ohlcv_data['high'].tail(24).max()) if len(ohlcv_data) >= 24 else float(ohlcv_data['high'].max()),
                "low_24h": float(ohlcv_data['low'].tail(24).min()) if len(ohlcv_data) >= 24 else float(ohlcv_data['low'].min()),
                "volume_24h": float(ohlcv_data['volume'].tail(24).sum()) if len(ohlcv_data) >= 24 else float(ohlcv_data['volume'].sum()),
                "price_change_24h": float(ohlcv_data['close'].iloc[-1] - ohlcv_data['close'].iloc[-25]) if len(ohlcv_data) >= 25 else 0,
                "price_change_pct_24h": float((ohlcv_data['close'].iloc[-1] / ohlcv_data['close'].iloc[-25] - 1) * 100) if len(ohlcv_data) >= 25 else 0
            }
        }

        return await create_response(
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "data": formatted_data,
                "statistics": stats
            },
            message="OHLCV数据获取成功"
        )

    except Exception as e:
        logger.error(f"获取OHLCV数据失败: {e}")
        return await create_error_response(
            message="获取OHLCV数据失败",
            code="OHLCV_ERROR",
            details=str(e)
        )

@router.get("/features/{symbol}")

async def get_feature_data(
    symbol: str,
    request: Request,
    timeframe: str = Query("1h", description="时间框架"),
    version: Optional[str] = Query(None, description="特征版本"),
    feature_names: Optional[str] = Query(None, description="特征名称筛选，逗号分隔"),
    limit: int = Query(100, description="数据条数限制"),
    data_manager: DataManager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    获取特征工程数据
    集成现有的FeatureEngineer功能
    """
    try:
        symbol = symbol.upper()

        # 首先获取原始OHLCV数据
        ohlcv_data = await data_manager.load_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            version=version
        )

        if ohlcv_data is None or len(ohlcv_data) == 0:
            return await create_error_response(
                message=f"未找到 {symbol} 的原始数据",
                code="NO_RAW_DATA"
            )

        # 生成特征
        feature_engineer = FeatureEngineer()
        features_df, metadata = feature_engineer.generate_features(ohlcv_data)

        if features_df is None or len(features_df) == 0:
            return await create_error_response(
                message="特征生成失败",
                code="FEATURE_GENERATION_ERROR"
            )

        # 筛选特定特征
        if feature_names:
            requested_features = [f.strip() for f in feature_names.split(',')]
            available_features = [f for f in requested_features if f in features_df.columns]

            if available_features:
                features_df = features_df[available_features + ['timestamp'] if 'timestamp' in features_df.columns else available_features]
            else:
                return await create_error_response(
                    message="请求的特征不存在",
                    code="FEATURES_NOT_FOUND",
                    details=f"可用特征: {list(features_df.columns)}"
                )

        # 限制数据量并格式化
        features_df = features_df.tail(min(limit, 1000))

        formatted_features = []
        for index, row in features_df.iterrows():
            feature_point = {
                "timestamp": row.get('timestamp', index).isoformat() + "Z" if pd.notnull(row.get('timestamp', index)) else None
            }

            # 添加特征值
            for col in features_df.columns:
                if col != 'timestamp':
                    value = row[col]
                    feature_point[col] = float(value) if pd.notnull(value) and not pd.isna(value) else None

            formatted_features.append(feature_point)

        # 特征统计信息
        feature_stats = {
            "total_features": len(features_df.columns) - (1 if 'timestamp' in features_df.columns else 0),
            "data_points": len(formatted_features),
            "feature_categories": metadata.get('feature_categories', {}),
            "generation_info": {
                "version": metadata.get('version'),
                "generation_time": metadata.get('generation_time'),
                "feature_selection_applied": metadata.get('feature_selection_applied', False)
            }
        }

        # 特征重要性（如果可用）
        if 'feature_importance' in metadata:
            feature_stats["importance_ranking"] = metadata['feature_importance']

        return await create_response(
            data={
                "symbol": symbol,
                "timeframe": timeframe,
                "version": version or metadata.get('version'),
                "features": formatted_features,
                "metadata": metadata,
                "statistics": feature_stats
            },
            message="特征数据获取成功"
        )

    except Exception as e:
        logger.error(f"获取特征数据失败: {e}")
        return await create_error_response(
            message="获取特征数据失败",
            code="FEATURES_ERROR",
            details=str(e)
        )

@router.get("/market-info/{symbol}")

async def get_market_info(
    symbol: str,
    request: Request,
    data_manager: DataManager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    获取交易对市场信息
    """
    try:
        symbol = symbol.upper()

        # 获取基本市场信息
        market_info = await data_manager.get_symbol_info(symbol)

        if not market_info:
            return await create_error_response(
                message=f"未找到 {symbol} 的市场信息",
                code="MARKET_INFO_NOT_FOUND"
            )

        # 获取实时价格数据
        try:
            current_ohlcv = await data_manager.fetch_ohlcv_data(
                symbol=symbol,
                timeframe="1d",
                limit=30  # 获取30天数据用于计算统计信息
            )

            if current_ohlcv is not None and len(current_ohlcv) > 0:
                latest_data = current_ohlcv.iloc[-1]

                # 计算价格统计
                price_stats = {
                    "current_price": float(latest_data['close']),
                    "open_24h": float(current_ohlcv.iloc[-2]['close']) if len(current_ohlcv) >= 2 else float(latest_data['open']),
                    "high_24h": float(current_ohlcv['high'].tail(2).max()),
                    "low_24h": float(current_ohlcv['low'].tail(2).min()),
                    "volume_24h": float(current_ohlcv['volume'].tail(2).sum()),
                    "price_change_24h": float(latest_data['close'] - current_ohlcv.iloc[-2]['close']) if len(current_ohlcv) >= 2 else 0,
                    "price_change_pct_24h": float((latest_data['close'] / current_ohlcv.iloc[-2]['close'] - 1) * 100) if len(current_ohlcv) >= 2 else 0,

                    # 30天统计
                    "high_30d": float(current_ohlcv['high'].max()),
                    "low_30d": float(current_ohlcv['low'].min()),
                    "avg_volume_30d": float(current_ohlcv['volume'].mean()),
                    "volatility_30d": float(current_ohlcv['close'].pct_change().std() * (30**0.5) * 100)
                }

                market_info["price_data"] = price_stats
            else:
                market_info["price_data"] = None

        except Exception as e:
            logger.warning(f"获取价格数据失败: {e}")
            market_info["price_data"] = None

        # 获取数据可用性信息
        try:
            data_availability = await data_manager.check_data_availability(symbol)
            market_info["data_availability"] = data_availability
        except Exception as e:
            logger.warning(f"检查数据可用性失败: {e}")
            market_info["data_availability"] = None

        # 添加时间戳
        market_info["last_updated"] = datetime.now().isoformat() + "Z"

        return await create_response(
            data=market_info,
            message="市场信息获取成功"
        )

    except Exception as e:
        logger.error(f"获取市场信息失败: {e}")
        return await create_error_response(
            message="获取市场信息失败",
            code="MARKET_INFO_ERROR",
            details=str(e)
        )

@router.get("/quality-check/{symbol}")

async def check_data_quality(
    symbol: str,
    request: Request,
    timeframe: str = Query("1h", description="时间框架"),
    days: int = Query(7, description="检查天数"),
    data_manager: DataManager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    检查数据质量
    集成现有的DataPreprocessor功能
    """
    try:
        symbol = symbol.upper()

        # 获取指定期间的数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        ohlcv_data = await data_manager.fetch_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if ohlcv_data is None or len(ohlcv_data) == 0:
            return await create_error_response(
                message=f"未找到 {symbol} 的数据进行质量检查",
                code="NO_DATA_FOR_QUALITY_CHECK"
            )

        # 使用数据预处理器检查质量
        preprocessor = DataPreprocessor()
        quality_report = preprocessor.check_data_quality(ohlcv_data)

        # 格式化质量报告
        quality_summary = {
            "symbol": symbol,
            "timeframe": timeframe,
            "period": {
                "start": start_date.isoformat() + "Z",
                "end": end_date.isoformat() + "Z",
                "days": days
            },

            "data_completeness": {
                "total_expected_points": quality_report.get('total_expected_points', 0),
                "actual_data_points": quality_report.get('actual_data_points', 0),
                "completeness_ratio": quality_report.get('completeness_ratio', 0),
                "missing_periods": quality_report.get('missing_periods', [])
            },

            "data_validity": {
                "valid_candles": quality_report.get('valid_candles', 0),
                "invalid_candles": quality_report.get('invalid_candles', 0),
                "zero_volume_candles": quality_report.get('zero_volume_candles', 0),
                "price_anomalies": quality_report.get('price_anomalies', 0)
            },

            "statistical_summary": {
                "price_range": quality_report.get('price_range', {}),
                "volume_stats": quality_report.get('volume_stats', {}),
                "volatility": quality_report.get('volatility', 0),
                "data_consistency_score": quality_report.get('consistency_score', 0)
            },

            "issues_found": quality_report.get('issues', []),
            "recommendations": quality_report.get('recommendations', []),

            "overall_quality_score": quality_report.get('quality_score', 0),
            "quality_grade": (
                "excellent" if quality_report.get('quality_score', 0) >= 90 else
                "good" if quality_report.get('quality_score', 0) >= 75 else
                "fair" if quality_report.get('quality_score', 0) >= 60 else
                "poor"
            )
        }

        return await create_response(
            data=quality_summary,
            message="数据质量检查完成"
        )

    except Exception as e:
        logger.error(f"数据质量检查失败: {e}")
        return await create_error_response(
            message="数据质量检查失败",
            code="DATA_QUALITY_ERROR",
            details=str(e)
        )

@router.post("/refresh/{symbol}")

async def refresh_data(
    symbol: str,
    request: Request,
    timeframes: Optional[str] = Query(None, description="要刷新的时间框架，逗号分隔"),
    force: bool = Query(False, description="强制刷新"),
    data_manager: DataManager = Depends(get_data_manager),
    current_user: dict = Depends(get_current_user)
):
    """
    刷新数据
    """
    try:
        symbol = symbol.upper()

        # 解析时间框架
        timeframe_list = ['1h']  # 默认时间框架
        if timeframes:
            timeframe_list = [tf.strip() for tf in timeframes.split(',')]

        refresh_results = {}

        for timeframe in timeframe_list:
            try:
                # 刷新指定交易对和时间框架的数据
                result = await data_manager.refresh_symbol_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    force_refresh=force
                )

                refresh_results[timeframe] = {
                    "success": result.get('success', False),
                    "new_data_points": result.get('new_data_points', 0),
                    "last_update": result.get('last_update'),
                    "message": result.get('message', '')
                }

            except Exception as e:
                refresh_results[timeframe] = {
                    "success": False,
                    "error": str(e)
                }

        # 总结刷新结果
        total_new_points = sum(r.get('new_data_points', 0) for r in refresh_results.values() if r.get('success'))
        successful_refreshes = sum(1 for r in refresh_results.values() if r.get('success'))

        overall_success = successful_refreshes > 0

        return await create_response(
            data={
                "symbol": symbol,
                "refresh_results": refresh_results,
                "summary": {
                    "successful_refreshes": successful_refreshes,
                    "total_timeframes": len(timeframe_list),
                    "total_new_data_points": total_new_points,
                    "overall_success": overall_success
                },
                "timestamp": datetime.now().isoformat() + "Z"
            },
            message=f"数据刷新完成，成功刷新 {successful_refreshes}/{len(timeframe_list)} 个时间框架"
        )

    except Exception as e:
        logger.error(f"刷新数据失败: {e}")
        return await create_error_response(
            message="刷新数据失败",
            code="DATA_REFRESH_ERROR",
            details=str(e)
        )
