"""
Monitoring API Routes
监控系统API路由，集成现有的SystemMonitor和AlertingSystem
提供系统监控、告警管理和健康检查功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入依赖和配置
from api.dependencies import (
    get_system_monitor, get_alerting_system, get_notification_service, get_health_checker,
    get_monitoring_params, get_current_user, create_response, create_error_response
)

# 导入现有系统组件
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.alerting import AlertingSystem
from src.monitoring.notifications import NotificationService
from src.monitoring.health_checker import HealthChecker
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

# 创建路由器
router = APIRouter()

@router.get("/metrics/current")

async def get_current_metrics(
    request: Request,
    metric_types: Optional[str] = Query(None, description="指标类型筛选，逗号分隔"),
    include_history: bool = Query(False, description="包含历史数据"),
    system_monitor: SystemMonitor = Depends(get_system_monitor),
    current_user: dict = Depends(get_current_user)
):
    """
    获取当前系统指标
    集成现有的SystemMonitor.get_current_metrics()
    """
    try:
        # 解析指标类型筛选
        filter_types = None
        if metric_types:
            filter_types = [t.strip().lower() for t in metric_types.split(',')]

        # 获取当前指标
        current_metrics = system_monitor.get_current_metrics()

        # 应用筛选
        if filter_types:
            filtered_metrics = {}
            for metric_name, metric_data in current_metrics.items():
                metric_type = metric_data.get('category', '').lower()
                if any(filter_type in metric_type or filter_type in metric_name.lower() for filter_type in filter_types):
                    filtered_metrics[metric_name] = metric_data
            current_metrics = filtered_metrics

        # 如果需要包含历史数据
        if include_history:
            try:
                for metric_name in current_metrics:
                    history = system_monitor.get_metric_history(
                        metric_name=metric_name,
                        time_range="1h"
                    )
                    current_metrics[metric_name]["history"] = history
            except Exception as e:
                logger.warning(f"获取历史指标失败: {e}")

        # 计算指标摘要
        metrics_summary = {
            "total_metrics": len(current_metrics),
            "healthy_metrics": len([m for m in current_metrics.values() if m.get('status') == 'healthy']),
            "warning_metrics": len([m for m in current_metrics.values() if m.get('status') == 'warning']),
            "critical_metrics": len([m for m in current_metrics.values() if m.get('status') == 'critical']),
            "last_update": max((m.get('timestamp', '1970-01-01T00:00:00Z') for m in current_metrics.values()), default='1970-01-01T00:00:00Z')
        }

        return await create_response(
            data={
                "metrics": current_metrics,
                "summary": metrics_summary,
                "filters": {
                    "metric_types": filter_types,
                    "include_history": include_history
                },
                "timestamp": datetime.now().isoformat() + "Z"
            },
            message="当前系统指标获取成功"
        )

    except Exception as e:
        logger.error(f"获取当前指标失败: {e}")
        return await create_error_response(
            message="获取当前指标失败",
            code="CURRENT_METRICS_ERROR",
            details=str(e)
        )

@router.get("/metrics/history")

async def get_metrics_history(
    request: Request,
    metric_name: str = Query(..., description="指标名称"),
    time_range: str = Query("1h", description="时间范围: 1h, 6h, 1d, 7d"),
    granularity: str = Query("1m", description="数据粒度: 1m, 5m, 15m, 1h"),
    system_monitor: SystemMonitor = Depends(get_system_monitor),
    current_user: dict = Depends(get_current_user)
):
    """
    获取指标历史数据
    """
    try:
        # 获取指标历史数据
        history_data = system_monitor.get_metric_history(
            metric_name=metric_name,
            time_range=time_range,
            granularity=granularity
        )

        if not history_data:
            return await create_error_response(
                message=f"指标 {metric_name} 的历史数据不存在",
                code="METRIC_HISTORY_NOT_FOUND"
            )

        # 计算历史统计
        values = [point.get('value', 0) for point in history_data if point.get('value') is not None]

        if values:
            stats = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "average": sum(values) / len(values),
                "latest": values[-1] if values else 0,
                "trend": "increasing" if len(values) >= 2 and values[-1] > values[0] else "decreasing" if len(values) >= 2 and values[-1] < values[0] else "stable"
            }
        else:
            stats = {"count": 0}

        return await create_response(
            data={
                "metric_name": metric_name,
                "time_range": time_range,
                "granularity": granularity,
                "data": history_data,
                "statistics": stats
            },
            message="指标历史数据获取成功"
        )

    except Exception as e:
        logger.error(f"获取指标历史失败: {e}")
        return await create_error_response(
            message="获取指标历史失败",
            code="METRIC_HISTORY_ERROR",
            details=str(e)
        )

@router.get("/alerts")

async def get_alerts(
    request: Request,
    severity: Optional[str] = Query(None, description="严重级别筛选: low, medium, high, critical"),
    status: Optional[str] = Query(None, description="状态筛选: active, acknowledged, resolved"),
    category: Optional[str] = Query(None, description="类别筛选: system, trading, data"),
    limit: int = Query(100, description="返回数量限制"),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
    current_user: dict = Depends(get_current_user)
):
    """
    获取系统告警
    集成现有的AlertingSystem.get_alerts()
    """
    try:
        # 获取告警列表
        alerts = alerting_system.get_alerts(
            severity=severity,
            status=status,
            category=category,
            limit=limit
        )

        # 格式化告警数据
        formatted_alerts = []
        for alert in alerts:
            alert_data = {
                "id": alert.get('id'),
                "title": alert.get('title'),
                "message": alert.get('message'),
                "severity": alert.get('severity'),
                "category": alert.get('category'),
                "status": alert.get('status'),
                "source": alert.get('source'),
                "metric_name": alert.get('metric_name'),
                "current_value": alert.get('current_value'),
                "threshold": alert.get('threshold'),
                "created_at": alert.get('created_at'),
                "updated_at": alert.get('updated_at'),
                "acknowledged_at": alert.get('acknowledged_at'),
                "acknowledged_by": alert.get('acknowledged_by'),
                "resolved_at": alert.get('resolved_at'),
                "tags": alert.get('tags', []),
                "metadata": alert.get('metadata', {})
            }
            formatted_alerts.append(alert_data)

        # 计算告警统计
        stats = {
            "total_alerts": len(formatted_alerts),
            "by_severity": {},
            "by_status": {},
            "by_category": {},
            "unacknowledged": len([a for a in formatted_alerts if a.get('status') == 'active']),
            "resolved_today": len([a for a in formatted_alerts if a.get('resolved_at') and
                                  a['resolved_at'].startswith(datetime.now().strftime('%Y-%m-%d'))])
        }

        # 按维度统计
        for alert in formatted_alerts:
            severity_key = alert.get('severity', 'unknown')
            status_key = alert.get('status', 'unknown')
            category_key = alert.get('category', 'unknown')

            stats["by_severity"][severity_key] = stats["by_severity"].get(severity_key, 0) + 1
            stats["by_status"][status_key] = stats["by_status"].get(status_key, 0) + 1
            stats["by_category"][category_key] = stats["by_category"].get(category_key, 0) + 1

        return await create_response(
            data={
                "alerts": formatted_alerts,
                "statistics": stats,
                "filters": {
                    "severity": severity,
                    "status": status,
                    "category": category,
                    "limit": limit
                }
            },
            message="系统告警获取成功"
        )

    except Exception as e:
        logger.error(f"获取系统告警失败: {e}")
        return await create_error_response(
            message="获取系统告警失败",
            code="ALERTS_ERROR",
            details=str(e)
        )

@router.post("/alerts/{alert_id}/acknowledge")

async def acknowledge_alert(
    alert_id: str,
    request: Request,
    notes: Optional[str] = Query(None, description="确认备注"),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
    current_user: dict = Depends(get_current_user)
):
    """
    确认告警
    """
    try:
        # 确认告警
        result = await alerting_system.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=current_user.get('user_id', 'unknown'),
            notes=notes
        )

        if result.get('success'):
            return await create_response(
                data={
                    "alert_id": alert_id,
                    "acknowledged_by": current_user.get('user_id'),
                    "acknowledged_at": datetime.now().isoformat() + "Z",
                    "notes": notes
                },
                message="告警确认成功"
            )
        else:
            return await create_error_response(
                message="告警确认失败",
                code="ALERT_ACKNOWLEDGE_ERROR",
                details=result.get('error', '未知错误')
            )

    except Exception as e:
        logger.error(f"确认告警失败: {e}")
        return await create_error_response(
            message="确认告警失败",
            code="ALERT_ACKNOWLEDGE_ERROR",
            details=str(e)
        )

@router.post("/alerts/{alert_id}/resolve")

async def resolve_alert(
    alert_id: str,
    request: Request,
    resolution_notes: Optional[str] = Query(None, description="解决备注"),
    alerting_system: AlertingSystem = Depends(get_alerting_system),
    current_user: dict = Depends(get_current_user)
):
    """
    解决告警
    """
    try:
        # 解决告警
        result = await alerting_system.resolve_alert(
            alert_id=alert_id,
            resolved_by=current_user.get('user_id', 'unknown'),
            resolution_notes=resolution_notes
        )

        if result.get('success'):
            return await create_response(
                data={
                    "alert_id": alert_id,
                    "resolved_by": current_user.get('user_id'),
                    "resolved_at": datetime.now().isoformat() + "Z",
                    "resolution_notes": resolution_notes
                },
                message="告警解决成功"
            )
        else:
            return await create_error_response(
                message="告警解决失败",
                code="ALERT_RESOLVE_ERROR",
                details=result.get('error', '未知错误')
            )

    except Exception as e:
        logger.error(f"解决告警失败: {e}")
        return await create_error_response(
            message="解决告警失败",
            code="ALERT_RESOLVE_ERROR",
            details=str(e)
        )

@router.get("/health")

async def get_system_health(
    request: Request,
    detailed: bool = Query(False, description="获取详细健康信息"),
    health_checker: HealthChecker = Depends(get_health_checker),
    current_user: dict = Depends(get_current_user)
):
    """
    获取系统健康状态
    集成现有的HealthChecker功能
    """
    try:
        # 获取系统健康状态
        health_status = await health_checker.check_system_health(detailed=detailed)

        # 格式化健康状态
        health_summary = {
            "overall_status": health_status.get('overall_status', 'unknown'),
            "health_score": health_status.get('health_score', 0),
            "last_check": health_status.get('last_check', datetime.now().isoformat() + "Z"),

            # 组件健康状态
            "components": health_status.get('components', {}),

            # 系统资源状态
            "system_resources": health_status.get('system_resources', {}),

            # 服务状态
            "services": health_status.get('services', {})
        }

        # 如果需要详细信息
        if detailed:
            health_summary.update({
                "detailed_checks": health_status.get('detailed_checks', {}),
                "performance_metrics": health_status.get('performance_metrics', {}),
                "recommendations": health_status.get('recommendations', []),
                "recent_issues": health_status.get('recent_issues', [])
            })

        # 确定HTTP状态码
        http_status = 200
        if health_status.get('overall_status') == 'unhealthy':
            http_status = 503
        elif health_status.get('overall_status') == 'degraded':
            http_status = 200  # 降级但仍可服务

        return await create_response(
            data=health_summary,
            message="系统健康检查完成"
        )

    except Exception as e:
        logger.error(f"系统健康检查失败: {e}")
        return await create_error_response(
            message="系统健康检查失败",
            code="HEALTH_CHECK_ERROR",
            details=str(e)
        )

@router.get("/notifications/status")

async def get_notification_status(
    request: Request,
    notification_service: NotificationService = Depends(get_notification_service),
    current_user: dict = Depends(get_current_user)
):
    """
    获取通知服务状态
    """
    try:
        # 获取通知服务状态
        notification_status = await notification_service.get_service_status()

        # 格式化状态信息
        status_summary = {
            "service_status": notification_status.get('status', 'unknown'),
            "enabled_channels": notification_status.get('enabled_channels', []),
            "disabled_channels": notification_status.get('disabled_channels', []),

            # 统计信息
            "statistics": {
                "total_sent_today": notification_status.get('total_sent_today', 0),
                "successful_deliveries": notification_status.get('successful_deliveries', 0),
                "failed_deliveries": notification_status.get('failed_deliveries', 0),
                "delivery_rate": notification_status.get('delivery_rate', 0)
            },

            # 各通道状态
            "channel_status": notification_status.get('channel_status', {}),

            # 最近通知
            "recent_notifications": notification_status.get('recent_notifications', []),

            "last_update": notification_status.get('last_update', datetime.now().isoformat() + "Z")
        }

        return await create_response(
            data=status_summary,
            message="通知服务状态获取成功"
        )

    except Exception as e:
        logger.error(f"获取通知服务状态失败: {e}")
        return await create_error_response(
            message="获取通知服务状态失败",
            code="NOTIFICATION_STATUS_ERROR",
            details=str(e)
        )

@router.post("/notifications/test")

async def test_notification(
    request: Request,
    channel: str = Query(..., description="通知渠道: telegram, email, discord, slack"),
    message: str = Query("This is a test notification", description="测试消息"),
    notification_service: NotificationService = Depends(get_notification_service),
    current_user: dict = Depends(get_current_user)
):
    """
    测试通知功能
    """
    try:
        # 发送测试通知
        result = await notification_service.send_notification(
            title="通知测试",
            content=message,
            channel=channel,
            sender=f"API Test by {current_user.get('user_id', 'unknown')}"
        )

        if result.get('success'):
            return await create_response(
                data={
                    "channel": channel,
                    "message": message,
                    "sent_at": datetime.now().isoformat() + "Z",
                    "delivery_info": result.get('delivery_info', {})
                },
                message="测试通知发送成功"
            )
        else:
            return await create_error_response(
                message="测试通知发送失败",
                code="TEST_NOTIFICATION_ERROR",
                details=result.get('error', '未知错误')
            )

    except Exception as e:
        logger.error(f"测试通知失败: {e}")
        return await create_error_response(
            message="测试通知失败",
            code="TEST_NOTIFICATION_ERROR",
            details=str(e)
        )

@router.get("/dashboard-data")

async def get_dashboard_data(
    request: Request,
    time_range: str = Query("1h", description="数据时间范围"),
    system_monitor: SystemMonitor = Depends(get_system_monitor),
    current_user: dict = Depends(get_current_user)
):
    """
    获取监控仪表盘数据
    聚合系统各项指标用于前端展示
    """
    try:
        # 获取当前关键指标
        current_metrics = system_monitor.get_current_metrics()

        # 提取关键系统指标
        key_metrics = {
            "cpu_usage": current_metrics.get('cpu_usage', {}).get('value', 0),
            "memory_usage": current_metrics.get('memory_usage', {}).get('value', 0),
            "disk_usage": current_metrics.get('disk_usage', {}).get('value', 0),
            "network_io": {
                "bytes_sent": current_metrics.get('network_bytes_sent', {}).get('value', 0),
                "bytes_received": current_metrics.get('network_bytes_received', {}).get('value', 0)
            }
        }

        # 获取历史趋势数据（简化版）
        trends = {}
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
            try:
                history = system_monitor.get_metric_history(metric_name, time_range)
                if history:
                    trends[metric_name] = {
                        "current": history[-1].get('value', 0) if history else 0,
                        "previous": history[-2].get('value', 0) if len(history) >= 2 else 0,
                        "trend": "up" if len(history) >= 2 and history[-1].get('value', 0) > history[-2].get('value', 0) else "down"
                    }
            except:
                trends[metric_name] = {"current": 0, "previous": 0, "trend": "stable"}

        # 系统状态总览
        system_status = {
            "overall_health": "healthy",  # 这里应该基于实际指标计算
            "active_alerts": 0,  # 需要从告警系统获取
            "uptime_seconds": current_metrics.get('system_uptime', {}).get('value', 0),
            "last_restart": None
        }

        dashboard_data = {
            "timestamp": datetime.now().isoformat() + "Z",
            "time_range": time_range,
            "key_metrics": key_metrics,
            "trends": trends,
            "system_status": system_status,
            "quick_stats": {
                "metrics_collected": len(current_metrics),
                "healthy_metrics": len([m for m in current_metrics.values() if m.get('status') == 'healthy']),
                "warning_metrics": len([m for m in current_metrics.values() if m.get('status') == 'warning']),
                "critical_metrics": len([m for m in current_metrics.values() if m.get('status') == 'critical'])
            }
        }

        return await create_response(
            data=dashboard_data,
            message="监控仪表盘数据获取成功"
        )

    except Exception as e:
        logger.error(f"获取仪表盘数据失败: {e}")
        return await create_error_response(
            message="获取仪表盘数据失败",
            code="DASHBOARD_DATA_ERROR",
            details=str(e)
        )
