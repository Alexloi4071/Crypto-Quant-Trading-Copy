"""
WebSocket Performance Monitor
WebSocket性能监控系统，专门监控WebSocket连接的性能指标
集成现有监控系统，提供实时性能分析和优化建议
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import statistics
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.alert_manager import AlertManager, AlertLevel
from src.utils.logger import setup_logger
from config.settings import config

logger = setup_logger(__name__)

@dataclass
class WebSocketMetrics:
    """WebSocket性能指标"""
    connection_id: str
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 连接指标
    connection_latency_ms: float = 0.0
    handshake_time_ms: float = 0.0
    last_ping_latency_ms: Optional[float] = None
    
    # 消息指标
    messages_sent_per_sec: float = 0.0
    messages_received_per_sec: float = 0.0
    bytes_sent_per_sec: float = 0.0
    bytes_received_per_sec: float = 0.0
    
    # 累计指标
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    # 错误和重连指标
    error_rate: float = 0.0
    reconnection_count: int = 0
    connection_drops_per_hour: float = 0.0
    
    # 缓冲区和队列指标
    send_queue_size: int = 0
    receive_buffer_size: int = 0
    max_queue_size_reached: int = 0
    
    # 压缩和优化指标
    compression_ratio: Optional[float] = None
    message_processing_time_ms: float = 0.0

@dataclass
class ConnectionPerformanceProfile:
    """连接性能画像"""
    connection_id: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # 性能等级
    performance_grade: str = "Unknown"  # Excellent, Good, Fair, Poor
    bottleneck_factors: List[str] = field(default_factory=list)
    
    # 历史趋势
    latency_trend: str = "stable"  # improving, stable, degrading
    throughput_trend: str = "stable"
    reliability_trend: str = "stable"
    
    # 优化建议
    recommendations: List[str] = field(default_factory=list)
    
    # 风险评估
    stability_risk: str = "low"  # low, medium, high
    performance_risk: str = "low"

class WebSocketPerformanceTracker:
    """WebSocket性能跟踪器"""
    
    def __init__(self, connection_id: str, url: str):
        self.connection_id = connection_id
        self.url = url
        
        # 时序数据窗口
        self.window_size = 1000
        self.time_window_seconds = 300  # 5分钟
        
        # 指标缓冲区
        self.latency_history = deque(maxlen=self.window_size)
        self.throughput_history = deque(maxlen=self.window_size)
        self.error_history = deque(maxlen=self.window_size)
        
        # 消息统计
        self.message_timestamps = deque(maxlen=self.window_size)
        self.byte_counts = deque(maxlen=self.window_size)
        
        # 连接事件
        self.connection_events = deque(maxlen=100)
        
        # 当前统计
        self.current_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0,
            'reconnections': 0,
            'connection_start_time': datetime.now(),
            'last_message_time': None
        }
        
        # 性能阈值
        self.thresholds = {
            'max_latency_ms': 1000,
            'max_error_rate': 0.05,
            'min_throughput_msgs_per_sec': 1.0,
            'max_queue_size': 1000
        }
        
    def record_connection_event(self, event_type: str, details: dict = None):
        """记录连接事件"""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'details': details or {}
        }
        self.connection_events.append(event)
        
        if event_type == 'reconnection':
            self.current_stats['reconnections'] += 1
        elif event_type == 'error':
            self.current_stats['errors'] += 1
    
    def record_message(self, direction: str, size_bytes: int, processing_time_ms: float = None):
        """记录消息"""
        timestamp = datetime.now()
        
        # 更新统计
        if direction == 'sent':
            self.current_stats['messages_sent'] += 1
            self.current_stats['bytes_sent'] += size_bytes
        elif direction == 'received':
            self.current_stats['messages_received'] += 1
            self.current_stats['bytes_received'] += size_bytes
        
        self.current_stats['last_message_time'] = timestamp
        
        # 记录时序数据
        self.message_timestamps.append(timestamp)
        self.byte_counts.append(size_bytes)
        
        # 记录处理时间
        if processing_time_ms is not None:
            self.record_processing_time(processing_time_ms)
    
    def record_latency(self, latency_ms: float):
        """记录延迟"""
        self.latency_history.append({
            'timestamp': datetime.now(),
            'latency_ms': latency_ms
        })
    
    def record_processing_time(self, processing_time_ms: float):
        """记录消息处理时间"""
        self.throughput_history.append({
            'timestamp': datetime.now(),
            'processing_time_ms': processing_time_ms
        })
    
    def record_queue_size(self, send_queue_size: int, receive_buffer_size: int):
        """记录队列大小"""
        if send_queue_size > self.current_stats.get('max_queue_size', 0):
            self.current_stats['max_queue_size'] = send_queue_size
    
    def calculate_metrics(self) -> WebSocketMetrics:
        """计算性能指标"""
        current_time = datetime.now()
        
        # 计算时间窗口内的数据
        window_start = current_time - timedelta(seconds=self.time_window_seconds)
        
        # 过滤时间窗口内的数据
        recent_messages = [ts for ts in self.message_timestamps if ts >= window_start]
        recent_latencies = [item for item in self.latency_history if item['timestamp'] >= window_start]
        
        # 计算吞吐量
        window_duration = self.time_window_seconds
        messages_per_sec = len(recent_messages) / window_duration if window_duration > 0 else 0
        
        # 计算字节速率
        recent_bytes = [bc for bc, ts in zip(self.byte_counts, self.message_timestamps) if ts >= window_start]
        bytes_per_sec = sum(recent_bytes) / window_duration if window_duration > 0 else 0
        
        # 计算延迟统计
        avg_latency = 0
        if recent_latencies:
            latencies = [item['latency_ms'] for item in recent_latencies]
            avg_latency = statistics.mean(latencies)
        
        # 计算错误率
        connection_duration = (current_time - self.current_stats['connection_start_time']).total_seconds()
        error_rate = self.current_stats['errors'] / max(connection_duration / 3600, 1)  # 每小时错误数
        
        # 计算连接掉线率
        connection_drops_per_hour = self.current_stats['reconnections'] / max(connection_duration / 3600, 1)
        
        return WebSocketMetrics(
            connection_id=self.connection_id,
            url=self.url,
            connection_latency_ms=avg_latency,
            last_ping_latency_ms=avg_latency if recent_latencies else None,
            messages_sent_per_sec=messages_per_sec,
            messages_received_per_sec=messages_per_sec,  # 简化处理
            bytes_sent_per_sec=bytes_per_sec,
            bytes_received_per_sec=bytes_per_sec,
            total_messages_sent=self.current_stats['messages_sent'],
            total_messages_received=self.current_stats['messages_received'],
            total_bytes_sent=self.current_stats['bytes_sent'],
            total_bytes_received=self.current_stats['bytes_received'],
            error_rate=error_rate,
            reconnection_count=self.current_stats['reconnections'],
            connection_drops_per_hour=connection_drops_per_hour,
            send_queue_size=0,  # 需要外部提供
            max_queue_size_reached=self.current_stats.get('max_queue_size', 0)
        )
    
    def generate_performance_profile(self) -> ConnectionPerformanceProfile:
        """生成性能画像"""
        metrics = self.calculate_metrics()
        
        # 计算性能等级
        performance_grade = self._calculate_performance_grade(metrics)
        
        # 识别瓶颈因素
        bottlenecks = self._identify_bottlenecks(metrics)
        
        # 生成趋势分析
        latency_trend = self._analyze_latency_trend()
        throughput_trend = self._analyze_throughput_trend()
        reliability_trend = self._analyze_reliability_trend()
        
        # 生成优化建议
        recommendations = self._generate_recommendations(metrics, bottlenecks)
        
        # 评估风险
        stability_risk = self._assess_stability_risk(metrics)
        performance_risk = self._assess_performance_risk(metrics)
        
        return ConnectionPerformanceProfile(
            connection_id=self.connection_id,
            performance_grade=performance_grade,
            bottleneck_factors=bottlenecks,
            latency_trend=latency_trend,
            throughput_trend=throughput_trend,
            reliability_trend=reliability_trend,
            recommendations=recommendations,
            stability_risk=stability_risk,
            performance_risk=performance_risk
        )
    
    def _calculate_performance_grade(self, metrics: WebSocketMetrics) -> str:
        """计算性能等级"""
        score = 100
        
        # 延迟评分
        if metrics.connection_latency_ms > 500:
            score -= 20
        elif metrics.connection_latency_ms > 200:
            score -= 10
        
        # 错误率评分
        if metrics.error_rate > 0.1:
            score -= 30
        elif metrics.error_rate > 0.05:
            score -= 15
        
        # 连接稳定性评分
        if metrics.connection_drops_per_hour > 1:
            score -= 25
        elif metrics.connection_drops_per_hour > 0.5:
            score -= 10
        
        # 吞吐量评分
        if metrics.messages_sent_per_sec < 0.5:
            score -= 15
        
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Poor"
    
    def _identify_bottlenecks(self, metrics: WebSocketMetrics) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        if metrics.connection_latency_ms > self.thresholds['max_latency_ms']:
            bottlenecks.append("High Latency")
        
        if metrics.error_rate > self.thresholds['max_error_rate']:
            bottlenecks.append("High Error Rate")
        
        if metrics.messages_sent_per_sec < self.thresholds['min_throughput_msgs_per_sec']:
            bottlenecks.append("Low Throughput")
        
        if metrics.max_queue_size_reached > self.thresholds['max_queue_size']:
            bottlenecks.append("Queue Overflow")
        
        if metrics.connection_drops_per_hour > 1:
            bottlenecks.append("Connection Instability")
        
        return bottlenecks
    
    def _analyze_latency_trend(self) -> str:
        """分析延迟趋势"""
        if len(self.latency_history) < 10:
            return "stable"
        
        recent_latencies = [item['latency_ms'] for item in list(self.latency_history)[-20:]]
        early_latencies = [item['latency_ms'] for item in list(self.latency_history)[-40:-20]]
        
        if not early_latencies:
            return "stable"
        
        recent_avg = statistics.mean(recent_latencies)
        early_avg = statistics.mean(early_latencies)
        
        change_ratio = (recent_avg - early_avg) / early_avg
        
        if change_ratio < -0.1:
            return "improving"
        elif change_ratio > 0.1:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_throughput_trend(self) -> str:
        """分析吞吐量趋势"""
        if len(self.message_timestamps) < 20:
            return "stable"
        
        current_time = datetime.now()
        recent_window = current_time - timedelta(minutes=2)
        early_window = current_time - timedelta(minutes=4)
        
        recent_count = len([ts for ts in self.message_timestamps if ts >= recent_window])
        early_count = len([ts for ts in self.message_timestamps 
                          if early_window <= ts < recent_window])
        
        if early_count == 0:
            return "stable"
        
        change_ratio = (recent_count - early_count) / early_count
        
        if change_ratio > 0.2:
            return "improving"
        elif change_ratio < -0.2:
            return "degrading"
        else:
            return "stable"
    
    def _analyze_reliability_trend(self) -> str:
        """分析可靠性趋势"""
        recent_events = [event for event in self.connection_events 
                        if event['timestamp'] >= datetime.now() - timedelta(minutes=10)]
        
        error_events = [event for event in recent_events if event['type'] == 'error']
        reconnect_events = [event for event in recent_events if event['type'] == 'reconnection']
        
        if len(error_events) + len(reconnect_events) > 3:
            return "degrading"
        elif len(error_events) + len(reconnect_events) == 0:
            return "improving"
        else:
            return "stable"
    
    def _generate_recommendations(self, metrics: WebSocketMetrics, bottlenecks: List[str]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if "High Latency" in bottlenecks:
            recommendations.append("考虑使用更近的服务器节点或优化网络路径")
            recommendations.append("检查消息处理逻辑，优化处理时间")
        
        if "High Error Rate" in bottlenecks:
            recommendations.append("增加错误处理和重试机制")
            recommendations.append("检查网络稳定性和服务器负载")
        
        if "Low Throughput" in bottlenecks:
            recommendations.append("考虑批量发送消息以提高效率")
            recommendations.append("优化消息序列化和压缩")
        
        if "Queue Overflow" in bottlenecks:
            recommendations.append("增加队列容量或实现背压机制")
            recommendations.append("优化消息消费速度")
        
        if "Connection Instability" in bottlenecks:
            recommendations.append("增强重连机制和连接监控")
            recommendations.append("检查网络环境和服务器稳定性")
        
        if not bottlenecks:
            recommendations.append("性能表现良好，保持当前配置")
        
        return recommendations
    
    def _assess_stability_risk(self, metrics: WebSocketMetrics) -> str:
        """评估稳定性风险"""
        risk_score = 0
        
        if metrics.error_rate > 0.1:
            risk_score += 3
        elif metrics.error_rate > 0.05:
            risk_score += 2
        elif metrics.error_rate > 0.02:
            risk_score += 1
        
        if metrics.connection_drops_per_hour > 2:
            risk_score += 3
        elif metrics.connection_drops_per_hour > 1:
            risk_score += 2
        elif metrics.connection_drops_per_hour > 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_performance_risk(self, metrics: WebSocketMetrics) -> str:
        """评估性能风险"""
        risk_score = 0
        
        if metrics.connection_latency_ms > 1000:
            risk_score += 3
        elif metrics.connection_latency_ms > 500:
            risk_score += 2
        elif metrics.connection_latency_ms > 200:
            risk_score += 1
        
        if metrics.messages_sent_per_sec < 0.1:
            risk_score += 2
        elif metrics.messages_sent_per_sec < 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

class WebSocketPerformanceMonitor:
    """WebSocket性能监控器主类"""
    
    def __init__(self, system_monitor: SystemMonitor = None, alert_manager: AlertManager = None):
        self.system_monitor = system_monitor
        self.alert_manager = alert_manager
        
        # 性能跟踪器
        self.trackers = {}  # connection_id -> WebSocketPerformanceTracker
        
        # 全局统计
        self.global_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'total_messages': 0,
            'total_bytes': 0,
            'average_latency': 0.0,
            'error_rate': 0.0,
            'start_time': datetime.now()
        }
        
        # 性能基线
        self.performance_baseline = {
            'latency_p95': 500,  # 95分位延迟
            'throughput_baseline': 10,  # 基准吞吐量
            'error_rate_baseline': 0.01  # 基准错误率
        }
        
        # 监控任务
        self.monitoring_task = None
        self.reporting_task = None
        self.is_running = False
        
        # 性能报告
        self.performance_reports = deque(maxlen=100)
        
        logger.info("WebSocket性能监控器初始化完成")
    
    async def start(self):
        """启动性能监控器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.reporting_task = asyncio.create_task(self._reporting_loop())
        
        logger.info("WebSocket性能监控器启动完成")
    
    async def stop(self):
        """停止性能监控器"""
        if not self.is_running:
            return
        
        logger.info("正在停止WebSocket性能监控器...")
        self.is_running = False
        
        # 取消监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.reporting_task:
            self.reporting_task.cancel()
        
        logger.info("WebSocket性能监控器已停止")
    
    def register_connection(self, connection_id: str, url: str) -> WebSocketPerformanceTracker:
        """注册WebSocket连接"""
        if connection_id in self.trackers:
            return self.trackers[connection_id]
        
        tracker = WebSocketPerformanceTracker(connection_id, url)
        self.trackers[connection_id] = tracker
        
        self.global_stats['total_connections'] += 1
        
        tracker.record_connection_event('registered')
        logger.debug(f"注册WebSocket连接: {connection_id}")
        
        return tracker
    
    def unregister_connection(self, connection_id: str):
        """注销WebSocket连接"""
        if connection_id in self.trackers:
            tracker = self.trackers[connection_id]
            tracker.record_connection_event('unregistered')
            
            del self.trackers[connection_id]
            logger.debug(f"注销WebSocket连接: {connection_id}")
    
    def get_tracker(self, connection_id: str) -> Optional[WebSocketPerformanceTracker]:
        """获取连接跟踪器"""
        return self.trackers.get(connection_id)
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._collect_metrics()
                await self._check_performance_alerts()
                await asyncio.sleep(10)  # 每10秒收集一次指标
                
            except Exception as e:
                logger.error(f"性能监控循环错误: {e}")
                await asyncio.sleep(10)
    
    async def _collect_metrics(self):
        """收集性能指标"""
        active_connections = 0
        total_messages = 0
        total_bytes = 0
        latencies = []
        errors = 0
        
        for tracker in self.trackers.values():
            metrics = tracker.calculate_metrics()
            
            if metrics.last_ping_latency_ms:
                latencies.append(metrics.last_ping_latency_ms)
            
            total_messages += metrics.total_messages_sent + metrics.total_messages_received
            total_bytes += metrics.total_bytes_sent + metrics.total_bytes_received
            errors += tracker.current_stats['errors']
            
            if tracker.current_stats['last_message_time']:
                # 最近有消息活动的连接视为活跃
                if (datetime.now() - tracker.current_stats['last_message_time']).total_seconds() < 60:
                    active_connections += 1
        
        # 更新全局统计
        self.global_stats['active_connections'] = active_connections
        self.global_stats['total_messages'] = total_messages
        self.global_stats['total_bytes'] = total_bytes
        self.global_stats['average_latency'] = statistics.mean(latencies) if latencies else 0
        
        # 计算错误率
        uptime_hours = (datetime.now() - self.global_stats['start_time']).total_seconds() / 3600
        self.global_stats['error_rate'] = errors / max(uptime_hours, 1)
        
        # 发送到系统监控
        if self.system_monitor:
            await self.system_monitor.record_metric(
                'websocket_active_connections',
                active_connections,
                {'category': 'websocket'}
            )
            
            await self.system_monitor.record_metric(
                'websocket_average_latency',
                self.global_stats['average_latency'],
                {'category': 'websocket', 'unit': 'ms'}
            )
    
    async def _check_performance_alerts(self):
        """检查性能告警"""
        if not self.alert_manager:
            return
        
        # 检查全局指标
        if self.global_stats['average_latency'] > self.performance_baseline['latency_p95']:
            await self._send_performance_alert(
                'high_latency',
                f"平均延迟过高: {self.global_stats['average_latency']:.1f}ms"
            )
        
        if self.global_stats['error_rate'] > self.performance_baseline['error_rate_baseline']:
            await self._send_performance_alert(
                'high_error_rate',
                f"错误率过高: {self.global_stats['error_rate']:.3f}/小时"
            )
        
        # 检查个别连接
        for connection_id, tracker in self.trackers.items():
            metrics = tracker.calculate_metrics()
            
            if metrics.connection_latency_ms > 2000:  # 2秒
                await self._send_performance_alert(
                    f'connection_high_latency_{connection_id}',
                    f"连接 {connection_id} 延迟过高: {metrics.connection_latency_ms:.1f}ms"
                )
            
            if metrics.connection_drops_per_hour > 5:
                await self._send_performance_alert(
                    f'connection_unstable_{connection_id}',
                    f"连接 {connection_id} 不稳定: {metrics.connection_drops_per_hour:.1f} 次/小时"
                )
    
    async def _send_performance_alert(self, alert_id: str, message: str):
        """发送性能告警"""
        try:
            await self.alert_manager.create_alert(
                alert_id=alert_id,
                title="WebSocket性能告警",
                message=message,
                level=AlertLevel.WARNING,
                source="websocket_monitor",
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'category': 'performance'
                }
            )
        except Exception as e:
            logger.error(f"发送性能告警失败: {e}")
    
    async def _reporting_loop(self):
        """性能报告循环"""
        while self.is_running:
            try:
                await self._generate_performance_report()
                await asyncio.sleep(300)  # 每5分钟生成一次报告
            except Exception as e:
                logger.error(f"性能报告循环错误: {e}")
                await asyncio.sleep(300)
    
    async def _generate_performance_report(self):
        """生成性能报告"""
        report_time = datetime.now()
        
        # 收集所有连接的性能画像
        connection_profiles = {}
        for connection_id, tracker in self.trackers.items():
            profile = tracker.generate_performance_profile()
            connection_profiles[connection_id] = profile
        
        # 生成聚合报告
        report = {
            'timestamp': report_time,
            'global_stats': self.global_stats.copy(),
            'connection_count': len(self.trackers),
            'connection_profiles': connection_profiles,
            'performance_summary': self._generate_performance_summary(connection_profiles)
        }
        
        self.performance_reports.append(report)
        
        # 记录关键指标到日志
        logger.info(
            f"WebSocket性能报告 - "
            f"连接数: {len(self.trackers)}, "
            f"活跃连接: {self.global_stats['active_connections']}, "
            f"平均延迟: {self.global_stats['average_latency']:.1f}ms, "
            f"错误率: {self.global_stats['error_rate']:.3f}/小时"
        )
    
    def _generate_performance_summary(self, profiles: Dict[str, ConnectionPerformanceProfile]) -> dict:
        """生成性能摘要"""
        if not profiles:
            return {}
        
        grade_counts = defaultdict(int)
        risk_counts = defaultdict(int)
        common_bottlenecks = defaultdict(int)
        
        for profile in profiles.values():
            grade_counts[profile.performance_grade] += 1
            risk_counts[profile.stability_risk] += 1
            
            for bottleneck in profile.bottleneck_factors:
                common_bottlenecks[bottleneck] += 1
        
        return {
            'grade_distribution': dict(grade_counts),
            'risk_distribution': dict(risk_counts),
            'common_bottlenecks': dict(sorted(common_bottlenecks.items(), 
                                            key=lambda x: x[1], reverse=True)[:5])
        }
    
    def get_stats(self) -> dict:
        """获取监控器统计信息"""
        connection_stats = {}
        
        for connection_id, tracker in self.trackers.items():
            metrics = tracker.calculate_metrics()
            profile = tracker.generate_performance_profile()
            
            connection_stats[connection_id] = {
                'url': tracker.url,
                'metrics': {
                    'latency_ms': metrics.connection_latency_ms,
                    'messages_per_sec': metrics.messages_sent_per_sec,
                    'error_rate': metrics.error_rate,
                    'reconnections': metrics.reconnection_count
                },
                'profile': {
                    'performance_grade': profile.performance_grade,
                    'bottlenecks': profile.bottleneck_factors,
                    'stability_risk': profile.stability_risk
                }
            }
        
        return {
            'global_stats': self.global_stats,
            'connection_stats': connection_stats,
            'performance_baseline': self.performance_baseline,
            'is_running': self.is_running,
            'report_count': len(self.performance_reports)
        }
    
    def get_recent_reports(self, limit: int = 10) -> List[dict]:
        """获取最近的性能报告"""
        recent_reports = list(self.performance_reports)[-limit:]
        return [
            {
                **report,
                'timestamp': report['timestamp'].isoformat()
            }
            for report in recent_reports
        ]

# 全局实例
_ws_performance_monitor_instance = None

def get_websocket_performance_monitor() -> WebSocketPerformanceMonitor:
    """获取WebSocket性能监控器实例"""
    global _ws_performance_monitor_instance
    if _ws_performance_monitor_instance is None:
        _ws_performance_monitor_instance = WebSocketPerformanceMonitor()
    return _ws_performance_monitor_instance