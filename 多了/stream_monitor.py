"""
Stream Data Monitor
流式数据监控系统，提供实时数据流的监控、分析和告警功能
集成现有监控系统，提供数据质量检查和异常检测
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, asdict
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
class DataQualityMetrics:
    """数据质量指标"""
    symbol: str
    data_type: str
    timestamp: datetime
    
    # 延迟指标
    latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # 吞吐量指标
    throughput_per_second: float = 0.0
    total_messages: int = 0
    
    # 质量指标
    error_rate: float = 0.0
    duplicate_rate: float = 0.0
    missing_data_rate: float = 0.0
    
    # 数据完整性
    completeness_score: float = 1.0  # 0-1之间
    consistency_score: float = 1.0   # 0-1之间
    freshness_score: float = 1.0     # 0-1之间

@dataclass
class StreamHealthStatus:
    """数据流健康状态"""
    stream_name: str
    status: str  # healthy, degraded, unhealthy, offline
    health_score: float  # 0-100
    last_data_time: Optional[datetime]
    issues: List[str]
    metrics: DataQualityMetrics

class DataStreamMonitor:
    """单个数据流监控器"""
    
    def __init__(self, stream_name: str, config: dict = None):
        self.stream_name = stream_name
        self.config = config or {}
        
        # 监控窗口大小
        self.window_size = self.config.get('window_size', 1000)
        self.time_window_seconds = self.config.get('time_window', 60)
        
        # 数据缓冲区
        self.message_buffer = deque(maxlen=self.window_size)
        self.latency_buffer = deque(maxlen=self.window_size)
        self.error_buffer = deque(maxlen=self.window_size)
        
        # 时间窗口数据
        self.time_window_data = defaultdict(list)
        
        # 统计计数器
        self.counters = {
            'total_messages': 0,
            'error_messages': 0,
            'duplicate_messages': 0,
            'missing_sequences': 0,
            'out_of_order': 0
        }
        
        # 最后收到的数据
        self.last_message_time = None
        self.last_sequence = None
        self.expected_fields = set()
        
        # 阈值配置
        self.thresholds = {
            'max_latency_ms': self.config.get('max_latency_ms', 1000),
            'max_error_rate': self.config.get('max_error_rate', 0.05),  # 5%
            'min_throughput': self.config.get('min_throughput', 1.0),   # 1 msg/s
            'max_staleness_seconds': self.config.get('max_staleness_seconds', 30)
        }
        
        logger.info(f"数据流监控器初始化: {stream_name}")
    
    def process_message(self, message: dict, received_time: datetime = None) -> DataQualityMetrics:
        """处理接收到的消息"""
        received_time = received_time or datetime.now()
        
        try:
            # 更新计数器
            self.counters['total_messages'] += 1
            self.last_message_time = received_time
            
            # 检查消息完整性
            self._check_message_completeness(message)
            
            # 检查时间戳和延迟
            latency_ms = self._calculate_latency(message, received_time)
            if latency_ms is not None:
                self.latency_buffer.append(latency_ms)
            
            # 检查序列号（如果存在）
            self._check_message_sequence(message)
            
            # 添加到消息缓冲区
            self.message_buffer.append({
                'message': message,
                'received_time': received_time,
                'latency_ms': latency_ms
            })
            
            # 添加到时间窗口
            self._add_to_time_window(received_time, message)
            
            # 计算质量指标
            return self._calculate_quality_metrics()
            
        except Exception as e:
            logger.error(f"处理消息时出错 {self.stream_name}: {e}")
            self.counters['error_messages'] += 1
            self.error_buffer.append({
                'error': str(e),
                'time': received_time,
                'message': message
            })
            return self._calculate_quality_metrics()
    
    def _check_message_completeness(self, message: dict):
        """检查消息完整性"""
        if not isinstance(message, dict):
            raise ValueError("消息必须是字典类型")
        
        # 学习期望的字段
        current_fields = set(message.keys())
        if not self.expected_fields:
            self.expected_fields = current_fields
        else:
            # 检查是否有缺失字段
            missing_fields = self.expected_fields - current_fields
            if missing_fields:
                self.counters['missing_sequences'] += 1
                logger.warning(f"消息缺失字段 {self.stream_name}: {missing_fields}")
    
    def _calculate_latency(self, message: dict, received_time: datetime) -> Optional[float]:
        """计算消息延迟"""
        try:
            # 尝试从消息中提取时间戳
            msg_timestamp = None
            
            # 常见的时间戳字段
            timestamp_fields = ['timestamp', 'time', 'created_at', 'event_time']
            
            for field in timestamp_fields:
                if field in message:
                    timestamp_value = message[field]
                    
                    if isinstance(timestamp_value, str):
                        # 尝试解析ISO格式时间戳
                        try:
                            msg_timestamp = datetime.fromisoformat(timestamp_value.replace('Z', ''))
                            break
                        except ValueError:
                            continue
                    elif isinstance(timestamp_value, (int, float)):
                        # Unix时间戳
                        if timestamp_value > 1e12:  # 毫秒时间戳
                            msg_timestamp = datetime.fromtimestamp(timestamp_value / 1000)
                        else:  # 秒时间戳
                            msg_timestamp = datetime.fromtimestamp(timestamp_value)
                        break
            
            if msg_timestamp:
                latency = (received_time - msg_timestamp).total_seconds() * 1000
                return max(0, latency)  # 确保非负
            
        except Exception as e:
            logger.debug(f"计算延迟失败: {e}")
        
        return None
    
    def _check_message_sequence(self, message: dict):
        """检查消息序列"""
        # 检查序列号字段
        sequence_fields = ['sequence', 'seq', 'id', 'event_id']
        
        for field in sequence_fields:
            if field in message:
                try:
                    current_seq = int(message[field])
                    
                    if self.last_sequence is not None:
                        if current_seq <= self.last_sequence:
                            if current_seq == self.last_sequence:
                                self.counters['duplicate_messages'] += 1
                            else:
                                self.counters['out_of_order'] += 1
                        elif current_seq > self.last_sequence + 1:
                            # 跳过了一些序列号
                            self.counters['missing_sequences'] += current_seq - self.last_sequence - 1
                    
                    self.last_sequence = current_seq
                    break
                    
                except (ValueError, TypeError):
                    continue
    
    def _add_to_time_window(self, timestamp: datetime, message: dict):
        """添加到时间窗口"""
        # 清理过期的时间窗口数据
        cutoff_time = timestamp - timedelta(seconds=self.time_window_seconds)
        
        # 清理过期数据
        for ts in list(self.time_window_data.keys()):
            if ts < cutoff_time:
                del self.time_window_data[ts]
        
        # 添加新数据
        self.time_window_data[timestamp].append(message)
    
    def _calculate_quality_metrics(self) -> DataQualityMetrics:
        """计算数据质量指标"""
        current_time = datetime.now()
        
        # 延迟指标
        latency_ms = 0.0
        max_latency_ms = 0.0
        avg_latency_ms = 0.0
        
        if self.latency_buffer:
            latencies = list(self.latency_buffer)
            latency_ms = latencies[-1] if latencies else 0.0
            max_latency_ms = max(latencies)
            avg_latency_ms = statistics.mean(latencies)
        
        # 吞吐量指标（基于时间窗口）
        total_in_window = sum(len(msgs) for msgs in self.time_window_data.values())
        throughput_per_second = total_in_window / max(self.time_window_seconds, 1)
        
        # 错误率
        total_messages = max(self.counters['total_messages'], 1)
        error_rate = self.counters['error_messages'] / total_messages
        duplicate_rate = self.counters['duplicate_messages'] / total_messages
        missing_data_rate = self.counters['missing_sequences'] / total_messages
        
        # 数据完整性评分
        completeness_score = max(0, 1 - missing_data_rate - error_rate)
        consistency_score = max(0, 1 - duplicate_rate - (self.counters['out_of_order'] / total_messages))
        
        # 数据新鲜度评分
        freshness_score = 1.0
        if self.last_message_time:
            staleness = (current_time - self.last_message_time).total_seconds()
            if staleness > self.thresholds['max_staleness_seconds']:
                freshness_score = max(0, 1 - staleness / (self.thresholds['max_staleness_seconds'] * 2))
        
        return DataQualityMetrics(
            symbol=self.stream_name,
            data_type="stream",
            timestamp=current_time,
            latency_ms=latency_ms,
            max_latency_ms=max_latency_ms,
            avg_latency_ms=avg_latency_ms,
            throughput_per_second=throughput_per_second,
            total_messages=total_messages,
            error_rate=error_rate,
            duplicate_rate=duplicate_rate,
            missing_data_rate=missing_data_rate,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score
        )
    
    def get_health_status(self) -> StreamHealthStatus:
        """获取流健康状态"""
        metrics = self._calculate_quality_metrics()
        issues = []
        
        # 检查各种问题
        if metrics.latency_ms > self.thresholds['max_latency_ms']:
            issues.append(f"高延迟: {metrics.latency_ms:.1f}ms")
        
        if metrics.error_rate > self.thresholds['max_error_rate']:
            issues.append(f"错误率过高: {metrics.error_rate:.1%}")
        
        if metrics.throughput_per_second < self.thresholds['min_throughput']:
            issues.append(f"吞吐量过低: {metrics.throughput_per_second:.1f}/s")
        
        if metrics.freshness_score < 0.5:
            issues.append("数据过旧")
        
        # 计算健康分数
        health_score = (
            metrics.completeness_score * 30 +
            metrics.consistency_score * 30 +
            metrics.freshness_score * 20 +
            (1 - min(metrics.error_rate * 10, 1)) * 20
        )
        
        # 确定状态
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 50:
            status = "unhealthy"
        else:
            status = "offline"
        
        return StreamHealthStatus(
            stream_name=self.stream_name,
            status=status,
            health_score=health_score,
            last_data_time=self.last_message_time,
            issues=issues,
            metrics=metrics
        )
    
    def reset_stats(self):
        """重置统计信息"""
        self.counters = {key: 0 for key in self.counters}
        self.message_buffer.clear()
        self.latency_buffer.clear()
        self.error_buffer.clear()
        self.time_window_data.clear()

class StreamDataMonitorManager:
    """流式数据监控管理器"""
    
    def __init__(self, system_monitor: SystemMonitor = None, alert_manager: AlertManager = None):
        self.system_monitor = system_monitor
        self.alert_manager = alert_manager
        
        # 流监控器
        self.stream_monitors = {}
        
        # 全局统计
        self.global_stats = {
            'total_streams': 0,
            'healthy_streams': 0,
            'degraded_streams': 0,
            'unhealthy_streams': 0,
            'offline_streams': 0,
            'total_messages': 0,
            'total_errors': 0,
            'start_time': datetime.now()
        }
        
        # 监控任务
        self.monitoring_task = None
        self.is_running = False
        
        # 告警历史
        self.alert_history = deque(maxlen=1000)
        
        logger.info("流式数据监控管理器初始化完成")
    
    async def start(self):
        """启动监控管理器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("流式数据监控管理器启动完成")
    
    async def stop(self):
        """停止监控管理器"""
        if not self.is_running:
            return
        
        logger.info("正在停止流式数据监控管理器...")
        self.is_running = False
        
        # 取消监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("流式数据监控管理器已停止")
    
    def register_stream(self, stream_name: str, config: dict = None) -> DataStreamMonitor:
        """注册数据流监控"""
        if stream_name in self.stream_monitors:
            logger.warning(f"数据流 {stream_name} 已注册")
            return self.stream_monitors[stream_name]
        
        monitor = DataStreamMonitor(stream_name, config)
        self.stream_monitors[stream_name] = monitor
        self.global_stats['total_streams'] += 1
        
        logger.info(f"已注册数据流监控: {stream_name}")
        return monitor
    
    def unregister_stream(self, stream_name: str):
        """取消注册数据流"""
        if stream_name in self.stream_monitors:
            del self.stream_monitors[stream_name]
            self.global_stats['total_streams'] -= 1
            logger.info(f"已取消注册数据流: {stream_name}")
    
    async def process_stream_message(self, stream_name: str, message: dict, received_time: datetime = None) -> bool:
        """处理流消息"""
        if stream_name not in self.stream_monitors:
            logger.debug(f"未注册的数据流: {stream_name}")
            return False
        
        try:
            monitor = self.stream_monitors[stream_name]
            metrics = monitor.process_message(message, received_time)
            
            # 更新全局统计
            self.global_stats['total_messages'] += 1
            
            # 检查是否需要告警
            await self._check_stream_alerts(stream_name, metrics)
            
            return True
            
        except Exception as e:
            logger.error(f"处理流消息失败 {stream_name}: {e}")
            self.global_stats['total_errors'] += 1
            return False
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._update_global_stats()
                await self._check_stream_health()
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(10)
    
    async def _update_global_stats(self):
        """更新全局统计"""
        healthy = degraded = unhealthy = offline = 0
        
        for stream_name, monitor in self.stream_monitors.items():
            health_status = monitor.get_health_status()
            
            if health_status.status == 'healthy':
                healthy += 1
            elif health_status.status == 'degraded':
                degraded += 1
            elif health_status.status == 'unhealthy':
                unhealthy += 1
            else:
                offline += 1
        
        self.global_stats.update({
            'healthy_streams': healthy,
            'degraded_streams': degraded,
            'unhealthy_streams': unhealthy,
            'offline_streams': offline
        })
    
    async def _check_stream_health(self):
        """检查流健康状态"""
        for stream_name, monitor in self.stream_monitors.items():
            health_status = monitor.get_health_status()
            
            # 发送健康状态到系统监控
            if self.system_monitor:
                metric_name = f"stream_health_{stream_name}"
                await self.system_monitor.record_metric(
                    metric_name,
                    health_status.health_score,
                    {"stream": stream_name, "status": health_status.status}
                )
    
    async def _check_stream_alerts(self, stream_name: str, metrics: DataQualityMetrics):
        """检查流告警"""
        if not self.alert_manager:
            return
        
        monitor = self.stream_monitors[stream_name]
        
        # 高延迟告警
        if metrics.latency_ms > monitor.thresholds['max_latency_ms']:
            alert_id = f"high_latency_{stream_name}"
            await self._send_alert(
                alert_id,
                AlertLevel.WARNING,
                f"数据流 {stream_name} 延迟过高",
                f"当前延迟: {metrics.latency_ms:.1f}ms, 阈值: {monitor.thresholds['max_latency_ms']}ms"
            )
        
        # 错误率告警
        if metrics.error_rate > monitor.thresholds['max_error_rate']:
            alert_id = f"high_error_rate_{stream_name}"
            await self._send_alert(
                alert_id,
                AlertLevel.CRITICAL,
                f"数据流 {stream_name} 错误率过高",
                f"当前错误率: {metrics.error_rate:.1%}, 阈值: {monitor.thresholds['max_error_rate']:.1%}"
            )
        
        # 吞吐量告警
        if metrics.throughput_per_second < monitor.thresholds['min_throughput']:
            alert_id = f"low_throughput_{stream_name}"
            await self._send_alert(
                alert_id,
                AlertLevel.WARNING,
                f"数据流 {stream_name} 吞吐量过低",
                f"当前吞吐量: {metrics.throughput_per_second:.1f}/s, 阈值: {monitor.thresholds['min_throughput']}/s"
            )
        
        # 数据过旧告警
        if metrics.freshness_score < 0.5:
            alert_id = f"stale_data_{stream_name}"
            await self._send_alert(
                alert_id,
                AlertLevel.WARNING,
                f"数据流 {stream_name} 数据过旧",
                f"新鲜度评分: {metrics.freshness_score:.2f}"
            )
    
    async def _send_alert(self, alert_id: str, level: AlertLevel, title: str, message: str):
        """发送告警"""
        try:
            await self.alert_manager.create_alert(
                alert_id=alert_id,
                title=title,
                message=message,
                level=level,
                source="stream_monitor",
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'category': 'data_quality'
                }
            )
            
            # 记录告警历史
            self.alert_history.append({
                'alert_id': alert_id,
                'level': level.value,
                'title': title,
                'message': message,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"发送告警失败: {e}")
    
    def get_stream_health(self, stream_name: str) -> Optional[StreamHealthStatus]:
        """获取流健康状态"""
        if stream_name in self.stream_monitors:
            return self.stream_monitors[stream_name].get_health_status()
        return None
    
    def get_all_stream_health(self) -> Dict[str, StreamHealthStatus]:
        """获取所有流的健康状态"""
        return {
            name: monitor.get_health_status()
            for name, monitor in self.stream_monitors.items()
        }
    
    def get_stream_metrics(self, stream_name: str) -> Optional[DataQualityMetrics]:
        """获取流质量指标"""
        if stream_name in self.stream_monitors:
            return self.stream_monitors[stream_name]._calculate_quality_metrics()
        return None
    
    def get_global_stats(self) -> dict:
        """获取全局统计"""
        uptime = datetime.now() - self.global_stats['start_time']
        
        return {
            **self.global_stats,
            'uptime_seconds': uptime.total_seconds(),
            'registered_streams': list(self.stream_monitors.keys()),
            'is_running': self.is_running
        }
    
    def get_recent_alerts(self, limit: int = 50) -> List[dict]:
        """获取最近的告警"""
        recent_alerts = list(self.alert_history)[-limit:]
        return [
            {
                **alert,
                'timestamp': alert['timestamp'].isoformat()
            }
            for alert in recent_alerts
        ]
    
    async def reset_stream_stats(self, stream_name: str = None):
        """重置流统计"""
        if stream_name:
            if stream_name in self.stream_monitors:
                self.stream_monitors[stream_name].reset_stats()
                logger.info(f"已重置流统计: {stream_name}")
        else:
            for monitor in self.stream_monitors.values():
                monitor.reset_stats()
            
            # 重置全局统计
            self.global_stats.update({
                'total_messages': 0,
                'total_errors': 0,
                'start_time': datetime.now()
            })
            
            logger.info("已重置所有流统计")

# 全局实例
_stream_monitor_instance = None

def get_stream_monitor_manager() -> StreamDataMonitorManager:
    """获取流监控管理器实例"""
    global _stream_monitor_instance
    if _stream_monitor_instance is None:
        _stream_monitor_instance = StreamDataMonitorManager()
    return _stream_monitor_instance