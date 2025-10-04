"""
I/O Performance Monitor
I/O性能监控系统，监控磁盘、网络和内存I/O性能
提供性能分析、瓶颈识别和优化建议
"""

import asyncio
import time
import psutil
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
class DiskIOMetrics:
    """磁盘I/O指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 读写统计
    read_bytes_per_sec: float = 0.0
    write_bytes_per_sec: float = 0.0
    read_ops_per_sec: float = 0.0
    write_ops_per_sec: float = 0.0
    
    # 延迟统计
    read_latency_ms: float = 0.0
    write_latency_ms: float = 0.0
    
    # 队列和利用率
    io_queue_depth: float = 0.0
    disk_utilization: float = 0.0
    
    # 每个磁盘的详细信息
    disk_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class NetworkIOMetrics:
    """网络I/O指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 网络流量
    bytes_sent_per_sec: float = 0.0
    bytes_recv_per_sec: float = 0.0
    packets_sent_per_sec: float = 0.0
    packets_recv_per_sec: float = 0.0
    
    # 连接统计
    active_connections: int = 0
    connection_errors: int = 0
    
    # 每个接口的详细信息
    interface_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class MemoryIOMetrics:
    """内存I/O指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 内存使用
    memory_usage_percent: float = 0.0
    available_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    
    # 缓存和缓冲
    cached_memory_mb: float = 0.0
    buffer_memory_mb: float = 0.0
    
    # 交换空间
    swap_usage_percent: float = 0.0
    swap_used_mb: float = 0.0
    
    # 页面错误
    page_faults_per_sec: float = 0.0

@dataclass
class IOPerformanceProfile:
    """I/O性能画像"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 整体评分
    overall_score: int = 100  # 0-100
    
    # 各子系统评分
    disk_score: int = 100
    network_score: int = 100
    memory_score: int = 100
    
    # 瓶颈识别
    bottlenecks: List[str] = field(default_factory=list)
    
    # 性能等级
    performance_grade: str = "Excellent"  # Excellent, Good, Fair, Poor
    
    # 优化建议
    recommendations: List[str] = field(default_factory=list)

class DiskIOMonitor:
    """磁盘I/O监控器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.last_disk_io = None
        
        # 性能阈值
        self.thresholds = {
            'read_latency_ms': 20,
            'write_latency_ms': 20,
            'utilization_percent': 80,
            'queue_depth': 10
        }
    
    def collect_metrics(self) -> DiskIOMetrics:
        """收集磁盘I/O指标"""
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            
            metrics = DiskIOMetrics()
            
            if self.last_disk_io and hasattr(self, 'last_collect_time'):
                time_delta = current_time - self.last_collect_time
                
                if time_delta > 0:
                    # 计算速率
                    metrics.read_bytes_per_sec = (current_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
                    metrics.write_bytes_per_sec = (current_io.write_bytes - self.last_disk_io.write_bytes) / time_delta
                    metrics.read_ops_per_sec = (current_io.read_count - self.last_disk_io.read_count) / time_delta
                    metrics.write_ops_per_sec = (current_io.write_count - self.last_disk_io.write_count) / time_delta
                    
                    # 计算延迟（简化估算）
                    if metrics.read_ops_per_sec > 0:
                        metrics.read_latency_ms = (current_io.read_time - self.last_disk_io.read_time) / metrics.read_ops_per_sec
                    
                    if metrics.write_ops_per_sec > 0:
                        metrics.write_latency_ms = (current_io.write_time - self.last_disk_io.write_time) / metrics.write_ops_per_sec
            
            # 收集每个磁盘的统计
            try:
                disk_usage = psutil.disk_usage('/')
                metrics.disk_utilization = (disk_usage.used / disk_usage.total) * 100
                
                # 尝试获取每个分区的I/O统计
                for partition in psutil.disk_partitions():
                    try:
                        partition_usage = psutil.disk_usage(partition.mountpoint)
                        metrics.disk_stats[partition.device] = {
                            'mountpoint': partition.mountpoint,
                            'fstype': partition.fstype,
                            'usage_percent': (partition_usage.used / partition_usage.total) * 100,
                            'free_gb': partition_usage.free / (1024**3),
                            'used_gb': partition_usage.used / (1024**3)
                        }
                    except (OSError, PermissionError):
                        continue
                        
            except Exception as e:
                logger.debug(f"收集磁盘使用率失败: {e}")
            
            # 保存当前状态
            self.last_disk_io = current_io
            self.last_collect_time = current_time
            
            # 添加到历史
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集磁盘I/O指标失败: {e}")
            return DiskIOMetrics()
    
    def analyze_performance(self) -> dict:
        """分析磁盘性能"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # 最近10个指标
        
        issues = []
        score = 100
        
        # 分析延迟
        avg_read_latency = statistics.mean([m.read_latency_ms for m in recent_metrics if m.read_latency_ms > 0])
        avg_write_latency = statistics.mean([m.write_latency_ms for m in recent_metrics if m.write_latency_ms > 0])
        
        if avg_read_latency > self.thresholds['read_latency_ms']:
            issues.append(f"读取延迟过高: {avg_read_latency:.1f}ms")
            score -= 20
        
        if avg_write_latency > self.thresholds['write_latency_ms']:
            issues.append(f"写入延迟过高: {avg_write_latency:.1f}ms")
            score -= 20
        
        # 分析利用率
        avg_utilization = statistics.mean([m.disk_utilization for m in recent_metrics])
        if avg_utilization > self.thresholds['utilization_percent']:
            issues.append(f"磁盘使用率过高: {avg_utilization:.1f}%")
            score -= 15
        
        return {
            'score': max(0, score),
            'issues': issues,
            'avg_read_latency': avg_read_latency,
            'avg_write_latency': avg_write_latency,
            'avg_utilization': avg_utilization
        }

class NetworkIOMonitor:
    """网络I/O监控器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.last_net_io = None
        
        # 性能阈值
        self.thresholds = {
            'bandwidth_utilization_percent': 80,
            'packet_loss_rate': 0.01,  # 1%
            'connection_errors_per_min': 10
        }
    
    def collect_metrics(self) -> NetworkIOMetrics:
        """收集网络I/O指标"""
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            metrics = NetworkIOMetrics()
            
            if self.last_net_io and hasattr(self, 'last_collect_time'):
                time_delta = current_time - self.last_collect_time
                
                if time_delta > 0:
                    # 计算速率
                    metrics.bytes_sent_per_sec = (current_io.bytes_sent - self.last_net_io.bytes_sent) / time_delta
                    metrics.bytes_recv_per_sec = (current_io.bytes_recv - self.last_net_io.bytes_recv) / time_delta
                    metrics.packets_sent_per_sec = (current_io.packets_sent - self.last_net_io.packets_sent) / time_delta
                    metrics.packets_recv_per_sec = (current_io.packets_recv - self.last_net_io.packets_recv) / time_delta
            
            # 收集连接统计
            try:
                connections = psutil.net_connections()
                metrics.active_connections = len([c for c in connections if c.status == psutil.CONN_ESTABLISHED])
            except (AccessDenied, PermissionError):
                metrics.active_connections = 0
            
            # 收集每个接口的统计
            try:
                for interface, stats in psutil.net_io_counters(pernic=True).items():
                    metrics.interface_stats[interface] = {
                        'bytes_sent': stats.bytes_sent,
                        'bytes_recv': stats.bytes_recv,
                        'packets_sent': stats.packets_sent,
                        'packets_recv': stats.packets_recv,
                        'errors_in': stats.errin,
                        'errors_out': stats.errout,
                        'drops_in': stats.dropin,
                        'drops_out': stats.dropout
                    }
            except Exception as e:
                logger.debug(f"收集接口统计失败: {e}")
            
            # 保存当前状态
            self.last_net_io = current_io
            self.last_collect_time = current_time
            
            # 添加到历史
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集网络I/O指标失败: {e}")
            return NetworkIOMetrics()
    
    def analyze_performance(self) -> dict:
        """分析网络性能"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        issues = []
        score = 100
        
        # 分析带宽使用
        avg_bytes_sent = statistics.mean([m.bytes_sent_per_sec for m in recent_metrics])
        avg_bytes_recv = statistics.mean([m.bytes_recv_per_sec for m in recent_metrics])
        
        # 简化的带宽利用率检查（假设千兆网络）
        max_bandwidth = 1000 * 1024 * 1024  # 1 Gbps in bytes
        utilization = ((avg_bytes_sent + avg_bytes_recv) / max_bandwidth) * 100
        
        if utilization > self.thresholds['bandwidth_utilization_percent']:
            issues.append(f"网络带宽使用率过高: {utilization:.1f}%")
            score -= 25
        
        # 分析连接数
        avg_connections = statistics.mean([m.active_connections for m in recent_metrics])
        if avg_connections > 1000:  # 简单阈值
            issues.append(f"活跃连接数过多: {avg_connections:.0f}")
            score -= 15
        
        return {
            'score': max(0, score),
            'issues': issues,
            'avg_bandwidth_utilization': utilization,
            'avg_connections': avg_connections,
            'avg_bytes_sent_per_sec': avg_bytes_sent,
            'avg_bytes_recv_per_sec': avg_bytes_recv
        }

class MemoryIOMonitor:
    """内存I/O监控器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        
        # 性能阈值
        self.thresholds = {
            'memory_usage_percent': 80,
            'swap_usage_percent': 10,
            'page_faults_per_sec': 1000
        }
    
    def collect_metrics(self) -> MemoryIOMetrics:
        """收集内存I/O指标"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics = MemoryIOMetrics(
                memory_usage_percent=memory.percent,
                available_memory_mb=memory.available / (1024 * 1024),
                used_memory_mb=memory.used / (1024 * 1024),
                swap_usage_percent=swap.percent,
                swap_used_mb=swap.used / (1024 * 1024)
            )
            
            # 尝试获取缓存信息（Linux特有）
            try:
                if hasattr(memory, 'cached'):
                    metrics.cached_memory_mb = memory.cached / (1024 * 1024)
                if hasattr(memory, 'buffers'):
                    metrics.buffer_memory_mb = memory.buffers / (1024 * 1024)
            except AttributeError:
                pass
            
            # 添加到历史
            self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集内存I/O指标失败: {e}")
            return MemoryIOMetrics()
    
    def analyze_performance(self) -> dict:
        """分析内存性能"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        issues = []
        score = 100
        
        # 分析内存使用率
        avg_memory_usage = statistics.mean([m.memory_usage_percent for m in recent_metrics])
        if avg_memory_usage > self.thresholds['memory_usage_percent']:
            issues.append(f"内存使用率过高: {avg_memory_usage:.1f}%")
            score -= 25
        
        # 分析交换空间使用
        avg_swap_usage = statistics.mean([m.swap_usage_percent for m in recent_metrics])
        if avg_swap_usage > self.thresholds['swap_usage_percent']:
            issues.append(f"交换空间使用率过高: {avg_swap_usage:.1f}%")
            score -= 30
        
        return {
            'score': max(0, score),
            'issues': issues,
            'avg_memory_usage': avg_memory_usage,
            'avg_swap_usage': avg_swap_usage,
            'avg_available_memory_mb': statistics.mean([m.available_memory_mb for m in recent_metrics])
        }

class IOPerformanceMonitor:
    """I/O性能监控器主类"""
    
    def __init__(self, system_monitor: SystemMonitor = None, alert_manager: AlertManager = None):
        self.system_monitor = system_monitor
        self.alert_manager = alert_manager
        
        # 子监控器
        self.disk_monitor = DiskIOMonitor()
        self.network_monitor = NetworkIOMonitor()
        self.memory_monitor = MemoryIOMonitor()
        
        # 性能历史
        self.performance_profiles = deque(maxlen=1000)
        
        # 统计信息
        self.stats = {
            'monitoring_duration_hours': 0,
            'total_alerts_sent': 0,
            'performance_degradations': 0,
            'start_time': datetime.now()
        }
        
        # 监控任务
        self.monitoring_task = None
        self.analysis_task = None
        self.is_running = False
        
        logger.info("I/O性能监控器初始化完成")
    
    async def start(self):
        """启动I/O监控器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("I/O性能监控器启动完成")
    
    async def stop(self):
        """停止I/O监控器"""
        if not self.is_running:
            return
        
        logger.info("正在停止I/O性能监控器...")
        self.is_running = False
        
        # 取消监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.analysis_task:
            self.analysis_task.cancel()
        
        logger.info("I/O性能监控器已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(5)  # 5秒采集间隔
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _collect_all_metrics(self):
        """收集所有I/O指标"""
        try:
            # 并发收集所有指标
            disk_metrics = self.disk_monitor.collect_metrics()
            network_metrics = self.network_monitor.collect_metrics()
            memory_metrics = self.memory_monitor.collect_metrics()
            
            # 发送到系统监控
            if self.system_monitor:
                await self._send_metrics_to_system_monitor(disk_metrics, network_metrics, memory_metrics)
                
        except Exception as e:
            logger.error(f"收集I/O指标失败: {e}")
    
    async def _send_metrics_to_system_monitor(self, disk_metrics: DiskIOMetrics, 
                                           network_metrics: NetworkIOMetrics,
                                           memory_metrics: MemoryIOMetrics):
        """发送指标到系统监控"""
        try:
            # 磁盘指标
            await self.system_monitor.record_metric(
                'disk_read_bytes_per_sec', disk_metrics.read_bytes_per_sec, {'category': 'io'}
            )
            await self.system_monitor.record_metric(
                'disk_write_bytes_per_sec', disk_metrics.write_bytes_per_sec, {'category': 'io'}
            )
            await self.system_monitor.record_metric(
                'disk_utilization', disk_metrics.disk_utilization, {'category': 'io', 'unit': 'percent'}
            )
            
            # 网络指标
            await self.system_monitor.record_metric(
                'network_bytes_sent_per_sec', network_metrics.bytes_sent_per_sec, {'category': 'io'}
            )
            await self.system_monitor.record_metric(
                'network_bytes_recv_per_sec', network_metrics.bytes_recv_per_sec, {'category': 'io'}
            )
            await self.system_monitor.record_metric(
                'network_active_connections', network_metrics.active_connections, {'category': 'io'}
            )
            
            # 内存指标
            await self.system_monitor.record_metric(
                'memory_usage_percent', memory_metrics.memory_usage_percent, {'category': 'io', 'unit': 'percent'}
            )
            await self.system_monitor.record_metric(
                'swap_usage_percent', memory_metrics.swap_usage_percent, {'category': 'io', 'unit': 'percent'}
            )
            
        except Exception as e:
            logger.error(f"发送I/O指标到系统监控失败: {e}")
    
    async def _analysis_loop(self):
        """分析循环"""
        while self.is_running:
            try:
                await self._analyze_performance()
                await asyncio.sleep(60)  # 每分钟分析一次
            except Exception as e:
                logger.error(f"分析循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_performance(self):
        """分析I/O性能"""
        try:
            # 分析各子系统性能
            disk_analysis = self.disk_monitor.analyze_performance()
            network_analysis = self.network_monitor.analyze_performance()
            memory_analysis = self.memory_monitor.analyze_performance()
            
            # 生成性能画像
            profile = self._generate_performance_profile(disk_analysis, network_analysis, memory_analysis)
            self.performance_profiles.append(profile)
            
            # 检查告警
            await self._check_performance_alerts(profile)
            
            # 更新统计
            self.stats['monitoring_duration_hours'] = (
                datetime.now() - self.stats['start_time']
            ).total_seconds() / 3600
            
            logger.debug(f"I/O性能分析完成 - 整体评分: {profile.overall_score}, 等级: {profile.performance_grade}")
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
    
    def _generate_performance_profile(self, disk_analysis: dict, 
                                    network_analysis: dict, 
                                    memory_analysis: dict) -> IOPerformanceProfile:
        """生成I/O性能画像"""
        profile = IOPerformanceProfile()
        
        # 各子系统评分
        profile.disk_score = disk_analysis.get('score', 100)
        profile.network_score = network_analysis.get('score', 100)
        profile.memory_score = memory_analysis.get('score', 100)
        
        # 计算整体评分
        profile.overall_score = int((profile.disk_score + profile.network_score + profile.memory_score) / 3)
        
        # 收集瓶颈
        profile.bottlenecks = []
        profile.bottlenecks.extend(disk_analysis.get('issues', []))
        profile.bottlenecks.extend(network_analysis.get('issues', []))
        profile.bottlenecks.extend(memory_analysis.get('issues', []))
        
        # 确定性能等级
        if profile.overall_score >= 90:
            profile.performance_grade = "Excellent"
        elif profile.overall_score >= 75:
            profile.performance_grade = "Good"
        elif profile.overall_score >= 60:
            profile.performance_grade = "Fair"
        else:
            profile.performance_grade = "Poor"
        
        # 生成优化建议
        profile.recommendations = self._generate_recommendations(profile)
        
        return profile
    
    def _generate_recommendations(self, profile: IOPerformanceProfile) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if profile.disk_score < 80:
            recommendations.append("考虑升级到更快的存储设备(SSD)")
            recommendations.append("优化数据库查询以减少磁盘I/O")
            recommendations.append("实现数据缓存策略")
        
        if profile.network_score < 80:
            recommendations.append("检查网络带宽和延迟")
            recommendations.append("优化数据传输协议")
            recommendations.append("考虑数据压缩")
        
        if profile.memory_score < 80:
            recommendations.append("增加系统内存")
            recommendations.append("优化内存使用模式")
            recommendations.append("减少内存泄漏")
        
        if profile.overall_score >= 90:
            recommendations.append("I/O性能表现优秀，维持当前配置")
        
        return recommendations
    
    async def _check_performance_alerts(self, profile: IOPerformanceProfile):
        """检查性能告警"""
        if not self.alert_manager:
            return
        
        # 整体性能告警
        if profile.overall_score < 60:
            await self._send_alert(
                'io_performance_poor',
                f"I/O性能严重下降 (评分: {profile.overall_score})",
                AlertLevel.CRITICAL
            )
            self.stats['performance_degradations'] += 1
        elif profile.overall_score < 75:
            await self._send_alert(
                'io_performance_degraded',
                f"I/O性能下降 (评分: {profile.overall_score})",
                AlertLevel.WARNING
            )
        
        # 特定瓶颈告警
        for bottleneck in profile.bottlenecks:
            if "延迟过高" in bottleneck:
                await self._send_alert(
                    'io_high_latency',
                    bottleneck,
                    AlertLevel.WARNING
                )
            elif "使用率过高" in bottleneck:
                await self._send_alert(
                    'io_high_utilization',
                    bottleneck,
                    AlertLevel.WARNING
                )
    
    async def _send_alert(self, alert_id: str, message: str, level: AlertLevel):
        """发送告警"""
        try:
            await self.alert_manager.create_alert(
                alert_id=alert_id,
                title="I/O性能告警",
                message=message,
                level=level,
                source="io_monitor",
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'category': 'io_performance'
                }
            )
            self.stats['total_alerts_sent'] += 1
            
        except Exception as e:
            logger.error(f"发送I/O性能告警失败: {e}")
    
    def get_current_metrics(self) -> dict:
        """获取当前I/O指标"""
        disk_metrics = self.disk_monitor.collect_metrics()
        network_metrics = self.network_monitor.collect_metrics()
        memory_metrics = self.memory_monitor.collect_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'disk': {
                'read_bytes_per_sec': disk_metrics.read_bytes_per_sec,
                'write_bytes_per_sec': disk_metrics.write_bytes_per_sec,
                'read_ops_per_sec': disk_metrics.read_ops_per_sec,
                'write_ops_per_sec': disk_metrics.write_ops_per_sec,
                'utilization': disk_metrics.disk_utilization
            },
            'network': {
                'bytes_sent_per_sec': network_metrics.bytes_sent_per_sec,
                'bytes_recv_per_sec': network_metrics.bytes_recv_per_sec,
                'active_connections': network_metrics.active_connections
            },
            'memory': {
                'usage_percent': memory_metrics.memory_usage_percent,
                'available_mb': memory_metrics.available_memory_mb,
                'swap_usage_percent': memory_metrics.swap_usage_percent
            }
        }
    
    def get_performance_profile(self) -> Optional[IOPerformanceProfile]:
        """获取最新性能画像"""
        return self.performance_profiles[-1] if self.performance_profiles else None
    
    def get_stats(self) -> dict:
        """获取监控器统计信息"""
        current_profile = self.get_performance_profile()
        
        return {
            'is_running': self.is_running,
            'stats': self.stats,
            'current_profile': {
                'overall_score': current_profile.overall_score if current_profile else 0,
                'performance_grade': current_profile.performance_grade if current_profile else "Unknown",
                'bottleneck_count': len(current_profile.bottlenecks) if current_profile else 0
            } if current_profile else {},
            'monitoring_counts': {
                'disk_metrics': len(self.disk_monitor.metrics_history),
                'network_metrics': len(self.network_monitor.metrics_history),
                'memory_metrics': len(self.memory_monitor.metrics_history),
                'performance_profiles': len(self.performance_profiles)
            }
        }
    
    def get_recent_profiles(self, limit: int = 10) -> List[dict]:
        """获取最近的性能画像"""
        recent_profiles = list(self.performance_profiles)[-limit:]
        return [
            {
                'timestamp': profile.timestamp.isoformat(),
                'overall_score': profile.overall_score,
                'disk_score': profile.disk_score,
                'network_score': profile.network_score,
                'memory_score': profile.memory_score,
                'performance_grade': profile.performance_grade,
                'bottlenecks': profile.bottlenecks,
                'recommendations': profile.recommendations
            }
            for profile in recent_profiles
        ]

# 全局实例
_io_monitor_instance = None

def get_io_performance_monitor() -> IOPerformanceMonitor:
    """获取I/O性能监控器实例"""
    global _io_monitor_instance
    if _io_monitor_instance is None:
        _io_monitor_instance = IOPerformanceMonitor()
    return _io_monitor_instance