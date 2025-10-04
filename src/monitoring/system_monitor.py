"""
System Monitor Module
Comprehensive system monitoring for the trading platform
Tracks performance, resources, errors, and business metrics
"""

import asyncio
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator
from src.utils.database_manager import DatabaseManager

logger = setup_logger(__name__)

class MetricType(Enum):
    """Types of metrics to monitor"""
    SYSTEM = "system"
    TRADING = "trading"
    DATABASE = "database"
    NETWORK = "network"
    APPLICATION = "application"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass

class SystemMetric:
    """System metric data structure"""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }

@dataclass

class SystemAlert:
    """System alert data structure"""
    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'metric_name': self.metric_name,
            'level': self.level.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }

class SystemMonitor:
    """Comprehensive system monitoring"""

    def __init__(self):
        self.db_manager = DatabaseManager()

        # Monitoring configuration
        self.monitoring_interval = config.get('MONITORING_INTERVAL', 60)  # seconds
        self.metrics_retention_days = config.get('METRICS_RETENTION_DAYS', 30)

        # Metric thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 85},
            'memory_usage': {'warning': 80, 'critical': 90},
            'disk_usage': {'warning': 80, 'critical': 90},
            'network_errors': {'warning': 10, 'critical': 50},
            'database_connections': {'warning': 80, 'critical': 95},
            'response_time': {'warning': 1000, 'critical': 5000},  # milliseconds
            'error_rate': {'warning': 5, 'critical': 10},  # percentage
            'daily_pnl': {'warning': -500, 'critical': -1000},  # dollars
            'drawdown': {'warning': 10, 'critical': 20},  # percentage
            'position_count': {'warning': 8, 'critical': 10}
        }

        # Metrics storage
        self.current_metrics: Dict[str, SystemMetric] = {}
        self.metric_history: List[SystemMetric] = []
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []

        # Monitoring state
        self.monitoring_active = False
        self.last_collection_time = None

        # Performance tracking
        self.collection_times = []
        self.collection_errors = 0

        logger.info("System monitor initialized")

    async def start_monitoring(self) -> bool:
        """Start system monitoring"""
        try:
            logger.info("Starting system monitoring")

            # Connect to database
            if not await self.db_manager.connect():
                logger.error("Failed to connect to database for monitoring")
                return False

            self.monitoring_active = True

            # Start monitoring loop
            await self._monitoring_loop()

            return True

        except Exception as e:
            logger.error(f"Failed to start system monitoring: {e}")
            return False

    def stop_monitoring(self):
        """Stop system monitoring"""
        try:
            logger.info("Stopping system monitoring")
            self.monitoring_active = False

        except Exception as e:
            logger.error(f"Error stopping system monitoring: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                collection_start = time.time()

                try:
                    # Collect all metrics
                    await self._collect_metrics()

                    # Check thresholds and generate alerts
                    await self._check_thresholds()

                    # Store metrics in database
                    await self._store_metrics()

                    # Clean up old data
                    await self._cleanup_old_data()

                    collection_time = (time.time() - collection_start) * 1000
                    self.collection_times.append(collection_time)

                    # Keep only recent collection times
                    if len(self.collection_times) > 100:
                        self.collection_times = self.collection_times[-100:]

                    self.last_collection_time = datetime.now()

                    logger.debug(f"Metrics collection completed in {collection_time:.1f}ms")

                except Exception as e:
                    self.collection_errors += 1
                    logger.error(f"Metrics collection error: {e}")

                # Wait for next collection
                await asyncio.sleep(self.monitoring_interval)

        except Exception as e:
            logger.critical(f"Monitoring loop crashed: {e}")
            self.monitoring_active = False

    @timing_decorator

    async def _collect_metrics(self):
        """Collect all system metrics"""
        try:
            # System metrics
            await self._collect_system_metrics()

            # Database metrics
            await self._collect_database_metrics()

            # Application metrics
            await self._collect_application_metrics()

            # Trading metrics
            await self._collect_trading_metrics()

            # Network metrics
            await self._collect_network_metrics()

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")

    async def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            current_time = datetime.now()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            self.current_metrics['cpu_usage'] = SystemMetric(
                name='cpu_usage',
                value=cpu_percent,
                unit='percent',
                metric_type=MetricType.SYSTEM,
                timestamp=current_time,
                tags={'cores': str(cpu_count)}
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            self.current_metrics['memory_usage'] = SystemMetric(
                name='memory_usage',
                value=memory.percent,
                unit='percent',
                metric_type=MetricType.SYSTEM,
                timestamp=current_time,
                tags={'total_gb': str(round(memory.total / (1024**3), 2))}
            )

            self.current_metrics['memory_available'] = SystemMetric(
                name='memory_available',
                value=memory.available / (1024**3),
                unit='gb',
                metric_type=MetricType.SYSTEM,
                timestamp=current_time
            )

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.current_metrics['disk_usage'] = SystemMetric(
                name='disk_usage',
                value=(disk.used / disk.total) * 100,
                unit='percent',
                metric_type=MetricType.SYSTEM,
                timestamp=current_time,
                tags={'total_gb': str(round(disk.total / (1024**3), 2))}
            )

            # Load average (Unix/Linux only)
            try:
                load_avg = psutil.getloadavg()
                self.current_metrics['load_average'] = SystemMetric(
                    name='load_average',
                    value=load_avg[0],
                    unit='load',
                    metric_type=MetricType.SYSTEM,
                    timestamp=current_time,
                    tags={'1min': str(load_avg[0]), '5min': str(load_avg[1]), '15min': str(load_avg[2])}
                )
            except AttributeError:
                # Windows doesn't have load average
                pass

        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

    async def _collect_database_metrics(self):
        """Collect database metrics"""
        try:
            current_time = datetime.now()

            if not self.db_manager or not self.db_manager.is_connected():
                return

            # Database connection metrics
            connection_info = await self.db_manager.get_connection_info()

            if connection_info:
                self.current_metrics['db_connections_active'] = SystemMetric(
                    name='db_connections_active',
                    value=connection_info.get('active_connections', 0),
                    unit='count',
                    metric_type=MetricType.DATABASE,
                    timestamp=current_time
                )

                self.current_metrics['db_connections_idle'] = SystemMetric(
                    name='db_connections_idle',
                    value=connection_info.get('idle_connections', 0),
                    unit='count',
                    metric_type=MetricType.DATABASE,
                    timestamp=current_time
                )

            # Database size metrics
            try:
                db_stats = await self.db_manager.get_database_stats()
                if db_stats:
                    self.current_metrics['db_size'] = SystemMetric(
                        name='db_size',
                        value=db_stats.get('size_mb', 0),
                        unit='mb',
                        metric_type=MetricType.DATABASE,
                        timestamp=current_time
                    )

                    self.current_metrics['db_table_count'] = SystemMetric(
                        name='db_table_count',
                        value=db_stats.get('table_count', 0),
                        unit='count',
                        metric_type=MetricType.DATABASE,
                        timestamp=current_time
                    )
            except Exception as e:
                logger.debug(f"Database stats collection failed: {e}")

        except Exception as e:
            logger.error(f"Database metrics collection failed: {e}")

    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            current_time = datetime.now()

            # Process metrics
            current_process = psutil.Process()

            self.current_metrics['app_memory_usage'] = SystemMetric(
                name='app_memory_usage',
                value=current_process.memory_info().rss / (1024**2),
                unit='mb',
                metric_type=MetricType.APPLICATION,
                timestamp=current_time
            )

            self.current_metrics['app_cpu_usage'] = SystemMetric(
                name='app_cpu_usage',
                value=current_process.cpu_percent(),
                unit='percent',
                metric_type=MetricType.APPLICATION,
                timestamp=current_time
            )

            # File descriptor usage (Unix/Linux)
            try:
                num_fds = current_process.num_fds()
                self.current_metrics['app_file_descriptors'] = SystemMetric(
                    name='app_file_descriptors',
                    value=num_fds,
                    unit='count',
                    metric_type=MetricType.APPLICATION,
                    timestamp=current_time
                )
            except AttributeError:
                # Windows doesn't have file descriptors
                pass

            # Thread count
            num_threads = current_process.num_threads()
            self.current_metrics['app_thread_count'] = SystemMetric(
                name='app_thread_count',
                value=num_threads,
                unit='count',
                metric_type=MetricType.APPLICATION,
                timestamp=current_time
            )

            # Collection performance
            if self.collection_times:
                avg_collection_time = np.mean(self.collection_times)
                self.current_metrics['metrics_collection_time'] = SystemMetric(
                    name='metrics_collection_time',
                    value=avg_collection_time,
                    unit='ms',
                    metric_type=MetricType.APPLICATION,
                    timestamp=current_time
                )

            self.current_metrics['metrics_collection_errors'] = SystemMetric(
                name='metrics_collection_errors',
                value=self.collection_errors,
                unit='count',
                metric_type=MetricType.APPLICATION,
                timestamp=current_time
            )

        except Exception as e:
            logger.error(f"Application metrics collection failed: {e}")

    async def _collect_trading_metrics(self):
        """Collect trading-specific metrics"""
        try:
            current_time = datetime.now()

            # Try to get trading metrics from database
            try:
                # Get recent portfolio performance
                portfolio_stats = await self.db_manager.get_portfolio_stats()

                if portfolio_stats:
                    self.current_metrics['portfolio_value'] = SystemMetric(
                        name='portfolio_value',
                        value=portfolio_stats.get('total_value', 0),
                        unit='usd',
                        metric_type=MetricType.TRADING,
                        timestamp=current_time
                    )

                    self.current_metrics['daily_pnl'] = SystemMetric(
                        name='daily_pnl',
                        value=portfolio_stats.get('daily_pnl', 0),
                        unit='usd',
                        metric_type=MetricType.TRADING,
                        timestamp=current_time
                    )

                    self.current_metrics['total_pnl'] = SystemMetric(
                        name='total_pnl',
                        value=portfolio_stats.get('total_pnl', 0),
                        unit='usd',
                        metric_type=MetricType.TRADING,
                        timestamp=current_time
                    )

                    self.current_metrics['open_positions'] = SystemMetric(
                        name='open_positions',
                        value=portfolio_stats.get('open_positions', 0),
                        unit='count',
                        metric_type=MetricType.TRADING,
                        timestamp=current_time
                    )

                    self.current_metrics['win_rate'] = SystemMetric(
                        name='win_rate',
                        value=portfolio_stats.get('win_rate', 0) * 100,
                        unit='percent',
                        metric_type=MetricType.TRADING,
                        timestamp=current_time
                    )

                # Get recent signals count
                signals_count = await self.db_manager.get_recent_signals_count()
                self.current_metrics['signals_generated_today'] = SystemMetric(
                    name='signals_generated_today',
                    value=signals_count,
                    unit='count',
                    metric_type=MetricType.TRADING,
                    timestamp=current_time
                )

                # Get recent trades count
                trades_count = await self.db_manager.get_recent_trades_count()
                self.current_metrics['trades_executed_today'] = SystemMetric(
                    name='trades_executed_today',
                    value=trades_count,
                    unit='count',
                    metric_type=MetricType.TRADING,
                    timestamp=current_time
                )

            except Exception as e:
                logger.debug(f"Trading metrics collection from DB failed: {e}")

                # Set default values if database query fails
                self.current_metrics['portfolio_value'] = SystemMetric(
                    name='portfolio_value',
                    value=0,
                    unit='usd',
                    metric_type=MetricType.TRADING,
                    timestamp=current_time
                )

        except Exception as e:
            logger.error(f"Trading metrics collection failed: {e}")

    async def _collect_network_metrics(self):
        """Collect network metrics"""
        try:
            current_time = datetime.now()

            # Network I/O
            network_io = psutil.net_io_counters()

            if hasattr(self, '_last_network_io'):
                # Calculate rates
                time_diff = (current_time - self._last_network_time).total_seconds()

                if time_diff > 0:
                    bytes_sent_rate = (network_io.bytes_sent - self._last_network_io.bytes_sent) / time_diff
                    bytes_recv_rate = (network_io.bytes_recv - self._last_network_io.bytes_recv) / time_diff

                    self.current_metrics['network_bytes_sent_rate'] = SystemMetric(
                        name='network_bytes_sent_rate',
                        value=bytes_sent_rate / 1024,  # KB/s
                        unit='kb_per_sec',
                        metric_type=MetricType.NETWORK,
                        timestamp=current_time
                    )

                    self.current_metrics['network_bytes_recv_rate'] = SystemMetric(
                        name='network_bytes_recv_rate',
                        value=bytes_recv_rate / 1024,  # KB/s
                        unit='kb_per_sec',
                        metric_type=MetricType.NETWORK,
                        timestamp=current_time
                    )

            # Store for next calculation
            self._last_network_io = network_io
            self._last_network_time = current_time

            # Network errors
            self.current_metrics['network_errors'] = SystemMetric(
                name='network_errors',
                value=network_io.errin + network_io.errout,
                unit='count',
                metric_type=MetricType.NETWORK,
                timestamp=current_time
            )

        except Exception as e:
            logger.error(f"Network metrics collection failed: {e}")

    async def _check_thresholds(self):
        """Check metric thresholds and generate alerts"""
        try:
            current_time = datetime.now()

            for metric_name, metric in self.current_metrics.items():
                if metric_name not in self.thresholds:
                    continue

                thresholds = self.thresholds[metric_name]
                value = metric.value

                # Check critical threshold
                if value > thresholds.get('critical', float('inf')) or value < thresholds.get('critical', float('-inf')):
                    await self._generate_alert(
                        metric_name, AlertLevel.CRITICAL, value, thresholds['critical']
                    )
                # Check warning threshold
                elif value > thresholds.get('warning', float('inf')) or value < thresholds.get('warning', float('-inf')):
                    await self._generate_alert(
                        metric_name, AlertLevel.WARNING, value, thresholds['warning']
                    )
                else:
                    # Resolve existing alerts if value is back to normal
                    await self._resolve_alert(metric_name)

        except Exception as e:
            logger.error(f"Threshold checking failed: {e}")

    async def _generate_alert(self, metric_name: str, level: AlertLevel,
                            value: float, threshold: float):
        """Generate an alert"""
        try:
            alert_key = f"{metric_name}_{level.value}"

            # Check if alert already exists
            if alert_key in self.active_alerts and not self.active_alerts[alert_key].resolved:
                return

            # Create new alert
            message = f"{metric_name.replace('_', ' ').title()} is {level.value}: {value:.2f} (threshold: {threshold:.2f})"

            alert = SystemAlert(
                metric_name=metric_name,
                level=level,
                message=message,
                value=value,
                threshold=threshold,
                timestamp=datetime.now()
            )

            # Store alert
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)

            # Keep only recent alert history
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]

            logger.warning(f"ALERT [{level.value.upper()}]: {message}")

            # Store in database
            try:
                await self.db_manager.store_alert(alert.to_dict())
            except Exception as e:
                logger.debug(f"Failed to store alert in database: {e}")

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")

    async def _resolve_alert(self, metric_name: str):
        """Resolve alerts for a metric"""
        try:
            alerts_to_resolve = [
                key for key in self.active_alerts.keys()
                if key.startswith(metric_name) and not self.active_alerts[key].resolved
            ]

            for alert_key in alerts_to_resolve:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolution_time = datetime.now()

                logger.info(f"RESOLVED: Alert for {metric_name}")

        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")

    async def _store_metrics(self):
        """Store metrics in database"""
        try:
            if not self.db_manager or not self.db_manager.is_connected():
                return

            # Convert metrics to list of dictionaries
            metrics_data = [metric.to_dict() for metric in self.current_metrics.values()]

            # Store in database
            await self.db_manager.store_system_metrics(metrics_data)

            # Also keep in memory for quick access
            self.metric_history.extend(self.current_metrics.values())

            # Keep only recent history in memory
            if len(self.metric_history) > 10000:
                self.metric_history = self.metric_history[-10000:]

        except Exception as e:
            logger.error(f"Metrics storage failed: {e}")

    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        try:
            if not self.db_manager or not self.db_manager.is_connected():
                return

            cutoff_date = datetime.now() - timedelta(days=self.metrics_retention_days)

            # Clean up old metrics from database
            await self.db_manager.cleanup_old_metrics(cutoff_date)

            # Clean up old alerts from memory
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_date
            ]

        except Exception as e:
            logger.debug(f"Cleanup failed: {e}")

    def get_current_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metric values"""
        return {name: metric.to_dict() for name, metric in self.current_metrics.items()}

    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            history = [
                metric.to_dict() for metric in self.metric_history
                if metric.name == metric_name and metric.timestamp > cutoff_time
            ]

            return sorted(history, key=lambda x: x['timestamp'])

        except Exception as e:
            logger.error(f"Metric history retrieval failed: {e}")
            return []

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get current active alerts"""
        return [
            alert.to_dict() for alert in self.active_alerts.values()
            if not alert.resolved
        ]

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            history = [
                alert.to_dict() for alert in self.alert_history
                if alert.timestamp > cutoff_time
            ]

            return sorted(history, key=lambda x: x['timestamp'], reverse=True)

        except Exception as e:
            logger.error(f"Alert history retrieval failed: {e}")
            return []

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        try:
            current_time = datetime.now()

            # Count alerts by level
            alert_counts = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
            for alert in self.active_alerts.values():
                if not alert.resolved:
                    alert_counts[alert.level.value] += 1

            # Determine overall health status
            if alert_counts['critical'] > 0:
                health_status = 'critical'
            elif alert_counts['error'] > 0:
                health_status = 'error'
            elif alert_counts['warning'] > 0:
                health_status = 'warning'
            else:
                health_status = 'healthy'

            # Calculate uptime
            uptime_seconds = 0
            if self.last_collection_time:
                uptime_seconds = (current_time - self.last_collection_time).total_seconds()

            return {
                'status': health_status,
                'timestamp': current_time.isoformat(),
                'monitoring_active': self.monitoring_active,
                'last_collection': self.last_collection_time.isoformat() if self.last_collection_time else None,
                'uptime_seconds': uptime_seconds,
                'alert_counts': alert_counts,
                'metrics_count': len(self.current_metrics),
                'collection_errors': self.collection_errors,
                'avg_collection_time_ms': np.mean(self.collection_times) if self.collection_times else 0
            }

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'system_performance': {},
                'trading_performance': {},
                'application_performance': {}
            }

            # System performance
            if 'cpu_usage' in self.current_metrics:
                summary['system_performance']['cpu_usage'] = self.current_metrics['cpu_usage'].value

            if 'memory_usage' in self.current_metrics:
                summary['system_performance']['memory_usage'] = self.current_metrics['memory_usage'].value

            if 'disk_usage' in self.current_metrics:
                summary['system_performance']['disk_usage'] = self.current_metrics['disk_usage'].value

            # Trading performance
            trading_metrics = ['portfolio_value', 'daily_pnl', 'total_pnl', 'open_positions', 'win_rate']
            for metric in trading_metrics:
                if metric in self.current_metrics:
                    summary['trading_performance'][metric] = self.current_metrics[metric].value

            # Application performance
            app_metrics = ['app_memory_usage', 'app_cpu_usage', 'metrics_collection_time']
            for metric in app_metrics:
                if metric in self.current_metrics:
                    summary['application_performance'][metric] = self.current_metrics[metric].value

            return summary

        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {'error': str(e)}

# Convenience functions

async def create_system_monitor() -> SystemMonitor:
    """Create and start system monitor"""
    monitor = SystemMonitor()
    return monitor

def get_system_health_quick() -> Dict[str, Any]:
    """Quick system health check"""
    try:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

# Usage example
if __name__ == "__main__":

    async def test_system_monitor():
        # Create system monitor
        monitor = SystemMonitor()

        # Test single metric collection
        await monitor._collect_metrics()

        print("Current metrics:")
        for name, metric in monitor.current_metrics.items():
            print(f"  {name}: {metric.value} {metric.unit}")

        # Test threshold checking
        await monitor._check_thresholds()

        active_alerts = monitor.get_active_alerts()
        print(f"\nActive alerts: {len(active_alerts)}")

        for alert in active_alerts:
            print(f"  - {alert['level'].upper()}: {alert['message']}")

        # Test system health
        health = monitor.get_system_health()
        print(f"\nSystem health: {health['status']}")
        print(f"Metrics collected: {health['metrics_count']}")
        print(f"Collection errors: {health['collection_errors']}")

        print("System monitor test completed!")

    # Run test
    asyncio.run(test_system_monitor())
