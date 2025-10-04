"""
Health Checker Module
System health monitoring and diagnostic tools
Provides comprehensive health checks and automated recovery actions
"""

import asyncio
import psutil
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import time
import socket
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator, retry_decorator
from src.utils.database_manager import DatabaseManager

logger = setup_logger(__name__)

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """System component types"""
    DATABASE = "database"
    EXCHANGE = "exchange"
    EXTERNAL_API = "external_api"
    SYSTEM_RESOURCE = "system_resource"
    APPLICATION = "application"
    NETWORK = "network"

@dataclass

class HealthCheck:
    """Health check definition"""
    name: str
    component_type: ComponentType
    check_function: Callable
    interval_seconds: int = 60
    timeout_seconds: int = 30
    retry_count: int = 3
    enabled: bool = True

    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None

    # Recovery actions
    recovery_actions: List[Callable] = field(default_factory=list)
    auto_recovery: bool = False

    # Metadata
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass

class HealthResult:
    """Health check result"""
    check_name: str
    status: HealthStatus
    value: Optional[float]
    message: str
    timestamp: datetime
    duration_ms: float

    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'value': self.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'details': self.details,
            'error': self.error
        }

class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        self.db_manager = DatabaseManager()

        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}

        # Results storage
        self.latest_results: Dict[str, HealthResult] = {}
        self.results_history: List[HealthResult] = []

        # Health checking state
        self.checking_active = False
        self.last_check_time = None

        # Statistics
        self.check_stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'avg_check_duration_ms': 0
        }

        # Recovery tracking
        self.recovery_attempts = {}
        self.last_recovery_times = {}

        logger.info("Health checker initialized")
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks"""
        try:
            # System resource checks
            self.register_health_check(HealthCheck(
                name='cpu_usage',
                component_type=ComponentType.SYSTEM_RESOURCE,
                check_function=self._check_cpu_usage,
                interval_seconds=30,
                warning_threshold=80.0,
                critical_threshold=95.0,
                description='Monitor CPU usage percentage'
            ))

            self.register_health_check(HealthCheck(
                name='memory_usage',
                component_type=ComponentType.SYSTEM_RESOURCE,
                check_function=self._check_memory_usage,
                interval_seconds=30,
                warning_threshold=85.0,
                critical_threshold=95.0,
                description='Monitor memory usage percentage'
            ))

            self.register_health_check(HealthCheck(
                name='disk_usage',
                component_type=ComponentType.SYSTEM_RESOURCE,
                check_function=self._check_disk_usage,
                interval_seconds=300,  # 5 minutes
                warning_threshold=80.0,
                critical_threshold=90.0,
                description='Monitor disk usage percentage'
            ))

            # Database checks
            self.register_health_check(HealthCheck(
                name='database_connection',
                component_type=ComponentType.DATABASE,
                check_function=self._check_database_connection,
                interval_seconds=60,
                timeout_seconds=10,
                auto_recovery=True,
                recovery_actions=[self._recover_database_connection],
                description='Check database connectivity'
            ))

            self.register_health_check(HealthCheck(
                name='database_response_time',
                component_type=ComponentType.DATABASE,
                check_function=self._check_database_response_time,
                interval_seconds=120,
                warning_threshold=1000,  # 1 second
                critical_threshold=5000,  # 5 seconds
                description='Monitor database response time'
            ))

            # Application checks
            self.register_health_check(HealthCheck(
                name='application_memory',
                component_type=ComponentType.APPLICATION,
                check_function=self._check_application_memory,
                interval_seconds=60,
                warning_threshold=1000,  # 1GB
                critical_threshold=2000,  # 2GB
                description='Monitor application memory usage'
            ))

            # Network checks
            self.register_health_check(HealthCheck(
                name='internet_connectivity',
                component_type=ComponentType.NETWORK,
                check_function=self._check_internet_connectivity,
                interval_seconds=300,  # 5 minutes
                timeout_seconds=10,
                description='Check internet connectivity'
            ))

            logger.info(f"Registered {len(self.health_checks)} default health checks")

        except Exception as e:
            logger.error(f"Failed to register default health checks: {e}")

    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        try:
            self.health_checks[health_check.name] = health_check
            logger.debug(f"Registered health check: {health_check.name}")

        except Exception as e:
            logger.error(f"Failed to register health check {health_check.name}: {e}")

    def unregister_health_check(self, check_name: str) -> bool:
        """Unregister a health check"""
        try:
            if check_name in self.health_checks:
                del self.health_checks[check_name]
                logger.debug(f"Unregistered health check: {check_name}")
                return True
            else:
                logger.warning(f"Health check not found: {check_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to unregister health check {check_name}: {e}")
            return False

    def enable_health_check(self, check_name: str) -> bool:
        """Enable a health check"""
        try:
            if check_name in self.health_checks:
                self.health_checks[check_name].enabled = True
                logger.debug(f"Enabled health check: {check_name}")
                return True
            else:
                logger.warning(f"Health check not found: {check_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to enable health check {check_name}: {e}")
            return False

    def disable_health_check(self, check_name: str) -> bool:
        """Disable a health check"""
        try:
            if check_name in self.health_checks:
                self.health_checks[check_name].enabled = False
                logger.debug(f"Disabled health check: {check_name}")
                return True
            else:
                logger.warning(f"Health check not found: {check_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to disable health check {check_name}: {e}")
            return False

    async def start_health_monitoring(self):
        """Start continuous health monitoring"""
        try:
            logger.info("Starting health monitoring")
            self.checking_active = True

            # Initialize database connection
            if not await self.db_manager.connect():
                logger.warning("Database connection failed for health monitoring")

            await self._health_monitoring_loop()

        except Exception as e:
            logger.error(f"Health monitoring failed to start: {e}")

    def stop_health_monitoring(self):
        """Stop health monitoring"""
        try:
            logger.info("Stopping health monitoring")
            self.checking_active = False

        except Exception as e:
            logger.error(f"Failed to stop health monitoring: {e}")

    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        try:
            check_schedules = {name: datetime.now() for name in self.health_checks.keys()}

            while self.checking_active:
                current_time = datetime.now()

                # Check which health checks need to run
                checks_to_run = []

                for check_name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue

                    last_run = check_schedules.get(check_name, datetime.min)
                    time_since_last = (current_time - last_run).total_seconds()

                    if time_since_last >= health_check.interval_seconds:
                        checks_to_run.append(check_name)
                        check_schedules[check_name] = current_time

                # Run scheduled health checks
                if checks_to_run:
                    await self._run_health_checks(checks_to_run)
                    self.last_check_time = current_time

                # Sleep for a short interval
                await asyncio.sleep(5)

        except Exception as e:
            logger.critical(f"Health monitoring loop crashed: {e}")
            self.checking_active = False

    async def _run_health_checks(self, check_names: List[str]):
        """Run specified health checks"""
        try:
            tasks = []

            for check_name in check_names:
                if check_name in self.health_checks:
                    task = self._run_single_health_check(check_name)
                    tasks.append(task)

            # Run checks concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Health checks execution failed: {e}")

    async def _run_single_health_check(self, check_name: str):
        """Run a single health check"""
        try:
            health_check = self.health_checks[check_name]
            start_time = time.time()

            # Run the check with timeout
            try:
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout_seconds
                )

                if isinstance(result, tuple):
                    status, value, message, details = result
                else:
                    # Handle simple boolean result
                    status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                    value = None
                    message = f"{check_name} check {'passed' if result else 'failed'}"
                    details = {}

                error_msg = None

            except asyncio.TimeoutError:
                status = HealthStatus.CRITICAL
                value = None
                message = f"{check_name} check timed out"
                details = {}
                error_msg = f"Check timed out after {health_check.timeout_seconds}s"

            except Exception as e:
                status = HealthStatus.CRITICAL
                value = None
                message = f"{check_name} check failed: {str(e)}"
                details = {}
                error_msg = str(e)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Evaluate thresholds
            if value is not None and status == HealthStatus.HEALTHY:
                if (health_check.critical_threshold is not None and
                    value >= health_check.critical_threshold):
                    status = HealthStatus.CRITICAL
                    message = f"{check_name}: {value} exceeds critical threshold {health_check.critical_threshold}"
                elif (health_check.warning_threshold is not None and
                      value >= health_check.warning_threshold):
                    status = HealthStatus.WARNING
                    message = f"{check_name}: {value} exceeds warning threshold {health_check.warning_threshold}"

            # Create result
            result = HealthResult(
                check_name=check_name,
                status=status,
                value=value,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details=details,
                error=error_msg
            )

            # Store result
            self.latest_results[check_name] = result
            self.results_history.append(result)

            # Keep only recent history
            if len(self.results_history) > 10000:
                self.results_history = self.results_history[-10000:]

            # Update statistics
            self.check_stats['total_checks'] += 1
            if status == HealthStatus.HEALTHY:
                self.check_stats['passed_checks'] += 1
            else:
                self.check_stats['failed_checks'] += 1

            # Update average duration
            total_checks = self.check_stats['total_checks']
            current_avg = self.check_stats['avg_check_duration_ms']
            self.check_stats['avg_check_duration_ms'] = (
                (current_avg * (total_checks - 1) + duration_ms) / total_checks
            )

            # Store in database
            try:
                if self.db_manager and self.db_manager.is_connected():
                    await self.db_manager.store_health_check_result(result.to_dict())
            except Exception as e:
                logger.debug(f"Failed to store health check result in database: {e}")

            # Log result
            if status == HealthStatus.CRITICAL:
                logger.error(f"HEALTH CHECK CRITICAL: {message}")
            elif status == HealthStatus.WARNING:
                logger.warning(f"HEALTH CHECK WARNING: {message}")
            else:
                logger.debug(f"Health check passed: {check_name}")

            # Attempt recovery if needed
            if status in [HealthStatus.CRITICAL, HealthStatus.WARNING] and health_check.auto_recovery:
                await self._attempt_recovery(check_name, health_check)

        except Exception as e:
            logger.error(f"Health check execution failed for {check_name}: {e}")

    async def _attempt_recovery(self, check_name: str, health_check: HealthCheck):
        """Attempt automatic recovery for a failed health check"""
        try:
            current_time = datetime.now()

            # Check if we've attempted recovery recently
            last_recovery = self.last_recovery_times.get(check_name)
            if last_recovery and (current_time - last_recovery).total_seconds() < 300:  # 5 minutes
                logger.debug(f"Skipping recovery for {check_name} - too recent")
                return

            # Track recovery attempts
            if check_name not in self.recovery_attempts:
                self.recovery_attempts[check_name] = 0

            self.recovery_attempts[check_name] += 1
            self.last_recovery_times[check_name] = current_time

            logger.info(f"Attempting recovery for {check_name} (attempt {self.recovery_attempts[check_name]})")

            # Run recovery actions
            for recovery_action in health_check.recovery_actions:
                try:
                    await recovery_action()
                    logger.info(f"Recovery action executed for {check_name}")

                    # Wait a bit and re-run the health check
                    await asyncio.sleep(5)
                    await self._run_single_health_check(check_name)

                    # Check if recovery was successful
                    latest_result = self.latest_results.get(check_name)
                    if latest_result and latest_result.status == HealthStatus.HEALTHY:
                        logger.info(f"Recovery successful for {check_name}")
                        self.recovery_attempts[check_name] = 0
                        return

                except Exception as e:
                    logger.error(f"Recovery action failed for {check_name}: {e}")

            logger.warning(f"Recovery failed for {check_name}")

        except Exception as e:
            logger.error(f"Recovery attempt failed for {check_name}: {e}")

    # Health check implementations

    async def _check_cpu_usage(self) -> Tuple[HealthStatus, float, str, Dict[str, Any]]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)

            status = HealthStatus.HEALTHY
            message = f"CPU usage: {cpu_percent:.1f}%"

            details = {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': dict(psutil.cpu_freq()._asdict()) if psutil.cpu_freq() else None
            }

            return status, cpu_percent, message, details

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"CPU check failed: {e}", {}

    async def _check_memory_usage(self) -> Tuple[HealthStatus, float, str, Dict[str, Any]]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()

            status = HealthStatus.HEALTHY
            message = f"Memory usage: {memory.percent:.1f}%"

            details = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2)
            }

            return status, memory.percent, message, details

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"Memory check failed: {e}", {}

    async def _check_disk_usage(self) -> Tuple[HealthStatus, float, str, Dict[str, Any]]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100

            status = HealthStatus.HEALTHY
            message = f"Disk usage: {usage_percent:.1f}%"

            details = {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2)
            }

            return status, usage_percent, message, details

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"Disk check failed: {e}", {}

    async def _check_database_connection(self) -> Tuple[HealthStatus, Optional[float], str, Dict[str, Any]]:
        """Check database connectivity"""
        try:
            if not self.db_manager:
                return HealthStatus.CRITICAL, None, "Database manager not initialized", {}

            start_time = time.time()
            is_connected = self.db_manager.is_connected()

            if not is_connected:
                # Try to reconnect
                connected = await self.db_manager.connect()
                if not connected:
                    return HealthStatus.CRITICAL, None, "Database connection failed", {}

            # Test with a simple query
            connection_info = await self.db_manager.get_connection_info()
            connection_time_ms = (time.time() - start_time) * 1000

            status = HealthStatus.HEALTHY
            message = f"Database connected (response time: {connection_time_ms:.1f}ms)"

            details = connection_info or {}

            return status, connection_time_ms, message, details

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"Database check failed: {e}", {}

    async def _check_database_response_time(self) -> Tuple[HealthStatus, float, str, Dict[str, Any]]:
        """Check database response time"""
        try:
            if not self.db_manager or not self.db_manager.is_connected():
                return HealthStatus.CRITICAL, None, "Database not connected", {}

            start_time = time.time()

            # Run a simple query
            await self.db_manager.execute_query("SELECT 1")

            response_time_ms = (time.time() - start_time) * 1000

            status = HealthStatus.HEALTHY
            message = f"Database response time: {response_time_ms:.1f}ms"

            return status, response_time_ms, message, {}

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"Database response check failed: {e}", {}

    async def _check_application_memory(self) -> Tuple[HealthStatus, float, str, Dict[str, Any]]:
        """Check application memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)

            status = HealthStatus.HEALTHY
            message = f"Application memory: {memory_mb:.1f} MB"

            details = {
                'rss_mb': round(memory_mb, 2),
                'vms_mb': round(memory_info.vms / (1024**2), 2),
                'memory_percent': round(process.memory_percent(), 2),
                'num_threads': process.num_threads()
            }

            return status, memory_mb, message, details

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"Application memory check failed: {e}", {}

    async def _check_internet_connectivity(self) -> Tuple[HealthStatus, float, str, Dict[str, Any]]:
        """Check internet connectivity"""
        try:
            start_time = time.time()

            # Test connection to a reliable service
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get('https://httpbin.org/status/200') as response:
                    response_time_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = f"Internet connectivity OK (response time: {response_time_ms:.1f}ms)"
                    else:
                        status = HealthStatus.WARNING
                        message = f"Internet connectivity degraded (HTTP {response.status})"

                    details = {
                        'response_code': response.status,
                        'response_time_ms': round(response_time_ms, 1)
                    }

                    return status, response_time_ms, message, details

        except Exception as e:
            return HealthStatus.CRITICAL, None, f"Internet connectivity check failed: {e}", {}

    # Recovery actions

    async def _recover_database_connection(self):
        """Attempt to recover database connection"""
        try:
            logger.info("Attempting database connection recovery")

            if self.db_manager:
                await self.db_manager.disconnect()
                await asyncio.sleep(2)
                await self.db_manager.connect()

        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            raise

    # Public API methods

    async def run_health_check(self, check_name: str) -> Optional[HealthResult]:
        """Run a specific health check manually"""
        try:
            if check_name not in self.health_checks:
                logger.error(f"Health check not found: {check_name}")
                return None

            await self._run_single_health_check(check_name)
            return self.latest_results.get(check_name)

        except Exception as e:
            logger.error(f"Manual health check failed for {check_name}: {e}")
            return None

    async def run_all_health_checks(self) -> Dict[str, HealthResult]:
        """Run all enabled health checks"""
        try:
            enabled_checks = [
                name for name, check in self.health_checks.items()
                if check.enabled
            ]

            await self._run_health_checks(enabled_checks)

            return {name: result for name, result in self.latest_results.items()
                   if name in enabled_checks}

        except Exception as e:
            logger.error(f"Failed to run all health checks: {e}")
            return {}

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        try:
            current_time = datetime.now()

            # Count status by category
            status_counts = {'healthy': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
            component_health = {}

            for check_name, result in self.latest_results.items():
                status_counts[result.status.value] += 1

                health_check = self.health_checks.get(check_name)
                if health_check:
                    component_type = health_check.component_type.value
                    if component_type not in component_health:
                        component_health[component_type] = {'healthy': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
                    component_health[component_type][result.status.value] += 1

            # Determine overall health
            if status_counts['critical'] > 0:
                overall_status = HealthStatus.CRITICAL
            elif status_counts['warning'] > 0:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY

            # Recent check statistics
            recent_results = [
                r for r in self.results_history
                if (current_time - r.timestamp).total_seconds() < 3600  # Last hour
            ]

            recent_failure_rate = 0
            if recent_results:
                failed_recent = len([r for r in recent_results if r.status != HealthStatus.HEALTHY])
                recent_failure_rate = (failed_recent / len(recent_results)) * 100

            return {
                'overall_status': overall_status.value,
                'timestamp': current_time.isoformat(),
                'monitoring_active': self.checking_active,
                'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
                'total_checks': len(self.health_checks),
                'enabled_checks': len([c for c in self.health_checks.values() if c.enabled]),
                'status_counts': status_counts,
                'component_health': component_health,
                'statistics': self.check_stats,
                'recent_failure_rate_percent': round(recent_failure_rate, 1),
                'recovery_attempts': dict(self.recovery_attempts)
            }

        except Exception as e:
            logger.error(f"Health summary generation failed: {e}")
            return {'error': str(e)}

    def get_health_check_history(self, check_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get history for a specific health check"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            history = [
                result.to_dict() for result in self.results_history
                if result.check_name == check_name and result.timestamp > cutoff_time
            ]

            return sorted(history, key=lambda x: x['timestamp'])

        except Exception as e:
            logger.error(f"Health check history retrieval failed: {e}")
            return []

    def get_latest_results(self) -> Dict[str, Dict[str, Any]]:
        """Get latest results for all health checks"""
        return {name: result.to_dict() for name, result in self.latest_results.items()}

    def get_health_check_definitions(self) -> List[Dict[str, Any]]:
        """Get all health check definitions"""
        try:
            definitions = []

            for name, health_check in self.health_checks.items():
                definition = {
                    'name': name,
                    'component_type': health_check.component_type.value,
                    'interval_seconds': health_check.interval_seconds,
                    'timeout_seconds': health_check.timeout_seconds,
                    'enabled': health_check.enabled,
                    'warning_threshold': health_check.warning_threshold,
                    'critical_threshold': health_check.critical_threshold,
                    'auto_recovery': health_check.auto_recovery,
                    'description': health_check.description,
                    'tags': health_check.tags
                }
                definitions.append(definition)

            return definitions

        except Exception as e:
            logger.error(f"Health check definitions retrieval failed: {e}")
            return []

# Convenience functions

async def create_health_checker() -> HealthChecker:
    """Create health checker with default checks"""
    return HealthChecker()

def get_system_health_quick() -> Dict[str, Any]:
    """Quick system health check"""
    try:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None,
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}

# Usage example
if __name__ == "__main__":

    async def test_health_checker():
        # Create health checker
        checker = HealthChecker()

        # Run all health checks once
        print("Running all health checks...")
        results = await checker.run_all_health_checks()

        print(f"Completed {len(results)} health checks:")
        for check_name, result in results.items():
            status_icon = "✅" if result.status == HealthStatus.HEALTHY else "⚠️" if result.status == HealthStatus.WARNING else "❌"
            print(f"  {status_icon} {check_name}: {result.message}")

        # Get system health summary
        summary = checker.get_system_health_summary()
        print(f"\nSystem Health Summary:")
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Status Counts: {summary['status_counts']}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Recent Failure Rate: {summary['recent_failure_rate_percent']:.1f}%")

        # Test a specific check
        cpu_result = await checker.run_health_check('cpu_usage')
        if cpu_result:
            print(f"\nCPU Check: {cpu_result.message} (took {cpu_result.duration_ms:.1f}ms)")

        print("Health checker test completed!")

    # Run test
    asyncio.run(test_health_checker())
