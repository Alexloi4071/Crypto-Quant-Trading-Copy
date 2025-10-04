"""
Test Monitoring Module
Comprehensive tests for monitoring system functionality
Tests system monitoring, alerting, notifications, and health checking
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring.system_monitor import SystemMonitor, MetricType, SystemMetric, SystemAlert, AlertLevel
from src.monitoring.alerting import AlertingSystem, AlertTrigger, AlertSeverity, AlertStatus, AlertRule, Alert
from src.monitoring.notifications import NotificationService, NotificationChannel, NotificationPriority, NotificationMessage
from src.monitoring.health_checker import HealthChecker, HealthStatus, ComponentType, HealthCheck, HealthResult


class TestSystemMonitor:
    """Test suite for SystemMonitor class"""

    @pytest.fixture
    def system_monitor(self):
        """Create SystemMonitor instance for testing"""
        return SystemMonitor()

    @pytest.fixture
    def sample_metrics(self):
        """Generate sample metrics for testing"""
        return {
            'cpu_usage': {'value': 75.5, 'unit': 'percent'},
            'memory_usage': {'value': 82.3, 'unit': 'percent'},
            'disk_usage': {'value': 65.2, 'unit': 'percent'},
            'daily_pnl': {'value': -150.0, 'unit': 'usd'},
            'portfolio_value': {'value': 9850.0, 'unit': 'usd'}
        }


    def test_system_monitor_initialization(self, system_monitor):
        """Test SystemMonitor initialization"""
        assert hasattr(system_monitor, 'thresholds')
        assert hasattr(system_monitor, 'current_metrics')
        assert hasattr(system_monitor, 'metric_history')
        assert hasattr(system_monitor, 'active_alerts')

        # Check default thresholds
        assert 'cpu_usage' in system_monitor.thresholds
        assert 'memory_usage' in system_monitor.thresholds
        assert 'daily_pnl' in system_monitor.thresholds

    @pytest.mark.asyncio
    async def test_collect_system_metrics(self, system_monitor):
        """Test system metrics collection"""
        with patch('psutil.cpu_percent', return_value=75.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 80.0
                mock_memory.return_value.total = 8 * (1024**3)  # 8GB
                mock_memory.return_value.available = 1.6 * (1024**3)  # 1.6GB

                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value.used = 500 * (1024**3)  # 500GB
                    mock_disk.return_value.total = 1000 * (1024**3)  # 1TB

                    await system_monitor._collect_system_metrics()

                    # Check that metrics were collected
                    assert 'cpu_usage' in system_monitor.current_metrics
                    assert 'memory_usage' in system_monitor.current_metrics
                    assert 'disk_usage' in system_monitor.current_metrics

                    # Verify metric values
                    assert system_monitor.current_metrics['cpu_usage'].value == 75.0
                    assert system_monitor.current_metrics['memory_usage'].value == 80.0
                    assert abs(system_monitor.current_metrics['disk_usage'].value - 50.0) < 1.0

    @pytest.mark.asyncio
    async def test_collect_trading_metrics(self, system_monitor):
        """Test trading metrics collection"""
        mock_portfolio_stats = {
            'total_value': 10500.0,
            'daily_pnl': 150.0,
            'total_pnl': 500.0,
            'open_positions': 3,
            'win_rate': 0.65
        }

        with patch.object(system_monitor.db_manager, 'get_portfolio_stats', return_value=mock_portfolio_stats):
            with patch.object(system_monitor.db_manager, 'get_recent_signals_count', return_value=15):
                with patch.object(system_monitor.db_manager, 'get_recent_trades_count', return_value=8):

                    await system_monitor._collect_trading_metrics()

                    # Check trading metrics
                    assert 'portfolio_value' in system_monitor.current_metrics
                    assert 'daily_pnl' in system_monitor.current_metrics
                    assert 'open_positions' in system_monitor.current_metrics

                    # Verify values
                    assert system_monitor.current_metrics['portfolio_value'].value == 10500.0
                    assert system_monitor.current_metrics['daily_pnl'].value == 150.0
                    assert system_monitor.current_metrics['open_positions'].value == 3

    @pytest.mark.asyncio
    async def test_threshold_checking(self, system_monitor):
        """Test threshold checking and alert generation"""
        # Set up metrics that exceed thresholds
        current_time = datetime.now()

        system_monitor.current_metrics['cpu_usage'] = SystemMetric(
            name='cpu_usage',
            value=95.0,  # Exceeds critical threshold of 85
            unit='percent',
            metric_type=MetricType.SYSTEM,
            timestamp=current_time
        )

        system_monitor.current_metrics['daily_pnl'] = SystemMetric(
            name='daily_pnl',
            value=-1200.0,  # Exceeds critical threshold of -1000
            unit='usd',
            metric_type=MetricType.TRADING,
            timestamp=current_time
        )

        await system_monitor._check_thresholds()

        # Should have generated alerts
        assert len(system_monitor.active_alerts) >= 2

        # Check specific alerts
        cpu_alert_found = any('cpu_usage' in alert_key for alert_key in system_monitor.active_alerts.keys())
        pnl_alert_found = any('daily_pnl' in alert_key for alert_key in system_monitor.active_alerts.keys())

        assert cpu_alert_found
        assert pnl_alert_found


    def test_get_system_health(self, system_monitor):
        """Test system health summary generation"""
        # Add some mock alerts
        system_monitor.active_alerts['test_critical'] = SystemAlert(
            metric_name='test_metric',
            level=AlertLevel.CRITICAL,
            message='Test critical alert',
            value=100.0,
            threshold=80.0,
            timestamp=datetime.now()
        )

        system_monitor.active_alerts['test_warning'] = SystemAlert(
            metric_name='test_metric_2',
            level=AlertLevel.WARNING,
            message='Test warning alert',
            value=75.0,
            threshold=70.0,
            timestamp=datetime.now()
        )

        health = system_monitor.get_system_health()

        assert isinstance(health, dict)
        assert 'status' in health
        assert 'alert_counts' in health

        # Should show critical status due to critical alert
        assert health['status'] == 'critical'
        assert health['alert_counts']['critical'] == 1
        assert health['alert_counts']['warning'] == 1


    def test_get_current_metrics(self, system_monitor):
        """Test getting current metrics"""
        # Add some test metrics
        system_monitor.current_metrics['test_metric'] = SystemMetric(
            name='test_metric',
            value=50.0,
            unit='percent',
            metric_type=MetricType.SYSTEM,
            timestamp=datetime.now()
        )

        metrics = system_monitor.get_current_metrics()

        assert isinstance(metrics, dict)
        assert 'test_metric' in metrics
        assert metrics['test_metric']['value'] == 50.0
        assert metrics['test_metric']['unit'] == 'percent'


class TestAlertingSystem:
    """Test suite for AlertingSystem class"""

    @pytest.fixture
    def alerting_system(self):
        """Create AlertingSystem instance for testing"""
        return AlertingSystem()

    @pytest.fixture
    def sample_alert_rule(self):
        """Generate sample alert rule for testing"""
        return AlertRule(
            id='test_cpu_high',
            name='Test High CPU Usage',
            description='Test rule for high CPU usage',
            trigger_type=AlertTrigger.THRESHOLD,
            severity=AlertSeverity.HIGH,
            metric_name='cpu_usage',
            condition='> 80',
            time_window_minutes=5,
            min_trigger_count=2,
            notification_channels=['email', 'telegram']
        )


    def test_alerting_system_initialization(self, alerting_system):
        """Test AlertingSystem initialization"""
        assert hasattr(alerting_system, 'alert_rules')
        assert hasattr(alerting_system, 'active_alerts')
        assert hasattr(alerting_system, 'metric_history')

        # Should have loaded default rules
        assert len(alerting_system.alert_rules) > 0
        assert 'cpu_high' in alerting_system.alert_rules
        assert 'memory_critical' in alerting_system.alert_rules


    def test_add_alert_rule(self, alerting_system, sample_alert_rule):
        """Test adding alert rules"""
        initial_count = len(alerting_system.alert_rules)

        success = alerting_system.add_rule(sample_alert_rule)

        assert success is True
        assert len(alerting_system.alert_rules) == initial_count + 1
        assert sample_alert_rule.id in alerting_system.alert_rules


    def test_remove_alert_rule(self, alerting_system, sample_alert_rule):
        """Test removing alert rules"""
        # Add rule first
        alerting_system.add_rule(sample_alert_rule)

        # Remove it
        success = alerting_system.remove_rule(sample_alert_rule.id)

        assert success is True
        assert sample_alert_rule.id not in alerting_system.alert_rules

        # Try to remove non-existent rule
        success = alerting_system.remove_rule('non_existent_rule')
        assert success is False


    def test_enable_disable_rule(self, alerting_system, sample_alert_rule):
        """Test enabling and disabling alert rules"""
        alerting_system.add_rule(sample_alert_rule)

        # Test disable
        success = alerting_system.disable_rule(sample_alert_rule.id)
        assert success is True
        assert alerting_system.alert_rules[sample_alert_rule.id].enabled is False

        # Test enable
        success = alerting_system.enable_rule(sample_alert_rule.id)
        assert success is True
        assert alerting_system.alert_rules[sample_alert_rule.id].enabled is True

    @pytest.mark.asyncio
    async def test_evaluate_threshold_rule(self, alerting_system):
        """Test threshold rule evaluation"""
        # Create test data that should trigger alert
        test_data = [(datetime.now() - timedelta(minutes=i), 85.0) for i in range(5, 0, -1)]

        rule = AlertRule(
            id='test_threshold',
            name='Test Threshold',
            description='Test',
            trigger_type=AlertTrigger.THRESHOLD,
            severity=AlertSeverity.HIGH,
            metric_name='test_metric',
            condition='> 80',
            time_window_minutes=10,
            min_trigger_count=3
        )

        triggered, value = alerting_system._evaluate_threshold_rule(rule, test_data)

        assert triggered is True
        assert value == 85.0

    @pytest.mark.asyncio
    async def test_evaluate_rate_of_change_rule(self, alerting_system):
        """Test rate of change rule evaluation"""
        # Create test data with significant change
        base_time = datetime.now()
        test_data = [
            (base_time - timedelta(minutes=10), 100.0),
            (base_time - timedelta(minutes=5), 120.0),
            (base_time, 150.0)  # 50% increase
        ]

        rule = AlertRule(
            id='test_rate_change',
            name='Test Rate Change',
            description='Test',
            trigger_type=AlertTrigger.RATE_OF_CHANGE,
            severity=AlertSeverity.MEDIUM,
            metric_name='test_metric',
            condition='change > 30'
        )

        triggered, value = alerting_system._evaluate_rate_of_change_rule(rule, test_data)

        assert triggered is True
        assert value > 30  # Should be around 50%

    @pytest.mark.asyncio
    async def test_alert_suppression(self, alerting_system, sample_alert_rule):
        """Test alert suppression functionality"""
        alerting_system.add_rule(sample_alert_rule)

        current_time = datetime.now()

        # First alert should not be suppressed
        is_suppressed_1 = alerting_system._is_suppressed(sample_alert_rule, current_time)
        assert is_suppressed_1 is False

        # Add alert to counter
        alerting_system.alert_counters[sample_alert_rule.id] = [current_time]

        # Second alert within suppression window should be suppressed
        is_suppressed_2 = alerting_system._is_suppressed(sample_alert_rule, current_time + timedelta(minutes=30))
        assert is_suppressed_2 is True

    @pytest.mark.asyncio
    async def test_metric_evaluation(self, alerting_system):
        """Test complete metric evaluation process"""
        test_metrics = {
            'cpu_usage': {'value': 90.0, 'unit': 'percent'},
            'memory_usage': {'value': 95.0, 'unit': 'percent'}
        }

        # Mock notification handler
        mock_handler = AsyncMock()
        alerting_system.register_notification_handler('test_channel', mock_handler)

        await alerting_system.evaluate_metrics(test_metrics)

        # Should have generated alerts for high CPU and memory
        assert len(alerting_system.active_alerts) > 0


    def test_get_alerting_statistics(self, alerting_system):
        """Test alerting statistics generation"""
        # Add some test data
        alerting_system.stats['total_alerts_generated'] = 10
        alerting_system.stats['alerts_by_severity']['high'] = 3
        alerting_system.stats['alerts_by_severity']['medium'] = 7

        stats = alerting_system.get_alerting_statistics()

        assert isinstance(stats, dict)
        assert 'total_alerts_generated' in stats
        assert 'alerts_by_severity' in stats
        assert 'active_alerts_count' in stats
        assert stats['total_alerts_generated'] == 10


class TestNotificationService:
    """Test suite for NotificationService class"""

    @pytest.fixture
    def notification_service(self):
        """Create NotificationService instance for testing"""
        return NotificationService()

    @pytest.mark.asyncio
    async def test_send_notification_telegram(self, notification_service):
        """Test sending Telegram notification"""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'ok': True, 'result': {'message_id': 123}})
            mock_post.return_value.__aenter__.return_value = mock_response

            # Set up config for testing
            notification_service.config.telegram_bot_token = 'test_token'
            notification_service.config.telegram_chat_id = 'test_chat_id'

            success = await notification_service.send_notification(
                title='Test Alert',
                content='This is a test notification',
                channel=NotificationChannel.TELEGRAM,
                priority=NotificationPriority.HIGH
            )

            assert success is True
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_notification(self, notification_service):
        """Test sending email notification"""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            # Set up config for testing
            notification_service.config.smtp_server = 'smtp.test.com'
            notification_service.config.smtp_username = 'test@test.com'
            notification_service.config.smtp_password = 'password'
            notification_service.config.email_from = 'test@test.com'
            notification_service.config.email_to = ['recipient@test.com']

            success = await notification_service.send_notification(
                title='Test Email',
                content='This is a test email',
                channel=NotificationChannel.EMAIL,
                priority=NotificationPriority.NORMAL
            )

            assert success is True
            mock_server.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_notification(self, notification_service):
        """Test sending alert notification"""
        alert_dict = {
            'id': 'test_alert',
            'severity': 'high',
            'message': 'High CPU usage detected',
            'metric_name': 'cpu_usage',
            'value': 90.0,
            'threshold': 80.0,
            'timestamp': datetime.now().isoformat()
        }

        rule_dict = {
            'name': 'High CPU Alert',
            'description': 'Alert for high CPU usage',
            'notification_channels': ['telegram']
        }

        with patch.object(notification_service, 'send_notification', return_value=True) as mock_send:
            results = await notification_service.send_alert_notification(alert_dict, rule_dict)

            assert 'telegram' in results
            assert results['telegram'] is True
            mock_send.assert_called_once()


    def test_format_alert_message(self, notification_service):
        """Test alert message formatting"""
        alert_dict = {
            'severity': 'critical',
            'message': 'System overload detected',
            'metric_name': 'cpu_usage',
            'value': 95.0,
            'threshold': 80.0,
            'timestamp': '2024-01-01T12:00:00'
        }

        rule_dict = {
            'name': 'CPU Overload Alert',
            'description': 'Critical CPU usage detected'
        }

        formatted_message = notification_service._format_alert_message(alert_dict, rule_dict)

        assert isinstance(formatted_message, str)
        assert 'CRITICAL ALERT' in formatted_message
        assert 'cpu_usage' in formatted_message
        assert '95.00' in formatted_message
        assert '80.00' in formatted_message


    def test_notification_statistics(self, notification_service):
        """Test notification statistics"""
        # Add some test data
        notification_service.stats['total_sent'] = 50
        notification_service.stats['total_failed'] = 5
        notification_service.stats['by_channel']['telegram']['sent'] = 30
        notification_service.stats['by_channel']['email']['sent'] = 20

        stats = notification_service.get_notification_statistics()

        assert isinstance(stats, dict)
        assert 'success_rate' in stats
        assert 'by_channel' in stats
        assert abs(stats['success_rate'] - 90.9) < 0.1  # 50/55 * 100


    def test_rate_limiting(self, notification_service):
        """Test notification rate limiting"""
        channel = NotificationChannel.TELEGRAM
        current_time = datetime.now()

        # First check should pass
        can_send_1 = asyncio.run(notification_service._check_rate_limit(channel))
        assert can_send_1 is True

        # Simulate hitting rate limit
        rate_limit = notification_service.rate_limits[channel]
        rate_limit['sent_this_minute'] = rate_limit['max_per_minute']

        # Should be rate limited
        can_send_2 = asyncio.run(notification_service._check_rate_limit(channel))
        assert can_send_2 is False


class TestHealthChecker:
    """Test suite for HealthChecker class"""

    @pytest.fixture
    def health_checker(self):
        """Create HealthChecker instance for testing"""
        return HealthChecker()

    def test_health_checker_initialization(self, health_checker):
        """Test HealthChecker initialization"""
        assert hasattr(health_checker, 'health_checks')
        assert hasattr(health_checker, 'latest_results')

        # Should have registered default checks
        assert len(health_checker.health_checks) > 0
        assert 'cpu_usage' in health_checker.health_checks
        assert 'memory_usage' in health_checker.health_checks
        assert 'database_connection' in health_checker.health_checks


    def test_register_health_check(self, health_checker):
        """Test registering custom health checks"""


        def test_check():
            return True

        custom_check = HealthCheck(
            name='test_custom_check',
            component_type=ComponentType.APPLICATION,
            check_function=test_check,
            interval_seconds=30,
            description='Test custom health check'
        )

        health_checker.register_health_check(custom_check)

        assert 'test_custom_check' in health_checker.health_checks
        assert health_checker.health_checks['test_custom_check'].description == 'Test custom health check'

    @pytest.mark.asyncio
    async def test_cpu_health_check(self, health_checker):
        """Test CPU usage health check"""
        with patch('psutil.cpu_percent', return_value=75.0):
            with patch('psutil.cpu_count', return_value=8):
                result = await health_checker._check_cpu_usage()

                assert isinstance(result, tuple)
                assert len(result) == 4  # status, value, message, details

                status, value, message, details = result
                assert status == HealthStatus.HEALTHY
                assert value == 75.0
                assert 'CPU usage' in message
                assert details['cpu_count'] == 8

    @pytest.mark.asyncio
    async def test_memory_health_check(self, health_checker):
        """Test memory usage health check"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 65.0
            mock_memory.return_value.total = 8 * (1024**3)
            mock_memory.return_value.available = 2.8 * (1024**3)
            mock_memory.return_value.used = 5.2 * (1024**3)

            result = await health_checker._check_memory_usage()

            status, value, message, details = result
            assert status == HealthStatus.HEALTHY
            assert value == 65.0
            assert 'Memory usage' in message
            assert 'total_gb' in details

    @pytest.mark.asyncio
    async def test_database_health_check(self, health_checker):
        """Test database connection health check"""
        with patch.object(health_checker.db_manager, 'is_connected', return_value=True):
            with patch.object(health_checker.db_manager, 'get_connection_info', return_value={'status': 'connected'}):
                result = await health_checker._check_database_connection()

                status, value, message, details = result
                assert status == HealthStatus.HEALTHY
                assert 'Database connected' in message

    @pytest.mark.asyncio
    async def test_internet_connectivity_check(self, health_checker):
        """Test internet connectivity check"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await health_checker._check_internet_connectivity()

            status, value, message, details = result
            assert status == HealthStatus.HEALTHY
            assert 'Internet connectivity OK' in message
            assert 'response_code' in details

    @pytest.mark.asyncio
    async def test_run_health_check(self, health_checker):
        """Test running a specific health check"""
        with patch('psutil.cpu_percent', return_value=45.0):
            result = await health_checker.run_health_check('cpu_usage')

            assert isinstance(result, HealthResult)
            assert result.check_name == 'cpu_usage'
            assert result.status == HealthStatus.HEALTHY
            assert result.value == 45.0

    @pytest.mark.asyncio
    async def test_run_all_health_checks(self, health_checker):
        """Test running all health checks"""
        # Mock various system calls to avoid actual system dependencies
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.0
                mock_memory.return_value.total = 8 * (1024**3)
                mock_memory.return_value.available = 3.2 * (1024**3)
                mock_memory.return_value.used = 4.8 * (1024**3)

                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value.used = 400 * (1024**3)
                    mock_disk.return_value.total = 1000 * (1024**3)
                    mock_disk.return_value.free = 600 * (1024**3)

                    results = await health_checker.run_all_health_checks()

                    assert isinstance(results, dict)
                    assert len(results) > 0

                    # Should have results for system checks
                    assert 'cpu_usage' in results
                    assert 'memory_usage' in results
                    assert 'disk_usage' in results


    def test_get_system_health_summary(self, health_checker):
        """Test system health summary generation"""
        # Add some test results
        health_checker.latest_results['test_check_1'] = HealthResult(
            check_name='test_check_1',
            status=HealthStatus.HEALTHY,
            value=50.0,
            message='Test check passed',
            timestamp=datetime.now(),
            duration_ms=100.0
        )

        health_checker.latest_results['test_check_2'] = HealthResult(
            check_name='test_check_2',
            status=HealthStatus.WARNING,
            value=80.0,
            message='Test check warning',
            timestamp=datetime.now(),
            duration_ms=150.0
        )

        summary = health_checker.get_system_health_summary()

        assert isinstance(summary, dict)
        assert 'overall_status' in summary
        assert 'status_counts' in summary
        assert 'component_health' in summary

        # Should show warning status due to warning check
        assert summary['overall_status'] == 'warning'
        assert summary['status_counts']['healthy'] == 1
        assert summary['status_counts']['warning'] == 1

# Integration Tests


class TestMonitoringIntegration:
    """Integration tests for monitoring components working together"""

    @pytest.fixture
    async def integrated_monitoring(self):
        """Create integrated monitoring system"""
        system_monitor = SystemMonitor()
        alerting_system = AlertingSystem()
        notification_service = NotificationService()
        health_checker = HealthChecker()

        # Connect components
        alerting_system.register_notification_handler(
            'test_channel',
            notification_service.send_alert_notification
        )

        yield {
            'system_monitor': system_monitor,
            'alerting_system': alerting_system,
            'notification_service': notification_service,
            'health_checker': health_checker
        }

    @pytest.mark.asyncio
    async def test_monitoring_pipeline(self, integrated_monitoring):
        """Test complete monitoring pipeline"""
        system_monitor = integrated_monitoring['system_monitor']
        alerting_system = integrated_monitoring['alerting_system']

        # Mock high CPU usage
        with patch('psutil.cpu_percent', return_value=95.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 85.0
                mock_memory.return_value.total = 8 * (1024**3)
                mock_memory.return_value.available = 1.2 * (1024**3)

                # Collect metrics
                await system_monitor._collect_system_metrics()

                # Get metrics for alerting
                current_metrics = system_monitor.get_current_metrics()

                # Evaluate for alerts
                await alerting_system.evaluate_metrics(current_metrics)

                # Should have generated alerts
                active_alerts = alerting_system.get_active_alerts()
                assert len(active_alerts) > 0

                # Should have CPU alert
                cpu_alert_found = any(
                    alert['metric_name'] == 'cpu_usage'
                    for alert in active_alerts
                )
                assert cpu_alert_found

# Performance Tests


class TestMonitoringPerformance:
    """Performance tests for monitoring operations"""

    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance"""
        system_monitor = SystemMonitor()

        import time
        start_time = time.time()

        # Run multiple collection cycles
        for _ in range(10):
            await system_monitor._collect_system_metrics()

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete quickly
        assert total_time < 5.0  # 5 seconds for 10 collections

        # Should have collected metrics
        assert len(system_monitor.current_metrics) > 0


    def test_large_scale_alert_evaluation(self):
        """Test alert evaluation with many rules"""
        alerting_system = AlertingSystem()

        # Add many test rules
        for i in range(100):
            rule = AlertRule(
                id=f'test_rule_{i}',
                name=f'Test Rule {i}',
                description=f'Test rule number {i}',
                trigger_type=AlertTrigger.THRESHOLD,
                severity=AlertSeverity.MEDIUM,
                metric_name=f'test_metric_{i}',
                condition='> 50'
            )
            alerting_system.add_rule(rule)

        # Create test metrics
        test_metrics = {f'test_metric_{i}': {'value': 60.0, 'unit': 'percent'} for i in range(100)}

        import time
        start_time = time.time()

        # This is sync, so we use asyncio.run for the async method
        asyncio.run(alerting_system.evaluate_metrics(test_metrics))

        end_time = time.time()
        evaluation_time = end_time - start_time

        # Should complete within reasonable time
        assert evaluation_time < 2.0  # 2 seconds max

        # Should have evaluated all rules
        assert len(alerting_system.alert_rules) == 107  # 100 + 7 default rules

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
