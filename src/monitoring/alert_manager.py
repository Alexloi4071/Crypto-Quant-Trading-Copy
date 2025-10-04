"""
Alerting System Module
Intelligent alerting system with rule-based triggers and escalation
Supports multiple channels and alert suppression
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import re
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import timing_decorator

logger = setup_logger(__name__)

class AlertTrigger(Enum):
    """Alert trigger types"""
    THRESHOLD = "threshold"
    RATE_OF_CHANGE = "rate_of_change"
    ANOMALY = "anomaly"
    PATTERN = "pattern"
    COMPOSITE = "composite"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass

class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    description: str
    trigger_type: AlertTrigger
    severity: AlertSeverity
    metric_name: str
    condition: str  # e.g., "> 80", "< -1000", "change > 50%"
    time_window_minutes: int = 5
    min_trigger_count: int = 1
    enabled: bool = True

    # Escalation settings
    escalation_delay_minutes: int = 30
    escalation_severity: Optional[AlertSeverity] = None

    # Suppression settings
    suppression_window_minutes: int = 60
    max_alerts_per_hour: int = 5

    # Notification channels
    notification_channels: List[str] = field(default_factory=list)

    # Custom metadata
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'trigger_type': self.trigger_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'condition': self.condition,
            'time_window_minutes': self.time_window_minutes,
            'min_trigger_count': self.min_trigger_count,
            'enabled': self.enabled,
            'escalation_delay_minutes': self.escalation_delay_minutes,
            'escalation_severity': self.escalation_severity.value if self.escalation_severity else None,
            'suppression_window_minutes': self.suppression_window_minutes,
            'max_alerts_per_hour': self.max_alerts_per_hour,
            'notification_channels': self.notification_channels,
            'tags': self.tags
        }

@dataclass

class Alert:
    """Alert instance"""
    id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: float
    threshold: Optional[float]
    timestamp: datetime

    # Resolution tracking
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None

    # Escalation tracking
    escalated: bool = False
    escalated_at: Optional[datetime] = None

    # Notification tracking
    notifications_sent: List[str] = field(default_factory=list)

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'rule_id': self.rule_id,
            'metric_name': self.metric_name,
            'severity': self.severity.value,
            'status': self.status.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_note': self.resolution_note,
            'escalated': self.escalated,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'notifications_sent': self.notifications_sent,
            'context': self.context
        }

class AlertingSystem:
    """Intelligent alerting system"""

    def __init__(self):
        # Alert rules storage
        self.alert_rules: Dict[str, AlertRule] = {}

        # Active alerts
        self.active_alerts: Dict[str, Alert] = {}

        # Alert history
        self.alert_history: List[Alert] = []

        # Metric history for rule evaluation
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Alert counters for suppression
        self.alert_counters: Dict[str, List[datetime]] = {}

        # Notification callbacks
        self.notification_handlers: Dict[str, Callable] = {}

        # System state
        self.alerting_enabled = True
        self.last_evaluation_time = None

        # Statistics
        self.stats = {
            'total_alerts_generated': 0,
            'alerts_by_severity': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'alerts_by_rule': {},
            'avg_resolution_time_minutes': 0,
            'suppressed_alerts': 0
        }

        logger.info("Alerting system initialized")
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default alert rules"""
        try:
            default_rules = [
                AlertRule(
                    id='cpu_high',
                    name='High CPU Usage',
                    description='CPU usage is consistently high',
                    trigger_type=AlertTrigger.THRESHOLD,
                    severity=AlertSeverity.HIGH,
                    metric_name='cpu_usage',
                    condition='> 85',
                    time_window_minutes=5,
                    min_trigger_count=3,
                    notification_channels=['telegram', 'email']
                ),
                AlertRule(
                    id='memory_critical',
                    name='Critical Memory Usage',
                    description='Memory usage is critically high',
                    trigger_type=AlertTrigger.THRESHOLD,
                    severity=AlertSeverity.CRITICAL,
                    metric_name='memory_usage',
                    condition='> 95',
                    time_window_minutes=2,
                    min_trigger_count=2,
                    notification_channels=['telegram', 'email', 'discord'],
                    escalation_delay_minutes=15
                ),
                AlertRule(
                    id='trading_large_loss',
                    name='Large Trading Loss',
                    description='Daily P&L shows significant loss',
                    trigger_type=AlertTrigger.THRESHOLD,
                    severity=AlertSeverity.CRITICAL,
                    metric_name='daily_pnl',
                    condition='< -1000',
                    time_window_minutes=1,
                    min_trigger_count=1,
                    notification_channels=['telegram', 'email', 'discord']
                ),
                AlertRule(
                    id='portfolio_drawdown',
                    name='Portfolio Drawdown Warning',
                    description='Portfolio drawdown exceeds warning threshold',
                    trigger_type=AlertTrigger.THRESHOLD,
                    severity=AlertSeverity.HIGH,
                    metric_name='drawdown',
                    condition='> 15',
                    time_window_minutes=1,
                    min_trigger_count=1,
                    notification_channels=['telegram', 'email']
                ),
                AlertRule(
                    id='system_errors',
                    name='High System Error Rate',
                    description='System experiencing high error rate',
                    trigger_type=AlertTrigger.RATE_OF_CHANGE,
                    severity=AlertSeverity.MEDIUM,
                    metric_name='metrics_collection_errors',
                    condition='change > 10',
                    time_window_minutes=10,
                    notification_channels=['telegram']
                ),
                AlertRule(
                    id='disk_space_warning',
                    name='Disk Space Warning',
                    description='Disk space is running low',
                    trigger_type=AlertTrigger.THRESHOLD,
                    severity=AlertSeverity.MEDIUM,
                    metric_name='disk_usage',
                    condition='> 80',
                    time_window_minutes=5,
                    min_trigger_count=2,
                    notification_channels=['email']
                ),
                AlertRule(
                    id='no_trading_signals',
                    name='No Trading Signals Generated',
                    description='No trading signals have been generated recently',
                    trigger_type=AlertTrigger.THRESHOLD,
                    severity=AlertSeverity.MEDIUM,
                    metric_name='signals_generated_today',
                    condition='< 1',
                    time_window_minutes=60,
                    min_trigger_count=1,
                    notification_channels=['telegram']
                )
            ]

            for rule in default_rules:
                self.alert_rules[rule.id] = rule

            logger.info(f"Loaded {len(default_rules)} default alert rules")

        except Exception as e:
            logger.error(f"Failed to load default alert rules: {e}")

    def add_rule(self, rule: AlertRule) -> bool:
        """Add or update an alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        try:
            if rule_id in self.alert_rules:
                rule_name = self.alert_rules[rule_id].name
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_name} ({rule_id})")
                return True
            else:
                logger.warning(f"Alert rule not found: {rule_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove alert rule: {e}")
            return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule"""
        try:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = True
                logger.info(f"Enabled alert rule: {rule_id}")
                return True
            else:
                logger.warning(f"Alert rule not found: {rule_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to enable alert rule: {e}")
            return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule"""
        try:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = False
                logger.info(f"Disabled alert rule: {rule_id}")
                return True
            else:
                logger.warning(f"Alert rule not found: {rule_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to disable alert rule: {e}")
            return False

    def register_notification_handler(self, channel: str, handler: Callable):
        """Register a notification handler for a channel"""
        try:
            self.notification_handlers[channel] = handler
            logger.info(f"Registered notification handler for channel: {channel}")

        except Exception as e:
            logger.error(f"Failed to register notification handler: {e}")

    @timing_decorator

    async def evaluate_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """Evaluate metrics against alert rules"""
        try:
            if not self.alerting_enabled:
                return

            current_time = datetime.now()
            self.last_evaluation_time = current_time

            # Update metric history
            self._update_metric_history(metrics, current_time)

            # Evaluate each rule
            for rule_id, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue

                try:
                    await self._evaluate_rule(rule, current_time)
                except Exception as e:
                    logger.error(f"Rule evaluation failed for {rule_id}: {e}")

            # Check for escalations
            await self._check_escalations(current_time)

            # Clean up old metric history
            self._cleanup_metric_history(current_time)

        except Exception as e:
            logger.error(f"Metric evaluation failed: {e}")

    def _update_metric_history(self, metrics: Dict[str, Dict[str, Any]], timestamp: datetime):
        """Update metric history"""
        try:
            for metric_name, metric_data in metrics.items():
                value = metric_data.get('value', 0)

                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []

                self.metric_history[metric_name].append((timestamp, value))

                # Keep only recent history (last 24 hours)
                cutoff_time = timestamp - timedelta(hours=24)
                self.metric_history[metric_name] = [
                    (ts, val) for ts, val in self.metric_history[metric_name]
                    if ts > cutoff_time
                ]

        except Exception as e:
            logger.error(f"Metric history update failed: {e}")

    async def _evaluate_rule(self, rule: AlertRule, current_time: datetime):
        """Evaluate a single alert rule"""
        try:
            metric_name = rule.metric_name

            # Get metric history for the rule's time window
            if metric_name not in self.metric_history:
                return

            window_start = current_time - timedelta(minutes=rule.time_window_minutes)
            window_data = [
                (ts, val) for ts, val in self.metric_history[metric_name]
                if ts >= window_start
            ]

            if len(window_data) < rule.min_trigger_count:
                return

            # Evaluate based on trigger type
            triggered = False
            trigger_value = None

            if rule.trigger_type == AlertTrigger.THRESHOLD:
                triggered, trigger_value = self._evaluate_threshold_rule(rule, window_data)
            elif rule.trigger_type == AlertTrigger.RATE_OF_CHANGE:
                triggered, trigger_value = self._evaluate_rate_of_change_rule(rule, window_data)
            elif rule.trigger_type == AlertTrigger.ANOMALY:
                triggered, trigger_value = self._evaluate_anomaly_rule(rule, window_data)

            if triggered:
                # Check suppression
                if self._is_suppressed(rule, current_time):
                    self.stats['suppressed_alerts'] += 1
                    return

                # Generate alert
                await self._generate_alert(rule, trigger_value, current_time)
            else:
                # Check if we should resolve existing alerts
                await self._check_alert_resolution(rule, current_time)

        except Exception as e:
            logger.error(f"Rule evaluation failed for {rule.id}: {e}")

    def _evaluate_threshold_rule(self, rule: AlertRule, window_data: List[Tuple[datetime, float]]) -> Tuple[bool, Optional[float]]:
        """Evaluate threshold-based rule"""
        try:
            condition = rule.condition.strip()

            # Parse condition (e.g., "> 80", "< -1000", "<= 5")
            if condition.startswith('>='):
                operator = '>='
                threshold = float(condition[2:].strip())
            elif condition.startswith('<='):
                operator = '<='
                threshold = float(condition[2:].strip())
            elif condition.startswith('>'):
                operator = '>'
                threshold = float(condition[1:].strip())
            elif condition.startswith('<'):
                operator = '<'
                threshold = float(condition[1:].strip())
            elif condition.startswith('=='):
                operator = '=='
                threshold = float(condition[2:].strip())
            else:
                logger.warning(f"Unknown condition format: {condition}")
                return False, None

            # Count how many values in window trigger the condition
            trigger_count = 0
            latest_value = None

            for timestamp, value in window_data:
                latest_value = value

                if operator == '>' and value > threshold:
                    trigger_count += 1
                elif operator == '>=' and value >= threshold:
                    trigger_count += 1
                elif operator == '<' and value < threshold:
                    trigger_count += 1
                elif operator == '<=' and value <= threshold:
                    trigger_count += 1
                elif operator == '==' and value == threshold:
                    trigger_count += 1

            # Rule triggers if minimum number of values meet condition
            triggered = trigger_count >= rule.min_trigger_count

            return triggered, latest_value

        except Exception as e:
            logger.error(f"Threshold rule evaluation failed: {e}")
            return False, None

    def _evaluate_rate_of_change_rule(self, rule: AlertRule, window_data: List[Tuple[datetime, float]]) -> Tuple[bool, Optional[float]]:
        """Evaluate rate of change rule"""
        try:
            if len(window_data) < 2:
                return False, None

            # Sort by timestamp
            sorted_data = sorted(window_data, key=lambda x: x[0])

            # Calculate rate of change
            first_value = sorted_data[0][1]
            last_value = sorted_data[-1][1]

            if first_value == 0:
                # Avoid division by zero
                change_rate = float('inf') if last_value > 0 else float('-inf')
            else:
                change_rate = ((last_value - first_value) / abs(first_value)) * 100

            # Parse condition (e.g., "change > 50", "change < -20")
            condition = rule.condition.strip().lower()

            if 'change >' in condition:
                threshold = float(condition.split('change >')[1].strip().replace('%', ''))
                triggered = change_rate > threshold
            elif 'change <' in condition:
                threshold = float(condition.split('change <')[1].strip().replace('%', ''))
                triggered = change_rate < threshold
            else:
                logger.warning(f"Unknown rate of change condition: {condition}")
                return False, None

            return triggered, change_rate

        except Exception as e:
            logger.error(f"Rate of change rule evaluation failed: {e}")
            return False, None

    def _evaluate_anomaly_rule(self, rule: AlertRule, window_data: List[Tuple[datetime, float]]) -> Tuple[bool, Optional[float]]:
        """Evaluate anomaly detection rule"""
        try:
            if len(window_data) < 10:  # Need sufficient data for anomaly detection
                return False, None

            values = [val for _, val in window_data]

            # Simple anomaly detection using z-score
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:
                return False, None

            latest_value = values[-1]
            z_score = abs((latest_value - mean_val) / std_val)

            # Parse condition (e.g., "anomaly > 3", "anomaly > 2.5")
            condition = rule.condition.strip().lower()

            if 'anomaly >' in condition:
                threshold = float(condition.split('anomaly >')[1].strip())
                triggered = z_score > threshold
            else:
                logger.warning(f"Unknown anomaly condition: {condition}")
                return False, None

            return triggered, z_score

        except Exception as e:
            logger.error(f"Anomaly rule evaluation failed: {e}")
            return False, None

    def _is_suppressed(self, rule: AlertRule, current_time: datetime) -> bool:
        """Check if alert should be suppressed"""
        try:
            rule_id = rule.id

            # Initialize counter if needed
            if rule_id not in self.alert_counters:
                self.alert_counters[rule_id] = []

            # Clean old alerts outside suppression window
            window_start = current_time - timedelta(minutes=rule.suppression_window_minutes)
            self.alert_counters[rule_id] = [
                ts for ts in self.alert_counters[rule_id] if ts > window_start
            ]

            # Check if we've exceeded the maximum alerts per hour
            hour_start = current_time - timedelta(hours=1)
            alerts_last_hour = len([ts for ts in self.alert_counters[rule_id] if ts > hour_start])

            if alerts_last_hour >= rule.max_alerts_per_hour:
                return True

            # Check recent alerts within suppression window
            recent_alerts = len(self.alert_counters[rule_id])

            return recent_alerts > 0

        except Exception as e:
            logger.error(f"Suppression check failed: {e}")
            return False

    async def _generate_alert(self, rule: AlertRule, value: float, timestamp: datetime):
        """Generate a new alert"""
        try:
            # Create alert ID
            alert_id = f"{rule.id}_{int(timestamp.timestamp())}"

            # Create alert message
            if rule.trigger_type == AlertTrigger.THRESHOLD:
                threshold = self._extract_threshold_from_condition(rule.condition)
                message = f"{rule.name}: {rule.metric_name} is {value:.2f} (threshold: {threshold:.2f})"
            elif rule.trigger_type == AlertTrigger.RATE_OF_CHANGE:
                message = f"{rule.name}: {rule.metric_name} changed by {value:.1f}%"
            elif rule.trigger_type == AlertTrigger.ANOMALY:
                message = f"{rule.name}: {rule.metric_name} shows anomaly (z-score: {value:.2f})"
            else:
                message = f"{rule.name}: {rule.description}"

            # Create alert
            alert = Alert(
                id=alert_id,
                rule_id=rule.id,
                metric_name=rule.metric_name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=message,
                value=value,
                threshold=self._extract_threshold_from_condition(rule.condition) if rule.trigger_type == AlertTrigger.THRESHOLD else None,
                timestamp=timestamp,
                context={
                    'rule_name': rule.name,
                    'trigger_type': rule.trigger_type.value,
                    'condition': rule.condition
                }
            )

            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)

            # Update statistics
            self.stats['total_alerts_generated'] += 1
            self.stats['alerts_by_severity'][rule.severity.value] += 1

            if rule.id not in self.stats['alerts_by_rule']:
                self.stats['alerts_by_rule'][rule.id] = 0
            self.stats['alerts_by_rule'][rule.id] += 1

            # Update alert counter for suppression
            if rule.id not in self.alert_counters:
                self.alert_counters[rule.id] = []
            self.alert_counters[rule.id].append(timestamp)

            # Send notifications
            await self._send_notifications(alert, rule)

            logger.warning(f"ALERT GENERATED [{rule.severity.value.upper()}]: {message}")

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")

    def _extract_threshold_from_condition(self, condition: str) -> Optional[float]:
        """Extract threshold value from condition string"""
        try:
            condition = condition.strip()

            for operator in ['>=', '<=', '>', '<', '==']:
                if condition.startswith(operator):
                    return float(condition[len(operator):].strip())

            return None

        except Exception:
            return None

    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for an alert"""
        try:
            for channel in rule.notification_channels:
                if channel in self.notification_handlers:
                    try:
                        await self.notification_handlers[channel](alert, rule)
                        alert.notifications_sent.append(channel)
                        logger.debug(f"Notification sent via {channel} for alert {alert.id}")
                    except Exception as e:
                        logger.error(f"Notification failed for channel {channel}: {e}")
                else:
                    logger.warning(f"No handler registered for notification channel: {channel}")

        except Exception as e:
            logger.error(f"Notification sending failed: {e}")

    async def _check_escalations(self, current_time: datetime):
        """Check for alerts that need escalation"""
        try:
            for alert in self.active_alerts.values():
                if alert.status != AlertStatus.ACTIVE or alert.escalated:
                    continue

                rule = self.alert_rules.get(alert.rule_id)
                if not rule or not rule.escalation_severity:
                    continue

                # Check if escalation delay has passed
                escalation_time = alert.timestamp + timedelta(minutes=rule.escalation_delay_minutes)

                if current_time >= escalation_time:
                    await self._escalate_alert(alert, rule, current_time)

        except Exception as e:
            logger.error(f"Escalation check failed: {e}")

    async def _escalate_alert(self, alert: Alert, rule: AlertRule, current_time: datetime):
        """Escalate an alert"""
        try:
            alert.escalated = True
            alert.escalated_at = current_time
            alert.severity = rule.escalation_severity

            # Send escalation notifications
            escalation_message = f"ESCALATED: {alert.message}"
            escalated_alert = Alert(
                id=f"{alert.id}_escalated",
                rule_id=alert.rule_id,
                metric_name=alert.metric_name,
                severity=rule.escalation_severity,
                status=AlertStatus.ACTIVE,
                message=escalation_message,
                value=alert.value,
                threshold=alert.threshold,
                timestamp=current_time,
                context=alert.context
            )

            await self._send_notifications(escalated_alert, rule)

            logger.critical(f"ALERT ESCALATED: {escalation_message}")

        except Exception as e:
            logger.error(f"Alert escalation failed: {e}")

    async def _check_alert_resolution(self, rule: AlertRule, current_time: datetime):
        """Check if alerts should be automatically resolved"""
        try:
            # Find active alerts for this rule
            rule_alerts = [
                alert for alert in self.active_alerts.values()
                if alert.rule_id == rule.id and alert.status == AlertStatus.ACTIVE
            ]

            if not rule_alerts:
                return

            # Get recent metric values
            metric_name = rule.metric_name
            if metric_name not in self.metric_history:
                return

            # Check if condition is no longer met for threshold rules
            if rule.trigger_type == AlertTrigger.THRESHOLD:
                window_start = current_time - timedelta(minutes=5)  # 5 minute resolution window
                recent_data = [
                    (ts, val) for ts, val in self.metric_history[metric_name]
                    if ts >= window_start
                ]

                if recent_data:
                    triggered, _ = self._evaluate_threshold_rule(rule, recent_data)

                    if not triggered:
                        # Resolve alerts
                        for alert in rule_alerts:
                            await self._resolve_alert(alert, current_time, "Automatic resolution - condition no longer met")

        except Exception as e:
            logger.error(f"Alert resolution check failed: {e}")

    async def _resolve_alert(self, alert: Alert, resolved_at: datetime, resolution_note: str = ""):
        """Resolve an alert"""
        try:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = resolved_at
            alert.resolution_note = resolution_note

            # Remove from active alerts
            if alert.id in self.active_alerts:
                del self.active_alerts[alert.id]

            # Update statistics
            if alert.acknowledged_at:
                resolution_time_minutes = (resolved_at - alert.acknowledged_at).total_seconds() / 60
            else:
                resolution_time_minutes = (resolved_at - alert.timestamp).total_seconds() / 60

            # Update average resolution time
            current_avg = self.stats.get('avg_resolution_time_minutes', 0)
            total_resolved = sum(1 for a in self.alert_history if a.resolved_at)

            if total_resolved > 0:
                self.stats['avg_resolution_time_minutes'] = (
                    (current_avg * (total_resolved - 1) + resolution_time_minutes) / total_resolved
                )

            logger.info(f"ALERT RESOLVED: {alert.message} (Resolution: {resolution_note})")

        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")

    def _cleanup_metric_history(self, current_time: datetime):
        """Clean up old metric history"""
        try:
            cutoff_time = current_time - timedelta(hours=24)

            for metric_name in self.metric_history:
                self.metric_history[metric_name] = [
                    (ts, val) for ts, val in self.metric_history[metric_name]
                    if ts > cutoff_time
                ]

            # Clean up alert counters
            for rule_id in self.alert_counters:
                hour_cutoff = current_time - timedelta(hours=1)
                self.alert_counters[rule_id] = [
                    ts for ts in self.alert_counters[rule_id] if ts > hour_cutoff
                ]

            # Clean up old alert history
            if len(self.alert_history) > 10000:
                self.alert_history = self.alert_history[-10000:]

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    # Public API methods

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()

                logger.info(f"Alert acknowledged by {acknowledged_by}: {alert.message}")
                return True
            else:
                logger.warning(f"Alert not found: {alert_id}")
                return False

        except Exception as e:
            logger.error(f"Alert acknowledgment failed: {e}")
            return False

    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Manually resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                asyncio.create_task(self._resolve_alert(alert, datetime.now(), resolution_note))
                return True
            else:
                logger.warning(f"Alert not found: {alert_id}")
                return False

        except Exception as e:
            logger.error(f"Manual alert resolution failed: {e}")
            return False

    def suppress_alert(self, alert_id: str, suppression_time_minutes: int = 60) -> bool:
        """Suppress an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.SUPPRESSED

                # Schedule automatic reactivation
                asyncio.create_task(
                    self._reactivate_alert_after_delay(alert_id, suppression_time_minutes)
                )

                logger.info(f"Alert suppressed for {suppression_time_minutes} minutes: {alert.message}")
                return True
            else:
                logger.warning(f"Alert not found: {alert_id}")
                return False

        except Exception as e:
            logger.error(f"Alert suppression failed: {e}")
            return False

    async def _reactivate_alert_after_delay(self, alert_id: str, delay_minutes: int):
        """Reactivate suppressed alert after delay"""
        try:
            await asyncio.sleep(delay_minutes * 60)

            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                if alert.status == AlertStatus.SUPPRESSED:
                    alert.status = AlertStatus.ACTIVE
                    logger.info(f"Alert reactivated after suppression: {alert.message}")

        except Exception as e:
            logger.error(f"Alert reactivation failed: {e}")

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
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

    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        return [rule.to_dict() for rule in self.alert_rules.values()]

    def get_alerting_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        try:
            stats = self.stats.copy()
            stats.update({
                'alerting_enabled': self.alerting_enabled,
                'active_alerts_count': len(self.active_alerts),
                'total_rules': len(self.alert_rules),
                'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                'last_evaluation': self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
                'notification_channels': list(self.notification_handlers.keys())
            })

            return stats

        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {}

    def enable_alerting(self):
        """Enable the alerting system"""
        self.alerting_enabled = True
        logger.info("Alerting system enabled")

    def disable_alerting(self):
        """Disable the alerting system"""
        self.alerting_enabled = False
        logger.info("Alerting system disabled")

# Convenience functions

async def create_alerting_system() -> AlertingSystem:
    """Create alerting system with default configuration"""
    return AlertingSystem()

# Usage example
if __name__ == "__main__":

    async def test_alerting_system():
        # Create alerting system
        alerting = AlertingSystem()

        # Test rule management
        print(f"Loaded {len(alerting.get_alert_rules())} default rules")

        # Simulate metric data that would trigger alerts
        test_metrics = {
            'cpu_usage': {'value': 90.0, 'unit': 'percent'},
            'memory_usage': {'value': 96.0, 'unit': 'percent'},
            'daily_pnl': {'value': -1500.0, 'unit': 'usd'}
        }

        # Evaluate metrics
        await alerting.evaluate_metrics(test_metrics)

        # Check active alerts
        active_alerts = alerting.get_active_alerts()
        print(f"Active alerts: {len(active_alerts)}")

        for alert in active_alerts:
            print(f"  - {alert['severity'].upper()}: {alert['message']}")

        # Test acknowledgment
        if active_alerts:
            alert_id = active_alerts[0]['id']
            success = alerting.acknowledge_alert(alert_id, "test_user")
            print(f"Alert acknowledgment: {success}")

        # Get statistics
        stats = alerting.get_alerting_statistics()
        print(f"Total alerts generated: {stats['total_alerts_generated']}")
        print(f"Alerts by severity: {stats['alerts_by_severity']}")

        print("Alerting system test completed!")

    # Run test
    asyncio.run(test_alerting_system())
