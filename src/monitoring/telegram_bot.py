"""
Notifications Module
Multi-channel notification system supporting Telegram, Discord, Email, and Slack
Handles message formatting, delivery, and retry logic
"""

import asyncio
import aiohttp
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from config.settings import config
from src.utils.logger import setup_logger
from src.utils.helpers import retry_decorator

logger = setup_logger(__name__)

class NotificationChannel(Enum):
    """Supported notification channels"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass

class NotificationConfig:
    """Configuration for notification channels"""
    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_webhook_url: str = ""

    # Email
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Slack
    slack_webhook_url: str = ""
    slack_channel: str = ""

    # Webhook
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # General settings
    retry_attempts: int = 3
    retry_delay: float = 2.0
    timeout: int = 30

@dataclass

class NotificationMessage:
    """Notification message structure"""
    title: str
    content: str
    priority: NotificationPriority
    channel: NotificationChannel
    timestamp: datetime

    # Optional formatting
    markdown_enabled: bool = True
    emoji_enabled: bool = True

    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)

    # Tracking
    message_id: Optional[str] = None
    sent: bool = False
    sent_at: Optional[datetime] = None
    error: Optional[str] = None

class NotificationService:
    """Multi-channel notification service"""

    def __init__(self):
        # Load configuration
        self.config = NotificationConfig(
            telegram_bot_token=config.get('TELEGRAM_BOT_TOKEN', ''),
            telegram_chat_id=config.get('TELEGRAM_CHAT_ID', ''),
            discord_webhook_url=config.get('DISCORD_WEBHOOK_URL', ''),
            smtp_server=config.get('SMTP_SERVER', ''),
            smtp_port=config.get('SMTP_PORT', 587),
            smtp_username=config.get('SMTP_USERNAME', ''),
            smtp_password=config.get('SMTP_PASSWORD', ''),
            email_from=config.get('EMAIL_FROM', ''),
            email_to=config.get('EMAIL_TO', '').split(',') if config.get('EMAIL_TO') else [],
            slack_webhook_url=config.get('SLACK_WEBHOOK_URL', ''),
            slack_channel=config.get('SLACK_CHANNEL', ''),
            webhook_url=config.get('WEBHOOK_URL', ''),
            webhook_headers=json.loads(config.get('WEBHOOK_HEADERS', '{}')) if config.get('WEBHOOK_HEADERS') else {},
            retry_attempts=config.get('NOTIFICATION_RETRY_ATTEMPTS', 3),
            retry_delay=config.get('NOTIFICATION_RETRY_DELAY', 2.0),
            timeout=config.get('NOTIFICATION_TIMEOUT', 30)
        )

        # Notification history
        self.notification_history: List[NotificationMessage] = []

        # Statistics
        self.stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_channel': {channel.value: {'sent': 0, 'failed': 0} for channel in NotificationChannel},
            'by_priority': {priority.value: {'sent': 0, 'failed': 0} for priority in NotificationPriority}
        }

        # Rate limiting
        self.rate_limits = {
            NotificationChannel.TELEGRAM: {'max_per_minute': 30, 'sent_this_minute': 0, 'minute_start': datetime.now()},
            NotificationChannel.DISCORD: {'max_per_minute': 50, 'sent_this_minute': 0, 'minute_start': datetime.now()},
            NotificationChannel.EMAIL: {'max_per_minute': 10, 'sent_this_minute': 0, 'minute_start': datetime.now()},
            NotificationChannel.SLACK: {'max_per_minute': 100, 'sent_this_minute': 0, 'minute_start': datetime.now()}
        }

        logger.info("Notification service initialized")

    async def send_notification(self, title: str, content: str,
                              channel: NotificationChannel,
                              priority: NotificationPriority = NotificationPriority.NORMAL,
                              **kwargs) -> bool:
        """Send a notification via specified channel"""
        try:
            # Create notification message
            message = NotificationMessage(
                title=title,
                content=content,
                priority=priority,
                channel=channel,
                timestamp=datetime.now(),
                **kwargs
            )

            # Check rate limits
            if not await self._check_rate_limit(channel):
                logger.warning(f"Rate limit exceeded for {channel.value}")
                return False

            # Send via appropriate channel
            success = False

            if channel == NotificationChannel.TELEGRAM:
                success = await self._send_telegram(message)
            elif channel == NotificationChannel.DISCORD:
                success = await self._send_discord(message)
            elif channel == NotificationChannel.EMAIL:
                success = await self._send_email(message)
            elif channel == NotificationChannel.SLACK:
                success = await self._send_slack(message)
            elif channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(message)
            else:
                logger.error(f"Unsupported notification channel: {channel}")
                return False

            # Update message status
            message.sent = success
            if success:
                message.sent_at = datetime.now()
                self.stats['total_sent'] += 1
                self.stats['by_channel'][channel.value]['sent'] += 1
                self.stats['by_priority'][priority.value]['sent'] += 1
            else:
                self.stats['total_failed'] += 1
                self.stats['by_channel'][channel.value]['failed'] += 1
                self.stats['by_priority'][priority.value]['failed'] += 1

            # Store in history
            self.notification_history.append(message)

            # Keep only recent history
            if len(self.notification_history) > 1000:
                self.notification_history = self.notification_history[-1000:]

            return success

        except Exception as e:
            logger.error(f"Notification sending failed: {e}")
            return False

    async def send_alert_notification(self, alert_dict: Dict[str, Any],
                                    rule_dict: Dict[str, Any]) -> Dict[str, bool]:
        """Send alert notification via all configured channels"""
        try:
            results = {}

            # Format alert message
            title = f"🚨 {alert_dict['severity'].upper()} Alert"
            content = self._format_alert_message(alert_dict, rule_dict)

            # Determine priority based on severity
            severity_to_priority = {
                'low': NotificationPriority.LOW,
                'medium': NotificationPriority.NORMAL,
                'high': NotificationPriority.HIGH,
                'critical': NotificationPriority.URGENT
            }
            priority = severity_to_priority.get(alert_dict['severity'], NotificationPriority.NORMAL)

            # Send via configured channels
            for channel_name in rule_dict.get('notification_channels', []):
                try:
                    channel = NotificationChannel(channel_name)
                    success = await self.send_notification(title, content, channel, priority)
                    results[channel_name] = success

                    if success:
                        logger.info(f"Alert notification sent via {channel_name}")
                    else:
                        logger.error(f"Alert notification failed via {channel_name}")

                except ValueError:
                    logger.warning(f"Unknown notification channel: {channel_name}")
                    results[channel_name] = False
                except Exception as e:
                    logger.error(f"Alert notification error via {channel_name}: {e}")
                    results[channel_name] = False

            return results

        except Exception as e:
            logger.error(f"Alert notification sending failed: {e}")
            return {}

    def _format_alert_message(self, alert_dict: Dict[str, Any],
                            rule_dict: Dict[str, Any]) -> str:
        """Format alert message for notifications"""
        try:
            severity_emojis = {
                'low': '🔵',
                'medium': '🟡',
                'high': '🟠',
                'critical': '🔴'
            }

            emoji = severity_emojis.get(alert_dict['severity'], '⚠️')

            message_parts = [
                f"{emoji} **{alert_dict['severity'].upper()} ALERT**",
                "",
                f"**Metric:** {alert_dict['metric_name']}",
                f"**Value:** {alert_dict['value']:.2f}",
                f"**Message:** {alert_dict['message']}",
                f"**Time:** {alert_dict['timestamp']}",
                ""
            ]

            # Add threshold if available
            if alert_dict.get('threshold') is not None:
                message_parts.insert(-1, f"**Threshold:** {alert_dict['threshold']:.2f}")

            # Add rule information
            message_parts.extend([
                f"**Rule:** {rule_dict['name']}",
                f"**Description:** {rule_dict['description']}"
            ])

            # Add context if available
            if alert_dict.get('context'):
                context = alert_dict['context']
                if isinstance(context, dict):
                    message_parts.append("")
                    message_parts.append("**Additional Info:**")
                    for key, value in context.items():
                        if key not in ['rule_name', 'trigger_type', 'condition']:
                            message_parts.append(f"• {key}: {value}")

            return "\n".join(message_parts)

        except Exception as e:
            logger.error(f"Alert message formatting failed: {e}")
            return f"Alert: {alert_dict.get('message', 'Unknown alert')}"

    async def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if we can send notification without exceeding rate limit"""
        try:
            if channel not in self.rate_limits:
                return True

            rate_limit = self.rate_limits[channel]
            current_time = datetime.now()

            # Reset counter if a minute has passed
            if (current_time - rate_limit['minute_start']).total_seconds() >= 60:
                rate_limit['sent_this_minute'] = 0
                rate_limit['minute_start'] = current_time

            # Check if we're under the limit
            if rate_limit['sent_this_minute'] >= rate_limit['max_per_minute']:
                return False

            # Increment counter
            rate_limit['sent_this_minute'] += 1
            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow sending on error

    @retry_decorator(max_retries=3, delay=2.0)

    async def _send_telegram(self, message: NotificationMessage) -> bool:
        """Send notification via Telegram"""
        try:
            if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
                logger.debug("Telegram credentials not configured")
                return False

            # Format message for Telegram
            text = f"*{message.title}*\n\n{message.content}"

            # Add priority indicator
            priority_emojis = {
                NotificationPriority.LOW: '🔵',
                NotificationPriority.NORMAL: '⚪',
                NotificationPriority.HIGH: '🟠',
                NotificationPriority.URGENT: '🔴'
            }
            text = f"{priority_emojis.get(message.priority, '')} {text}"

            # Prepare request
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"

            payload = {
                'chat_id': self.config.telegram_chat_id,
                'text': text,
                'parse_mode': 'Markdown' if message.markdown_enabled else None
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        message.message_id = str(response_data.get('result', {}).get('message_id', ''))
                        logger.debug(f"Telegram message sent successfully: {message.message_id}")
                        return True
                    else:
                        error_text = await response.text()
                        message.error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Telegram send failed: {message.error}")
                        return False

        except Exception as e:
            message.error = str(e)
            logger.error(f"Telegram notification failed: {e}")
            return False

    @retry_decorator(max_retries=3, delay=2.0)

    async def _send_discord(self, message: NotificationMessage) -> bool:
        """Send notification via Discord webhook"""
        try:
            if not self.config.discord_webhook_url:
                logger.debug("Discord webhook not configured")
                return False

            # Format message for Discord
            embed = {
                "title": message.title,
                "description": message.content,
                "timestamp": message.timestamp.isoformat(),
                "color": self._get_priority_color(message.priority)
            }

            # Add priority field
            embed["fields"] = [
                {
                    "name": "Priority",
                    "value": message.priority.value.capitalize(),
                    "inline": True
                }
            ]

            payload = {"embeds": [embed]}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(self.config.discord_webhook_url, json=payload) as response:
                    if response.status in [200, 204]:
                        logger.debug("Discord message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        message.error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Discord send failed: {message.error}")
                        return False

        except Exception as e:
            message.error = str(e)
            logger.error(f"Discord notification failed: {e}")
            return False

    def _get_priority_color(self, priority: NotificationPriority) -> int:
        """Get color code for priority level"""
        colors = {
            NotificationPriority.LOW: 0x00FF00,      # Green
            NotificationPriority.NORMAL: 0x0099FF,   # Blue
            NotificationPriority.HIGH: 0xFF9900,     # Orange
            NotificationPriority.URGENT: 0xFF0000    # Red
        }
        return colors.get(priority, 0x0099FF)

    @retry_decorator(max_retries=3, delay=2.0)

    async def _send_email(self, message: NotificationMessage) -> bool:
        """Send notification via email"""
        try:
            if not all([self.config.smtp_server, self.config.smtp_username,
                       self.config.smtp_password, self.config.email_from]):
                logger.debug("Email configuration incomplete")
                return False

            if not self.config.email_to:
                logger.debug("No email recipients configured")
                return False

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[{message.priority.value.upper()}] {message.title}"

            # Email body
            body = f"""
Priority: {message.priority.value.capitalize()}
Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{message.content}

---
Trading System Notification
"""

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.debug("Email sent successfully")
            return True

        except Exception as e:
            message.error = str(e)
            logger.error(f"Email notification failed: {e}")
            return False

    @retry_decorator(max_retries=3, delay=2.0)

    async def _send_slack(self, message: NotificationMessage) -> bool:
        """Send notification via Slack webhook"""
        try:
            if not self.config.slack_webhook_url:
                logger.debug("Slack webhook not configured")
                return False

            # Format message for Slack
            priority_emojis = {
                NotificationPriority.LOW: ':large_blue_circle:',
                NotificationPriority.NORMAL: ':white_circle:',
                NotificationPriority.HIGH: ':large_orange_circle:',
                NotificationPriority.URGENT: ':red_circle:'
            }

            text = f"{priority_emojis.get(message.priority, '')} *{message.title}*\n{message.content}"

            payload = {
                "text": text,
                "channel": self.config.slack_channel if self.config.slack_channel else None,
                "attachments": [
                    {
                        "color": "good" if message.priority == NotificationPriority.LOW else
                               "warning" if message.priority == NotificationPriority.HIGH else
                               "danger" if message.priority == NotificationPriority.URGENT else "  # 439FE0",
                        "fields": [
                            {
                                "title": "Priority",
                                "value": message.priority.value.capitalize(),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": message.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ]
                    }
                ]
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(self.config.slack_webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.debug("Slack message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        message.error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Slack send failed: {message.error}")
                        return False

        except Exception as e:
            message.error = str(e)
            logger.error(f"Slack notification failed: {e}")
            return False

    @retry_decorator(max_retries=3, delay=2.0)

    async def _send_webhook(self, message: NotificationMessage) -> bool:
        """Send notification via generic webhook"""
        try:
            if not self.config.webhook_url:
                logger.debug("Webhook URL not configured")
                return False

            # Prepare payload
            payload = {
                "title": message.title,
                "content": message.content,
                "priority": message.priority.value,
                "timestamp": message.timestamp.isoformat(),
                "tags": message.tags
            }

            headers = {
                'Content-Type': 'application/json',
                **self.config.webhook_headers
            }

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(self.config.webhook_url, json=payload, headers=headers) as response:
                    if response.status in [200, 201, 202, 204]:
                        logger.debug("Webhook message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        message.error = f"HTTP {response.status}: {error_text}"
                        logger.error(f"Webhook send failed: {message.error}")
                        return False

        except Exception as e:
            message.error = str(e)
            logger.error(f"Webhook notification failed: {e}")
            return False

    async def send_system_status(self, status_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send system status notification"""
        try:
            title = f"📊 System Status Update"

            # Format status message
            status_parts = [
                f"**System Health:** {status_data.get('status', 'unknown').upper()}",
                f"**Portfolio Value:** ${status_data.get('portfolio_value', 0):,.2f}",
                f"**Active Alerts:** {status_data.get('active_alerts', 0)}",
                f"**Uptime:** {status_data.get('uptime_hours', 0):.1f} hours",
                f"**Last Update:** {status_data.get('timestamp', datetime.now().isoformat())}"
            ]

            content = "\n".join(status_parts)

            # Determine priority based on system health
            priority = NotificationPriority.LOW
            if status_data.get('status') == 'warning':
                priority = NotificationPriority.NORMAL
            elif status_data.get('status') in ['error', 'critical']:
                priority = NotificationPriority.HIGH

            # Send via configured channels
            channels = [NotificationChannel.TELEGRAM, NotificationChannel.EMAIL]
            results = {}

            for channel in channels:
                try:
                    success = await self.send_notification(title, content, channel, priority)
                    results[channel.value] = success
                except Exception as e:
                    logger.error(f"System status notification failed via {channel.value}: {e}")
                    results[channel.value] = False

            return results

        except Exception as e:
            logger.error(f"System status notification failed: {e}")
            return {}

    async def send_daily_report(self, report_data: Dict[str, Any]) -> Dict[str, bool]:
        """Send daily trading report"""
        try:
            title = f"📈 Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"

            # Format report message
            report_parts = [
                "**Daily Performance Summary:**",
                "",
                f"💰 **P&L Today:** ${report_data.get('daily_pnl', 0):,.2f}",
                f"💼 **Portfolio Value:** ${report_data.get('portfolio_value', 0):,.2f}",
                f"📊 **Total Return:** {report_data.get('total_return_pct', 0):.2f}%",
                "",
                f"📈 **Trades Today:** {report_data.get('trades_today', 0)}",
                f"🎯 **Win Rate:** {report_data.get('win_rate', 0):.1f}%",
                f"📍 **Open Positions:** {report_data.get('open_positions', 0)}",
                "",
                f"⚡ **Signals Generated:** {report_data.get('signals_today', 0)}",
                f"⚠️ **Active Alerts:** {report_data.get('active_alerts', 0)}",
                f"🔄 **System Uptime:** {report_data.get('uptime_hours', 0):.1f}h"
            ]

            # Add performance metrics if available
            if report_data.get('sharpe_ratio'):
                report_parts.extend([
                    "",
                    "**Risk Metrics:**",
                    f"📊 **Sharpe Ratio:** {report_data['sharpe_ratio']:.2f}",
                    f"📉 **Max Drawdown:** {report_data.get('max_drawdown', 0):.2f}%"
                ])

            content = "\n".join(report_parts)

            # Send via email and Telegram
            results = {}
            channels = [NotificationChannel.EMAIL, NotificationChannel.TELEGRAM]

            for channel in channels:
                try:
                    success = await self.send_notification(
                        title, content, channel, NotificationPriority.NORMAL
                    )
                    results[channel.value] = success
                except Exception as e:
                    logger.error(f"Daily report notification failed via {channel.value}: {e}")
                    results[channel.value] = False

            return results

        except Exception as e:
            logger.error(f"Daily report notification failed: {e}")
            return {}

    def get_notification_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get notification history"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            history = [
                {
                    'title': msg.title,
                    'content': msg.content[:100] + '...' if len(msg.content) > 100 else msg.content,
                    'channel': msg.channel.value,
                    'priority': msg.priority.value,
                    'timestamp': msg.timestamp.isoformat(),
                    'sent': msg.sent,
                    'sent_at': msg.sent_at.isoformat() if msg.sent_at else None,
                    'error': msg.error
                }
                for msg in self.notification_history
                if msg.timestamp > cutoff_time
            ]

            return sorted(history, key=lambda x: x['timestamp'], reverse=True)

        except Exception as e:
            logger.error(f"Notification history retrieval failed: {e}")
            return []

    def get_notification_statistics(self) -> Dict[str, Any]:
        """Get notification statistics"""
        try:
            stats = self.stats.copy()

            # Add success rates
            stats['success_rate'] = (
                stats['total_sent'] / (stats['total_sent'] + stats['total_failed'])
                if (stats['total_sent'] + stats['total_failed']) > 0 else 0
            ) * 100

            # Channel success rates
            for channel, channel_stats in stats['by_channel'].items():
                total = channel_stats['sent'] + channel_stats['failed']
                channel_stats['success_rate'] = (
                    channel_stats['sent'] / total * 100 if total > 0 else 0
                )

            # Priority success rates
            for priority, priority_stats in stats['by_priority'].items():
                total = priority_stats['sent'] + priority_stats['failed']
                priority_stats['success_rate'] = (
                    priority_stats['sent'] / total * 100 if total > 0 else 0
                )

            return stats

        except Exception as e:
            logger.error(f"Statistics generation failed: {e}")
            return {}

    def test_channels(self) -> Dict[str, bool]:
        """Test all configured notification channels"""

        async def test_all_channels():
            test_results = {}
            test_message = "🧪 Test notification from Trading System"

            channels_to_test = []

            # Check which channels are configured
            if self.config.telegram_bot_token and self.config.telegram_chat_id:
                channels_to_test.append(NotificationChannel.TELEGRAM)

            if self.config.discord_webhook_url:
                channels_to_test.append(NotificationChannel.DISCORD)

            if (self.config.smtp_server and self.config.smtp_username and
                self.config.smtp_password and self.config.email_to):
                channels_to_test.append(NotificationChannel.EMAIL)

            if self.config.slack_webhook_url:
                channels_to_test.append(NotificationChannel.SLACK)

            if self.config.webhook_url:
                channels_to_test.append(NotificationChannel.WEBHOOK)

            # Test each channel
            for channel in channels_to_test:
                try:
                    success = await self.send_notification(
                        "Test Notification",
                        test_message,
                        channel,
                        NotificationPriority.LOW
                    )
                    test_results[channel.value] = success
                except Exception as e:
                    logger.error(f"Channel test failed for {channel.value}: {e}")
                    test_results[channel.value] = False

            return test_results

        # Run the async test
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(test_all_channels())
        except RuntimeError:
            # If no event loop is running, create one
            return asyncio.run(test_all_channels())

# Convenience functions

async def send_quick_notification(title: str, content: str,
                                channel: str = "telegram",
                                priority: str = "normal") -> bool:
    """Quick notification sending function"""
    service = NotificationService()

    try:
        channel_enum = NotificationChannel(channel)
        priority_enum = NotificationPriority(priority)

        return await service.send_notification(title, content, channel_enum, priority_enum)
    except ValueError as e:
        logger.error(f"Invalid channel or priority: {e}")
        return False

def create_notification_service() -> NotificationService:
    """Create notification service instance"""
    return NotificationService()

# Usage example
if __name__ == "__main__":

    async def test_notifications():
        # Create notification service
        service = NotificationService()

        # Test channel configuration
        print("Testing notification channels...")
        test_results = service.test_channels()

        for channel, success in test_results.items():
            status = "✅" if success else "❌"
            print(f"{status} {channel}: {'Working' if success else 'Failed'}")

        # Test alert notification
        sample_alert = {
            'id': 'test_alert',
            'severity': 'high',
            'message': 'CPU usage is high: 90%',
            'metric_name': 'cpu_usage',
            'value': 90.0,
            'threshold': 80.0,
            'timestamp': datetime.now().isoformat()
        }

        sample_rule = {
            'name': 'High CPU Usage',
            'description': 'CPU usage exceeds threshold',
            'notification_channels': ['telegram']
        }

        print("\nTesting alert notification...")
        alert_results = await service.send_alert_notification(sample_alert, sample_rule)

        for channel, success in alert_results.items():
            status = "✅" if success else "❌"
            print(f"{status} Alert via {channel}: {'Sent' if success else 'Failed'}")

        # Get statistics
        stats = service.get_notification_statistics()
        print(f"\nNotification Statistics:")
        print(f"Total sent: {stats['total_sent']}")
        print(f"Total failed: {stats['total_failed']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")

        print("Notification service test completed!")

    # Run test
    asyncio.run(test_notifications())
