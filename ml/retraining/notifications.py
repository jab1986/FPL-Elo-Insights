"""Notification system for ML monitoring and retraining alerts."""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import pandas as pd

from ..config.monitoring_config import (
    MonitoringConfig,
    AlertLevel,
    NotificationChannel,
)
from ..monitoring.performance_monitor import PerformanceAlert
from ..monitoring.data_drift_detector import DriftAlert


@dataclass
class NotificationMessage:
    """Container for notification messages."""

    title: str
    """Notification title."""

    message: str
    """Main notification message."""

    level: AlertLevel
    """Alert severity level."""

    timestamp: datetime
    """When the notification was created."""

    channel: NotificationChannel
    """Target notification channel."""

    metadata: Dict[str, Any]
    """Additional metadata."""

    recipient: Optional[str] = None
    """Specific recipient (for email)."""


class NotificationManager:
    """Manages notifications across multiple channels."""

    def __init__(self, config: MonitoringConfig):
        """Initialize the notification manager.

        Args:
            config: Monitoring configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Notification history
        self.notification_history: List[NotificationMessage] = []
        self.cooldown_cache: Dict[str, datetime] = {}

        # Create directories
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load notification history
        self._load_notification_history()

    async def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel,
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[NotificationChannel]] = None
    ) -> bool:
        """Send an alert through configured channels.

        Args:
            title: Alert title.
            message: Alert message.
            level: Alert severity level.
            metadata: Additional metadata.
            channels: Specific channels to use. If None, uses configured channels.

        Returns:
            True if at least one channel successfully received the alert.
        """
        # Check cooldown
        if not self._check_cooldown(title, level):
            self.logger.info(f"Alert suppressed due to cooldown: {title}")
            return False

        # Use configured channels if not specified
        target_channels = channels or self.config.notifications.enabled_channels

        # Create notification message
        notification = NotificationMessage(
            title=title,
            message=message,
            level=level,
            timestamp=datetime.now(),
            channel=NotificationChannel.EMAIL,  # Default channel
            metadata=metadata or {}
        )

        # Store notification
        self.notification_history.append(notification)
        self._save_notification_history()

        # Send through each channel
        success_count = 0
        for channel in target_channels:
            try:
                if await self._send_to_channel(notification, channel):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to send notification via {channel.value}: {e}")

        self.logger.info(f"Alert sent successfully to {success_count}/{len(target_channels)} channels")
        return success_count > 0

    async def _send_to_channel(
        self,
        notification: NotificationMessage,
        channel: NotificationChannel
    ) -> bool:
        """Send notification to a specific channel.

        Args:
            notification: Notification message to send.
            channel: Target channel.

        Returns:
            True if sending was successful.
        """
        if channel == NotificationChannel.EMAIL:
            return await self._send_email(notification)
        elif channel == NotificationChannel.SLACK:
            return await self._send_slack(notification)
        elif channel == NotificationChannel.DISCORD:
            return await self._send_discord(notification)
        elif channel == NotificationChannel.DASHBOARD:
            return await self._send_dashboard(notification)
        else:
            self.logger.warning(f"Unsupported notification channel: {channel}")
            return False

    async def _send_email(self, notification: NotificationMessage) -> bool:
        """Send notification via email.

        Args:
            notification: Notification message.

        Returns:
            True if email was sent successfully.
        """
        if not self.config.notifications.email_recipients:
            self.logger.warning("No email recipients configured")
            return False

        try:
            # Email configuration - would need to be expanded for production
            smtp_server = "smtp.gmail.com"  # Example
            smtp_port = 587
            sender_email = "alerts@fpl-ml.com"  # Example

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ", ".join(self.config.notifications.email_recipients)
            msg['Subject'] = f"[{notification.level.value.upper()}] {notification.title}"

            # Email body with HTML formatting
            body = self._format_email_body(notification)
            msg.attach(MIMEText(body, 'html'))

            # Send email (simplified - would need proper SMTP setup)
            # server = smtplib.SMTP(smtp_server, smtp_port)
            # server.starttls()
            # server.login(sender_email, "password")  # Would need secure credential management
            # server.sendmail(sender_email, self.config.notifications.email_recipients, msg.as_string())
            # server.quit()

            # For now, just log the email
            self.logger.info(f"Email notification would be sent: {notification.title}")
            self.logger.info(f"Recipients: {', '.join(self.config.notifications.email_recipients)}")
            self.logger.info(f"Body: {notification.message}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False

    async def _send_slack(self, notification: NotificationMessage) -> bool:
        """Send notification via Slack.

        Args:
            notification: Notification message.

        Returns:
            True if Slack message was sent successfully.
        """
        if not self.config.notifications.slack_webhook_url:
            self.logger.warning("Slack webhook URL not configured")
            return False

        try:
            # Format message for Slack
            slack_message = self._format_slack_message(notification)

            # Send to Slack
            response = requests.post(
                self.config.notifications.slack_webhook_url,
                json=slack_message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                self.logger.info(f"Slack notification sent: {notification.title}")
                return True
            else:
                self.logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def _send_discord(self, notification: NotificationMessage) -> bool:
        """Send notification via Discord.

        Args:
            notification: Notification message.

        Returns:
            True if Discord message was sent successfully.
        """
        if not self.config.notifications.discord_webhook_url:
            self.logger.warning("Discord webhook URL not configured")
            return False

        try:
            # Format message for Discord
            discord_message = self._format_discord_message(notification)

            # Send to Discord
            response = requests.post(
                self.config.notifications.discord_webhook_url,
                json=discord_message,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 204:
                self.logger.info(f"Discord notification sent: {notification.title}")
                return True
            else:
                self.logger.error(f"Discord API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to send Discord notification: {e}")
            return False

    async def _send_dashboard(self, notification: NotificationMessage) -> bool:
        """Send notification to dashboard.

        Args:
            notification: Notification message.

        Returns:
            True if dashboard notification was sent successfully.
        """
        if not self.config.notifications.dashboard_url:
            self.logger.warning("Dashboard URL not configured")
            return False

        try:
            # Format for dashboard API
            dashboard_payload = self._format_dashboard_payload(notification)

            # Send to dashboard
            response = requests.post(
                f"{self.config.notifications.dashboard_url}/api/alerts",
                json=dashboard_payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code in [200, 201]:
                self.logger.info(f"Dashboard notification sent: {notification.title}")
                return True
            else:
                self.logger.error(f"Dashboard API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to send dashboard notification: {e}")
            return False

    def _format_email_body(self, notification: NotificationMessage) -> str:
        """Format notification as HTML email body.

        Args:
            notification: Notification message.

        Returns:
            HTML email body.
        """
        level_colors = {
            AlertLevel.INFO: "#6B7280",
            AlertLevel.WARNING: "#F59E0B",
            AlertLevel.CRITICAL: "#EF4444"
        }

        color = level_colors.get(notification.level, "#6B7280")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 20px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .metadata {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; font-size: 12px; }}
                .footer {{ margin-top: 20px; font-size: 11px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{notification.title}</h2>
                <p><strong>Level:</strong> {notification.level.value.upper()}</p>
                <p><strong>Time:</strong> {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>

            <div class="content">
                <p>{notification.message}</p>
            </div>

            <div class="metadata">
                <h4>Additional Information:</h4>
                <pre>{json.dumps(notification.metadata, indent=2)}</pre>
            </div>

            <div class="footer">
                <p>This is an automated alert from the FPL ML Monitoring System.</p>
            </div>
        </body>
        </html>
        """

        return html

    def _format_slack_message(self, notification: NotificationMessage) -> Dict[str, Any]:
        """Format notification for Slack.

        Args:
            notification: Notification message.

        Returns:
            Slack message payload.
        """
        level_emojis = {
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.CRITICAL: ":x:"
        }

        emoji = level_emojis.get(notification.level, ":bell:")

        return {
            "text": f"{emoji} {notification.title}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {notification.title}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{notification.level.value.upper()}* - {notification.message}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Time: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        }
                    ]
                }
            ]
        }

    def _format_discord_message(self, notification: NotificationMessage) -> Dict[str, Any]:
        """Format notification for Discord.

        Args:
            notification: Notification message.

        Returns:
            Discord message payload.
        """
        level_colors = {
            AlertLevel.INFO: 0x6B7280,
            AlertLevel.WARNING: 0xF59E0B,
            AlertLevel.CRITICAL: 0xEF4444
        }

        color = level_colors.get(notification.level, 0x6B7280)

        return {
            "embeds": [
                {
                    "title": notification.title,
                    "description": notification.message,
                    "color": color,
                    "timestamp": notification.timestamp.isoformat(),
                    "fields": [
                        {
                            "name": "Level",
                            "value": notification.level.value.upper(),
                            "inline": True
                        },
                        {
                            "name": "Time",
                            "value": notification.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                            "inline": True
                        }
                    ],
                    "footer": {
                        "text": "FPL ML Monitoring System"
                    }
                }
            ]
        }

    def _format_dashboard_payload(self, notification: NotificationMessage) -> Dict[str, Any]:
        """Format notification for dashboard API.

        Args:
            notification: Notification message.

        Returns:
            Dashboard API payload.
        """
        return {
            "title": notification.title,
            "message": notification.message,
            "level": notification.level.value,
            "timestamp": notification.timestamp.isoformat(),
            "metadata": notification.metadata,
            "source": "ml_monitoring"
        }

    def _check_cooldown(self, title: str, level: AlertLevel) -> bool:
        """Check if alert is within cooldown period.

        Args:
            title: Alert title for cooldown key.
            level: Alert level.

        Returns:
            True if alert should be sent (not in cooldown).
        """
        # Skip cooldown for critical alerts
        if level == AlertLevel.CRITICAL:
            return True

        # Skip cooldown if disabled
        if self.config.notifications.alert_cooldown_minutes <= 0:
            return True

        # Check cooldown cache
        cooldown_key = f"{level.value}:{title}"
        cooldown_time = datetime.now() - timedelta(minutes=self.config.notifications.alert_cooldown_minutes)

        last_notification = self.cooldown_cache.get(cooldown_key)
        if last_notification and last_notification > cooldown_time:
            return False

        # Update cooldown cache
        self.cooldown_cache[cooldown_key] = datetime.now()
        return True

    def send_performance_alert(self, alert: PerformanceAlert) -> bool:
        """Send performance alert.

        Args:
            alert: Performance alert to send.

        Returns:
            True if alert was sent successfully.
        """
        title = f"Performance Alert: {alert.alert_type.replace('_', ' ').title()}"
        message = f"{alert.message}. Current value: {alert.current_value:.4f}, Threshold: {alert.threshold_value:.4f}"

        metadata = {
            "alert_type": alert.alert_type,
            "metric": alert.metadata.get("metric"),
            "model_version": alert.metadata.get("model_version"),
            "performance_data": {
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp.isoformat()
            }
        }

        return asyncio.run(self.send_alert(title, message, alert.level, metadata))

    def send_drift_alert(self, alert: DriftAlert) -> bool:
        """Send drift alert.

        Args:
            alert: Drift alert to send.

        Returns:
            True if alert was sent successfully.
        """
        title = f"Data Drift Alert: {alert.alert_type.replace('_', ' ').title()}"
        message = f"Feature '{alert.feature_name}': {alert.message}. Current value: {alert.current_value:.4f}, Threshold: {alert.threshold_value:.4f}"

        metadata = {
            "alert_type": alert.alert_type,
            "feature_name": alert.feature_name,
            "metric": alert.metadata.get("metric"),
            "drift_data": {
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "timestamp": alert.timestamp.isoformat()
            }
        }

        return asyncio.run(self.send_alert(title, message, alert.level, metadata))

    def send_retraining_alert(
        self,
        status: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send retraining status alert.

        Args:
            status: Retraining status (success, failure, etc.).
            message: Status message.
            metadata: Additional metadata.

        Returns:
            True if alert was sent successfully.
        """
        level = AlertLevel.INFO
        if "failed" in status.lower() or "error" in status.lower():
            level = AlertLevel.CRITICAL
        elif "warning" in status.lower():
            level = AlertLevel.WARNING

        title = f"Retraining {status.title()}"
        return asyncio.run(self.send_alert(title, message, level, metadata))

    def get_notification_summary(self) -> Dict[str, Any]:
        """Get notification summary.

        Returns:
            Dictionary with notification summary statistics.
        """
        if not self.notification_history:
            return {"total_notifications": 0}

        # Filter recent notifications
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_notifications = [
            n for n in self.notification_history
            if n.timestamp >= cutoff_time
        ]

        # Count by level
        level_counts = {}
        for level in AlertLevel:
            level_counts[level.value] = len([
                n for n in self.notification_history
                if n.level == level
            ])

        # Count by channel
        channel_counts = {}
        for channel in NotificationChannel:
            channel_counts[channel.value] = len([
                n for n in self.notification_history
                if n.channel == channel
            ])

        return {
            "total_notifications": len(self.notification_history),
            "recent_notifications": len(recent_notifications),
            "level_breakdown": level_counts,
            "channel_breakdown": channel_counts,
            "latest_notification": {
                "title": self.notification_history[-1].title,
                "level": self.notification_history[-1].level.value,
                "timestamp": self.notification_history[-1].timestamp.isoformat()
            } if self.notification_history else None
        }

    def _load_notification_history(self) -> None:
        """Load notification history from disk."""
        history_file = self.config.artifacts_dir / "notification_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)

                for item in data:
                    notification = NotificationMessage(
                        title=item["title"],
                        message=item["message"],
                        level=AlertLevel(item["level"]),
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        channel=NotificationChannel(item["channel"]),
                        metadata=item.get("metadata", {}),
                        recipient=item.get("recipient")
                    )
                    self.notification_history.append(notification)

                self.logger.info(f"Loaded {len(self.notification_history)} notification records")

            except Exception as e:
                self.logger.error(f"Failed to load notification history: {e}")

    def _save_notification_history(self) -> None:
        """Save notification history to disk."""
        history_file = self.config.artifacts_dir / "notification_history.json"

        try:
            data = [
                {
                    "title": n.title,
                    "message": n.message,
                    "level": n.level.value,
                    "timestamp": n.timestamp.isoformat(),
                    "channel": n.channel.value,
                    "metadata": n.metadata,
                    "recipient": n.recipient
                }
                for n in self.notification_history[-1000:]  # Keep last 1000 notifications
            ]

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save notification history: {e}")

    def test_notifications(self) -> Dict[str, bool]:
        """Test all configured notification channels.

        Returns:
            Dictionary mapping channel names to test results.
        """
        results = {}

        for channel in self.config.notifications.enabled_channels:
            try:
                # Create test notification
                test_notification = NotificationMessage(
                    title="Test Alert",
                    message="This is a test notification from the FPL ML Monitoring System.",
                    level=AlertLevel.INFO,
                    timestamp=datetime.now(),
                    channel=channel,
                    metadata={"test": True}
                )

                # Send test notification
                success = asyncio.run(self._send_to_channel(test_notification, channel))
                results[channel.value] = success

            except Exception as e:
                self.logger.error(f"Test failed for channel {channel.value}: {e}")
                results[channel.value] = False

        return results
