#!/usr/bin/env python3
"""
Alert Handler for Gateway - Manages alert distribution via multiple channels
"""

import json
import requests
import logging
from typing import Dict, List
from datetime import datetime
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

class AlertHandler:
    def __init__(self, config_file: str = "alert_config.json"):
        self.config = self.load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.alert_history = []
        
    def load_config(self, config_file: str) -> Dict:
        """Load alert configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "sms_gateway": {
                    "enabled": True,
                    "url": "http://sms-gateway/api/send",
                    "api_key": "your_api_key"
                },
                "email": {
                    "enabled": True,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "alerts@cloudburst.com",
                    "password": "your_password"
                },
                "push_notifications": {
                    "enabled": True,
                    "service_key": "your_service_key"
                },
                "siren_control": {
                    "enabled": True,
                    "gpio_pin": 18
                }
            }
    
    def send_alert(self, alert_data: Dict):
        """Send alert through all configured channels"""
        self.logger.info(f"Sending alert: {alert_data}")
        
        # Add to history
        alert_data['sent_time'] = datetime.now().isoformat()
        self.alert_history.append(alert_data)
        
        # Send via SMS
        if self.config["sms_gateway"]["enabled"]:
            self.send_sms_alert(alert_data)
        
        # Send via Email
        if self.config["email"]["enabled"]:
            self.send_email_alert(alert_data)
        
        # Send push notifications
        if self.config["push_notifications"]["enabled"]:
            self.send_push_notification(alert_data)
        
        # Activate local sirens for critical alerts
        if (self.config["siren_control"]["enabled"] and 
            alert_data.get('level') in ['CRITICAL', 'SYSTEM_CRITICAL']):
            self.activate_siren(alert_data)
    
    def send_sms_alert(self, alert_data: Dict):
        """Send SMS alert to registered numbers"""
        try:
            message = self.format_sms_message(alert_data)
            
            # This would integrate with actual SMS gateway
            response = requests.post(
                self.config["sms_gateway"]["url"],
                json={
                    "to": self.config["sms_gateway"]["recipients"],
                    "message": message,
                    "priority": "high" if alert_data['level'] in ['CRITICAL', 'SYSTEM_CRITICAL'] else "normal"
                },
                headers={"Authorization": f"Bearer {self.config['sms_gateway']['api_key']}"}
            )
            
            if response.status_code == 200:
                self.logger.info("SMS alert sent successfully")
            else:
                self.logger.error(f"SMS alert failed: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Failed to send SMS alert: {e}")
    
    def send_email_alert(self, alert_data: Dict):
        """Send email alert to registered addresses"""
        try:
            message = MimeMultipart()
            message['Subject'] = f"Cloudburst Alert - {alert_data['level']}"
            message['From'] = self.config["email"]["username"]
            message['To'] = ", ".join(self.config["email"]["recipients"])
            
            # Create HTML email body
            html_body = self.format_email_message(alert_data)
            message.attach(MimeText(html_body, 'html'))
            
            with smtplib.SMTP(self.config["email"]["smtp_server"], self.config["email"]["smtp_port"]) as server:
                server.starttls()
                server.login(self.config["email"]["username"], self.config["email"]["password"])
                server.send_message(message)
            
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def send_push_notification(self, alert_data: Dict):
        """Send push notification to mobile apps"""
        try:
            # This would integrate with Firebase Cloud Messaging or similar
            push_data = {
                "to": "/topics/cloudburst_alerts",
                "notification": {
                    "title": f"Cloudburst {alert_data['level']} Alert",
                    "body": self.format_push_message(alert_data),
                    "sound": "default",
                    "priority": "high"
                },
                "data": {
                    "alert_level": alert_data['level'],
                    "risk_score": str(alert_data.get('risk_score', 0)),
                    "location": alert_data.get('location', 'Unknown'),
                    "timestamp": alert_data.get('timestamp')
                }
            }
            
            # Simulated push notification
            self.logger.info(f"Push notification prepared: {push_data}")
            
        except Exception as e:
            self.logger.error(f"Failed to send push notification: {e}")
    
    def activate_siren(self, alert_data: Dict):
        """Activate local warning sirens"""
        try:
            # This would control physical sirens via GPIO
            self.logger.info(f"Activating sirens for alert: {alert_data['level']}")
            
            # Simulated siren activation
            # In actual implementation, this would use RPi.GPIO or similar
            siren_pattern = "INTERMITTENT_HIGH" if alert_data['level'] == 'SYSTEM_CRITICAL' else "INTERMITTENT"
            self.logger.info(f"Siren pattern: {siren_pattern}")
            
        except Exception as e:
            self.logger.error(f"Failed to activate siren: {e}")
    
    def format_sms_message(self, alert_data: Dict) -> str:
        """Format alert message for SMS"""
        level = alert_data['level']
        risk_score = alert_data.get('risk_score', 0)
        location = alert_data.get('location', 'Unknown location')
        
        if level == 'SYSTEM_CRITICAL':
            return (f"🚨 CLOUDBURST EMERGENCY 🚨\n"
                   f"Multiple sensors detecting extreme conditions\n"
                   f"Location: {location}\n"
                   f"Risk: {risk_score:.1%}\n"
                   f"Take immediate safety measures!")
        
        elif level == 'CRITICAL':
            return (f"⚠️ CLOUDBURST WARNING ⚠️\n"
                   f"Imminent cloudburst detected\n"
                   f"Location: {location}\n"
                   f"Risk: {risk_score:.1%}\n"
                   f"Seek shelter immediately!")
        
        else:  # WARNING
            return (f"🔶 Cloudburst Alert\n"
                   f"Potential cloudburst conditions\n"
                   f"Location: {location}\n"
                   f"Risk: {risk_score:.1%}\n"
                   f"Stay alert and monitor updates")
    
    def format_email_message(self, alert_data: Dict) -> str:
        """Format alert message for email (HTML)"""
        level = alert_data['level']
        risk_score = alert_data.get('risk_score', 0)
        location = alert_data.get('location', 'Unknown location')
        timestamp = alert_data.get('timestamp', 'Unknown time')
        
        color = "#ff4444" if level in ['CRITICAL', 'SYSTEM_CRITICAL'] else "#ffaa00"
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1>🌧️ Cloudburst Early Warning System</h1>
                    <h2>{level} ALERT</h2>
                </div>
                <div style="padding: 20px; background-color: #f9f9f9;">
                    <h3>Alert Details:</h3>
                    <p><strong>Location:</strong> {location}</p>
                    <p><strong>Risk Score:</strong> {risk_score:.1%}</p>
                    <p><strong>Time:</strong> {timestamp}</p>
                    <p><strong>Alert Level:</strong> {level}</p>
                </div>
                <div style="padding: 20px; background-color: #e9f7ff;">
                    <h3>Recommended Actions:</h3>
                    {self.get_recommended_actions(level)}
                </div>
                <div style="padding: 20px; text-align: center; background-color: #333; color: white;">
                    <p>This is an automated alert from Cloudburst Early Warning System</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def format_push_message(self, alert_data: Dict) -> str:
        """Format alert message for push notifications"""
        level = alert_data['level']
        risk_score = alert_data.get('risk_score', 0)
        
        if level == 'SYSTEM_CRITICAL':
            return f"EMERGENCY: Cloudburst detected! Risk: {risk_score:.0%} - Take immediate shelter!"
        elif level == 'CRITICAL':
            return f"WARNING: Cloudburst likely! Risk: {risk_score:.0%} - Seek shelter now!"
        else:
            return f"Alert: Potential cloudburst. Risk: {risk_score:.0%} - Stay alert!"
    
    def get_recommended_actions(self, level: str) -> str:
        """Get recommended actions based on alert level"""
        if level == 'SYSTEM_CRITICAL':
            return """
            <ul>
                <li>🚨 EVACUATE TO HIGHER GROUND IMMEDIATELY</li>
                <li>📱 Stay tuned to official updates</li>
                <li>🏠 Avoid rivers and low-lying areas</li>
                <li>🔊 Listen for emergency sirens</li>
            </ul>
            """
        elif level == 'CRITICAL':
            return """
            <ul>
                <li>⚠️ Seek shelter in safe location</li>
                <li>📞 Inform family and neighbors</li>
                <li>💧 Secure important supplies</li>
                <li>🚗 Avoid unnecessary travel</li>
            </ul>
            """
        else:
            return """
            <ul>
                <li>🔶 Monitor weather updates</li>
                <li>📱 Keep communication devices charged</li>
                <li>🏠 Review emergency plans</li>
                <li>📊 Stay informed about risk levels</li>
            </ul>
            """
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['sent_time']).timestamp() > cutoff_time
        ]
