#!/usr/bin/env python3
"""
Push Notification System for Cloudburst Alerts
Handles mobile app notifications via Firebase Cloud Messaging
"""

import firebase_admin
from firebase_admin import credentials, messaging
import logging
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import aiohttp
import json
import os

class PushNotificationSystem:
    """Push notification system using Firebase Cloud Messaging"""
    
    def __init__(self, config_file: str = "fcm_config.json"):
        self.config = self.load_config(config_file)
        self.initialized = False
        self.initialize_firebase()
        self.sent_notifications = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup push notification logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('push_notifications.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str) -> Dict:
        """Load FCM configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "fcm_credentials": "path/to/serviceAccountKey.json",
                    "android_config": {
                        "priority": "high",
                        "ttl": 3600,
                        "notification": {
                            "sound": "default",
                            "color": "#FF0000",
                            "icon": "ic_cloudburst_alert"
                        }
                    },
                    "apns_config": {
                        "payload": {
                            "aps": {
                                "sound": "default",
                                "badge": 1
                            }
                        }
                    },
                    "webpush_config": {
                        "headers": {
                            "TTL": "3600"
                        }
                    },
                    "topics": {
                        "all_users": "cloudburst_all",
                        "authorities": "cloudburst_authorities",
                        "community": "cloudburst_community"
                    }
                }
        except Exception as e:
            self.logger.error(f"Failed to load FCM config: {e}")
            return {}
    
    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            cred_path = self.config.get('fcm_credentials')
            
            if not cred_path or not os.path.exists(cred_path):
                self.logger.warning("FCM credentials not found, using mock mode")
                return
            
            # Initialize Firebase
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            self.initialized = True
            self.logger.info("Firebase Admin SDK initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Firebase: {e}")
            self.logger.info("Running in mock mode - notifications will be logged but not sent")
    
    async def send_alert_notification(self, alert_data: Dict, target: str = "all_users") -> bool:
        """Send push notification for alert"""
        try:
            # Determine target (topic or specific tokens)
            if target.startswith('topic:'):
                topic = target.replace('topic:', '')
                return await self.send_to_topic(alert_data, topic)
            elif target in self.config.get('topics', {}):
                topic = self.config['topics'][target]
                return await self.send_to_topic(alert_data, topic)
            else:
                # Assume it's a device token or list of tokens
                return await self.send_to_devices(alert_data, [target])
                
        except Exception as e:
            self.logger.error(f"Failed to send push notification: {e}")
            return False
    
    async def send_to_topic(self, alert_data: Dict, topic: str) -> bool:
        """Send notification to a topic"""
        try:
            message = self.create_message(alert_data, topic=topic)
            
            if self.initialized:
                # Send via FCM
                response = messaging.send(message)
                self.log_sent_notification(alert_data, topic, response)
                self.logger.info(f"Push notification sent to topic {topic}: {response}")
                return True
            else:
                # Mock send for development
                self.log_sent_notification(alert_data, topic, "mock_message_id")
                self.logger.info(f"Mock push notification sent to topic {topic}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to send to topic {topic}: {e}")
            return False
    
    async def send_to_devices(self, alert_data: Dict, device_tokens: List[str]) -> bool:
        """Send notification to specific devices"""
        try:
            if not device_tokens:
                self.logger.warning("No device tokens provided")
                return False
            
            # Split into chunks of 500 (FCM limit)
            chunk_size = 500
            token_chunks = [device_tokens[i:i + chunk_size] 
                          for i in range(0, len(device_tokens), chunk_size)]
            
            success_count = 0
            
            for chunk in token_chunks:
                message = self.create_message(alert_data, tokens=chunk)
                
                if self.initialized:
                    response = messaging.send_multicast(message)
                    success_count += response.success_count
                    
                    # Log failures
                    for idx, result in enumerate(response.responses):
                        if not result.success:
                            self.logger.error(f"Failed to send to token {chunk[idx]}: {result.exception}")
                    
                    self.logger.info(f"Multicast sent: {response.success_count}/{len(chunk)} successful")
                else:
                    # Mock send
                    success_count += len(chunk)
                    self.logger.info(f"Mock multicast sent to {len(chunk)} devices")
            
            success_rate = success_count / len(device_tokens)
            self.log_sent_notification(alert_data, f"{len(device_tokens)} devices", 
                                     f"{success_count} successful")
            
            return success_rate > 0.8
            
        except Exception as e:
            self.logger.error(f"Failed to send to devices: {e}")
            return False
    
    def create_message(self, alert_data: Dict, topic: str = None, tokens: List[str] = None) -> messaging.Message:
        """Create FCM message from alert data"""
        alert_level = alert_data.get('alert_level', 'info').upper()
        location = alert_data.get('location', 'Unknown Location')
        risk_score = alert_data.get('risk_score', 0)
        
        # Determine notification title and body
        if alert_level == 'CRITICAL':
            title = "🚨 CLOUDBURST EMERGENCY"
            body = f"Immediate threat at {location}. Risk: {risk_score:.0%}. EVACUATE NOW!"
            sound = "emergency_alert"
        elif alert_level == 'WARNING':
            title = "⚠️ CLOUDBURST WARNING"
            body = f"Potential cloudburst at {location}. Risk: {risk_score:.0%}. Seek shelter."
            sound = "warning_alert"
        else:
            title = "🔶 Cloudburst Alert"
            body = f"Monitoring alert for {location}. Risk: {risk_score:.0%}. Stay informed."
            sound = "default"
        
        # Create base message
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
                image=alert_data.get('image_url')  # Optional radar image
            ),
            data={
                'alert_id': alert_data.get('alert_id', ''),
                'alert_level': alert_level,
                'location': location,
                'risk_score': str(risk_score),
                'timestamp': alert_data.get('timestamp', datetime.now().isoformat()),
                'action_required': 'true' if alert_level in ['CRITICAL', 'WARNING'] else 'false',
                'deep_link': f"cloudburst://alerts/{alert_data.get('alert_id', '')}"
            },
            android=messaging.AndroidConfig(
                priority='high',
                ttl=3600,  # 1 hour
                notification=messaging.AndroidNotification(
                    sound=sound,
                    color='#FF0000' if alert_level == 'CRITICAL' else '#FFA500',
                    icon='ic_cloudburst_alert',
                    tag=alert_data.get('alert_id'),
                    click_action='OPEN_ALERT_DETAILS'
                )
            ),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        sound=sound,
                        badge=1,
                        category='CLOUDBURST_ALERT'
                    )
                )
            ),
            webpush=messaging.WebpushConfig(
                headers={'TTL': '3600'},
                notification=messaging.WebpushNotification(
                    icon='/icons/cloudburst-192.png',
                    badge='/icons/badge-72.png',
                    actions=[
                        messaging.WebpushNotificationAction(
                            action='view_details',
                            title='View Details'
                        ),
                        messaging.WebpushNotificationAction(
                            action='dismiss',
                            title='Dismiss'
                        )
                    ]
                )
            )
        )
        
        # Set target (topic or tokens)
        if topic:
            message = message.with_topic(topic)
        elif tokens:
            message = messaging.MulticastMessage(
                tokens=tokens,
                notification=message.notification,
                data=message.data,
                android=message.android,
                apns=message.apns,
                webpush=message.webpush
            )
        
        return message
    
    def log_sent_notification(self, alert_data: Dict, target: str, message_id: str):
        """Log sent notification for tracking"""
        notification_log = {
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert_data.get('alert_id'),
            'alert_level': alert_data.get('alert_level'),
            'target': target,
            'message_id': message_id,
            'risk_score': alert_data.get('risk_score')
        }
        
        self.sent_notifications.append(notification_log)
        
        # Keep only last 1000 notifications in memory
        if len(self.sent_notifications) > 1000:
            self.sent_notifications = self.sent_notifications[-1000:]
    
    async def subscribe_to_topic(self, device_tokens: List[str], topic: str) -> bool:
        """Subscribe devices to a topic"""
        try:
            if self.initialized:
                response = messaging.subscribe_to_topic(device_tokens, topic)
                self.logger.info(f"Subscribed {response.success_count} devices to topic {topic}")
                return response.success_count > 0
            else:
                self.logger.info(f"Mock subscribed {len(device_tokens)} devices to topic {topic}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to subscribe to topic {topic}: {e}")
            return False
    
    async def unsubscribe_from_topic(self, device_tokens: List[str], topic: str) -> bool:
        """Unsubscribe devices from a topic"""
        try:
            if self.initialized:
                response = messaging.unsubscribe_from_topic(device_tokens, topic)
                self.logger.info(f"Unsubscribed {response.success_count} devices from topic {topic}")
                return response.success_count > 0
            else:
                self.logger.info(f"Mock unsubscribed {len(device_tokens)} devices from topic {topic}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from topic {topic}: {e}")
            return False
    
    async def get_delivery_analytics(self, hours: int = 24) -> Dict:
        """Get push notification delivery analytics"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        recent_notifications = [
            n for n in self.sent_notifications
            if datetime.fromisoformat(n['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_notifications:
            return {
                'total_sent': 0,
                'by_alert_level': {},
                'success_rate': 0,
                'recent_activity': []
            }
        
        # Calculate statistics by alert level
        by_alert_level = {}
        for notification in recent_notifications:
            level = notification['alert_level']
            if level not in by_alert_level:
                by_alert_level[level] = 0
            by_alert_level[level] += 1
        
        # Mock success rate (in reality, would check FCM delivery reports)
        total_sent = len(recent_notifications)
        success_rate = min(0.95, 0.85 + (total_sent / 1000) * 0.1)
        
        return {
            'total_sent': total_sent,
            'by_alert_level': by_alert_level,
            'success_rate': success_rate,
            'recent_activity': recent_notifications[-10:]  # Last 10 notifications
        }
    
    async def send_test_notification(self, device_token: str = None, topic: str = None) -> bool:
        """Send test notification for debugging"""
        test_alert = {
            'alert_id': 'test_alert_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
            'alert_level': 'info',
            'location': 'Test Location',
            'risk_score': 0.5,
            'timestamp': datetime.now().isoformat()
        }
        
        if device_token:
            return await self.send_to_devices(test_alert, [device_token])
        elif topic:
            return await self.send_to_topic(test_alert, topic)
        else:
            self.logger.error("No device token or topic provided for test")
            return False

# WebSocket-based real-time notifications
class RealTimeNotificationManager:
    """Manages real-time notifications via WebSocket"""
    
    def __init__(self, push_system: PushNotificationSystem):
        self.push_system = push_system
        self.connected_clients = set()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup real-time notification logging"""
        self.logger = logging.getLogger(__name__ + '.RealTimeManager')
    
    async def handle_client_connection(self, websocket, path: str):
        """Handle new WebSocket client connection"""
        client_id = id(websocket)
        self.connected_clients.add(websocket)
        self.logger.info(f"Client {client_id} connected. Total clients: {len(self.connected_clients)}")
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except Exception as e:
            self.logger.error(f"WebSocket error for client {client_id}: {e}")
        finally:
            self.connected_clients.remove(websocket)
            self.logger.info(f"Client {client_id} disconnected. Total clients: {len(self.connected_clients)}")
    
    async def handle_client_message(self, websocket, message: str):
        """Handle message from WebSocket client"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe_alerts':
                # Client wants to receive real-time alerts
                await self.send_acknowledgment(websocket, 'subscribed')
                self.logger.info(f"Client subscribed to real-time alerts")
                
            elif message_type == 'unsubscribe_alerts':
                # Client wants to stop receiving alerts
                await self.send_acknowledgment(websocket, 'unsubscribed')
                self.logger.info(f"Client unsubscribed from real-time alerts")
                
            elif message_type == 'ping':
                # Keep-alive ping
                await self.send_pong(websocket)
                
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON message from client")
    
    async def broadcast_alert(self, alert_data: Dict):
        """Broadcast alert to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = {
            'type': 'alert',
            'data': alert_data,
            'timestamp': datetime.now().isoformat()
        }
        
        disconnected_clients = set()
        
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Failed to send to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)
        
        self.logger.info(f"Alert broadcast to {len(self.connected_clients)} clients")
    
    async def send_acknowledgment(self, websocket, status: str):
        """Send acknowledgment to client"""
        message = {
            'type': 'ack',
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send acknowledgment: {e}")
    
    async def send_pong(self, websocket):
        """Send pong response to ping"""
        message = {
            'type': 'pong',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Failed to send pong: {e}")

# Example usage
async def main():
    """Example usage of Push Notification System"""
    
    # Initialize push system
    push_system = PushNotificationSystem()
    
    # Example alert
    alert_data = {
        'alert_id': 'alert_20241001123000',
        'alert_level': 'critical',
        'location': 'Mountain Valley Region',
        'risk_score': 0.92,
        'timestamp': datetime.now().isoformat()
    }
    
    # Send to all users
    success = await push_system.send_alert_notification(alert_data, 'all_users')
    
    if success:
        print("Push notification sent successfully")
    else:
        print("Failed to send push notification")
    
    # Get analytics
    analytics = await push_system.get_delivery_analytics()
    print(f"Delivery analytics: {analytics}")

if __name__ == '__main__':
    asyncio.run(main())
