#!/usr/bin/env python3
"""
SMS Gateway for Cloudburst Alert System
Handles SMS notifications via multiple providers
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import aiohttp
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
import json
import os

class SMSGateway:
    """SMS Gateway supporting multiple providers with fallback"""
    
    def __init__(self, config_file: str = "sms_config.json"):
        self.config = self.load_config(config_file)
        self.providers = self.initialize_providers()
        self.sent_messages = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup SMS gateway logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sms_gateway.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str) -> Dict:
        """Load SMS gateway configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    "providers": {
                        "twilio": {
                            "enabled": True,
                            "account_sid": "your_account_sid",
                            "auth_token": "your_auth_token",
                            "from_number": "+1234567890",
                            "priority": 1
                        },
                        "plivo": {
                            "enabled": False,
                            "auth_id": "your_auth_id",
                            "auth_token": "your_auth_token",
                            "from_number": "+1234567890",
                            "priority": 2
                        },
                        "nexmo": {
                            "enabled": False,
                            "api_key": "your_api_key",
                            "api_secret": "your_api_secret",
                            "from_number": "Cloudburst",
                            "priority": 3
                        }
                    },
                    "retry_attempts": 3,
                    "retry_delay": 5,
                    "message_templates": {
                        "critical": "🚨 CLOUDBURST EMERGENCY 🚨\nLocation: {location}\nRisk Level: {risk_score:.0%}\nTime: {timestamp}\nAction: EVACUATE IMMEDIATELY",
                        "warning": "⚠️ CLOUDBURST WARNING ⚠️\nLocation: {location}\nRisk Level: {risk_score:.0%}\nTime: {timestamp}\nAction: SEEK SHELTER",
                        "info": "🔶 Cloudburst Alert\nLocation: {location}\nRisk Level: {risk_score:.0%}\nTime: {timestamp}\nAction: STAY ALERT"
                    },
                    "recipient_groups": {
                        "authorities": ["+911234567890", "+911234567891"],
                        "emergency_services": ["+911234567892", "+911234567893"],
                        "community_leaders": ["+911234567894", "+911234567895"]
                    }
                }
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def initialize_providers(self) -> Dict:
        """Initialize SMS providers"""
        providers = {}
        
        # Twilio provider
        if self.config.get('providers', {}).get('twilio', {}).get('enabled', False):
            try:
                twilio_config = self.config['providers']['twilio']
                providers['twilio'] = TwilioProvider(
                    account_sid=twilio_config['account_sid'],
                    auth_token=twilio_config['auth_token'],
                    from_number=twilio_config['from_number']
                )
                self.logger.info("Twilio provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Twilio: {e}")
        
        # Plivo provider
        if self.config.get('providers', {}).get('plivo', {}).get('enabled', False):
            try:
                plivo_config = self.config['providers']['plivo']
                providers['plivo'] = PlivoProvider(
                    auth_id=plivo_config['auth_id'],
                    auth_token=plivo_config['auth_token'],
                    from_number=plivo_config['from_number']
                )
                self.logger.info("Plivo provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Plivo: {e}")
        
        # Nexmo/Vonage provider
        if self.config.get('providers', {}).get('nexmo', {}).get('enabled', False):
            try:
                nexmo_config = self.config['providers']['nexmo']
                providers['nexmo'] = NexmoProvider(
                    api_key=nexmo_config['api_key'],
                    api_secret=nexmo_config['api_secret'],
                    from_number=nexmo_config['from_number']
                )
                self.logger.info("Nexmo provider initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Nexmo: {e}")
        
        # Sort providers by priority
        sorted_providers = sorted(
            providers.items(),
            key=lambda x: self.config['providers'][x[0]]['priority']
        )
        
        return dict(sorted_providers)
    
    async def send_alert(self, alert_data: Dict, recipient_group: str = "authorities") -> bool:
        """Send SMS alert to specified recipient group"""
        try:
            # Get recipients for the group
            recipients = self.config['recipient_groups'].get(recipient_group, [])
            
            if not recipients:
                self.logger.warning(f"No recipients found for group: {recipient_group}")
                return False
            
            # Format message based on alert level
            alert_level = alert_data.get('alert_level', 'info').lower()
            template = self.config['message_templates'].get(alert_level, 
                                                          self.config['message_templates']['info'])
            
            message = template.format(
                location=alert_data.get('location', 'Unknown Location'),
                risk_score=alert_data.get('risk_score', 0),
                timestamp=alert_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            )
            
            self.logger.info(f"Sending {alert_level} alert to {len(recipients)} recipients")
            
            # Send messages with retry logic
            success_count = 0
            for recipient in recipients:
                if await self.send_single_message(recipient, message, alert_data):
                    success_count += 1
            
            success_rate = success_count / len(recipients) if recipients else 0
            self.logger.info(f"SMS delivery completed: {success_count}/{len(recipients)} successful")
            
            return success_rate > 0.8  # Consider successful if 80%+ delivered
        
        except Exception as e:
            self.logger.error(f"Failed to send SMS alert: {e}")
            return False
    
    async def send_single_message(self, to_number: str, message: str, alert_data: Dict) -> bool:
        """Send single SMS message with provider fallback"""
        max_retries = self.config.get('retry_attempts', 3)
        
        for attempt in range(max_retries):
            for provider_name, provider in self.providers.items():
                try:
                    self.logger.info(f"Attempt {attempt + 1} with {provider_name} to {to_number}")
                    
                    success = await provider.send_message(to_number, message)
                    
                    if success:
                        # Log successful delivery
                        self.log_sent_message(to_number, message, provider_name, alert_data)
                        self.logger.info(f"Message sent successfully via {provider_name}")
                        return True
                    else:
                        self.logger.warning(f"Provider {provider_name} failed, trying next provider")
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Provider {provider_name} error: {e}")
                    continue
            
            # If all providers failed, wait before retry
            if attempt < max_retries - 1:
                retry_delay = self.config.get('retry_delay', 5)
                self.logger.info(f"All providers failed, retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
        
        self.logger.error(f"Failed to send message to {to_number} after {max_retries} attempts")
        return False
    
    def log_sent_message(self, to_number: str, message: str, provider: str, alert_data: Dict):
        """Log sent message for tracking and analytics"""
        sent_message = {
            'timestamp': datetime.now().isoformat(),
            'to_number': to_number,
            'message': message,
            'provider': provider,
            'alert_id': alert_data.get('alert_id'),
            'alert_level': alert_data.get('alert_level'),
            'message_length': len(message)
        }
        
        self.sent_messages.append(sent_message)
        
        # Keep only last 1000 messages in memory
        if len(self.sent_messages) > 1000:
            self.sent_messages = self.sent_messages[-1000:]
    
    async def get_delivery_status(self) -> Dict:
        """Get SMS delivery statistics"""
        total_sent = len(self.sent_messages)
        
        if total_sent == 0:
            return {
                'total_sent': 0,
                'success_rate': 0,
                'provider_stats': {},
                'recent_failures': []
            }
        
        # Calculate provider statistics
        provider_stats = {}
        for message in self.sent_messages:
            provider = message['provider']
            if provider not in provider_stats:
                provider_stats[provider] = 0
            provider_stats[provider] += 1
        
        # Calculate success rate (simplified - in reality would check delivery receipts)
        success_rate = min(0.95, 0.8 + (len(self.sent_messages) / 1000) * 0.15)  # Mock success rate
        
        return {
            'total_sent': total_sent,
            'success_rate': success_rate,
            'provider_stats': provider_stats,
            'recent_messages': self.sent_messages[-10:]  # Last 10 messages
        }
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate phone number format"""
        # Simple validation - in production, use a proper validation library
        import re
        pattern = r'^\+?[1-9]\d{1,14}$'  # E.164 format
        return re.match(pattern, phone_number) is not None
    
    async def add_recipient(self, phone_number: str, groups: List[str]):
        """Add recipient to specified groups"""
        if not self.validate_phone_number(phone_number):
            raise ValueError(f"Invalid phone number format: {phone_number}")
        
        for group in groups:
            if group not in self.config['recipient_groups']:
                self.config['recipient_groups'][group] = []
            
            if phone_number not in self.config['recipient_groups'][group]:
                self.config['recipient_groups'][group].append(phone_number)
        
        # Save updated configuration
        await self.save_config()
        self.logger.info(f"Added {phone_number} to groups: {groups}")
    
    async def remove_recipient(self, phone_number: str, groups: List[str] = None):
        """Remove recipient from specified groups (or all groups)"""
        if groups is None:
            # Remove from all groups
            for group_recipients in self.config['recipient_groups'].values():
                if phone_number in group_recipients:
                    group_recipients.remove(phone_number)
        else:
            # Remove from specified groups only
            for group in groups:
                if group in self.config['recipient_groups'] and phone_number in self.config['recipient_groups'][group]:
                    self.config['recipient_groups'][group].remove(phone_number)
        
        # Save updated configuration
        await self.save_config()
        self.logger.info(f"Removed {phone_number} from groups: {groups or 'all'}")
    
    async def save_config(self):
        """Save current configuration to file"""
        try:
            with open('sms_config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("SMS configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save SMS configuration: {e}")

# Provider Implementations

class TwilioProvider:
    """Twilio SMS provider implementation"""
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.logger = logging.getLogger(__name__)
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """Send SMS via Twilio"""
        try:
            # Twilio API call
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            
            # Check if message was successfully queued
            return message.status in ['queued', 'sent', 'delivered']
            
        except TwilioRestException as e:
            self.logger.error(f"Twilio API error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Twilio unexpected error: {e}")
            return False

class PlivoProvider:
    """Plivo SMS provider implementation"""
    
    def __init__(self, auth_id: str, auth_token: str, from_number: str):
        self.auth_id = auth_id
        self.auth_token = auth_token
        self.from_number = from_number
        self.base_url = f"https://api.plivo.com/v1/Account/{auth_id}"
        self.logger = logging.getLogger(__name__)
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """Send SMS via Plivo"""
        try:
            import base64
            
            # Basic authentication
            credentials = base64.b64encode(
                f"{self.auth_id}:{self.auth_token}".encode()
            ).decode()
            
            headers = {
                'Authorization': f'Basic {credentials}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'src': self.from_number,
                'dst': to_number,
                'text': message
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/Message/",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 202:
                        result = await response.json()
                        return result.get('message') == 'message(s) queued'
                    else:
                        self.logger.error(f"Plivo API error: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Plivo unexpected error: {e}")
            return False

class NexmoProvider:
    """Nexmo/Vonage SMS provider implementation"""
    
    def __init__(self, api_key: str, api_secret: str, from_number: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.from_number = from_number
        self.base_url = "https://rest.nexmo.com/sms/json"
        self.logger = logging.getLogger(__name__)
    
    async def send_message(self, to_number: str, message: str) -> bool:
        """Send SMS via Nexmo/Vonage"""
        try:
            params = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'from': self.from_number,
                'to': to_number,
                'text': message
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, data=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        messages = result.get('messages', [])
                        if messages:
                            return messages[0].get('status') == '0'  # Success status
                        return False
                    else:
                        self.logger.error(f"Nexmo API error: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Nexmo unexpected error: {e}")
            return False

# Example usage
async def main():
    """Example usage of SMS Gateway"""
    
    # Initialize gateway
    sms_gateway = SMSGateway()
    
    # Example alert data
    alert_data = {
        'alert_id': 'alert_20241001123000',
        'alert_level': 'critical',
        'location': 'Mountain Valley Region',
        'risk_score': 0.95,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Send alert to authorities
    success = await sms_gateway.send_alert(alert_data, 'authorities')
    
    if success:
        print("SMS alerts sent successfully")
    else:
        print("Failed to send SMS alerts")
    
    # Get delivery statistics
    stats = await sms_gateway.get_delivery_status()
    print(f"Delivery statistics: {stats}")

if __name__ == '__main__':
    asyncio.run(main())
