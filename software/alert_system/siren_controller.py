#!/usr/bin/env python3
"""
Siren Controller for Cloudburst Alert System
Controls physical warning sirens and audio alerts
"""

import RPi.GPIO as GPIO
import pygame
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import os
from pathlib import Path

class SirenController:
    """Controls physical sirens and audio warning systems"""
    
    def __init__(self, config_file: str = "siren_config.json"):
        self.config = self.load_config(config_file)
        self.siren_states = {}
        self.audio_files = self.load_audio_files()
        self.gpio_initialized = False
        self.audio_initialized = False
        self.setup_logging()
        self.initialize_hardware()
    
    def setup_logging(self):
        """Setup siren controller logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('siren_controller.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str) -> Dict:
        """Load siren configuration"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "gpio_pins": {
                        "siren_1": 18,
                        "siren_2": 23,
                        "siren_3": 24,
                        "status_led": 25
                    },
                    "audio_settings": {
                        "volume": 0.8,
                        "sample_rate": 44100,
                        "channels": 2,
                        "buffer_size": 2048
                    },
                    "siren_patterns": {
                        "critical": {
                            "pattern": [1, 0, 1, 0, 1, 0],  # Fast alternating
                            "duration": 300,  # seconds
                            "audio_file": "critical_alert.wav",
                            "gpio_sequence": [1, 0, 1, 0]
                        },
                        "warning": {
                            "pattern": [1, 1, 0, 0],  # Slow pulse
                            "duration": 180,
                            "audio_file": "warning_alert.wav",
                            "gpio_sequence": [1, 1, 0, 0]
                        },
                        "test": {
                            "pattern": [1, 0],
                            "duration": 10,
                            "audio_file": "test_alert.wav",
                            "gpio_sequence": [1, 0]
                        }
                    },
                    "siren_locations": {
                        "valley_entrance": {"gpio_pin": 18, "volume": 0.9},
                        "mountain_base": {"gpio_pin": 23, "volume": 0.8},
                        "community_center": {"gpio_pin": 24, "volume": 0.7}
                    }
                }
        except Exception as e:
            self.logger.error(f"Failed to load siren config: {e}")
            return {}
    
    def load_audio_files(self) -> Dict:
        """Load audio alert files"""
        audio_dir = Path("audio_alerts")
        audio_dir.mkdir(exist_ok=True)
        
        audio_files = {
            "critical_alert": audio_dir / "critical_alert.wav",
            "warning_alert": audio_dir / "warning_alert.wav",
            "test_alert": audio_dir / "test_alert.wav",
            "all_clear": audio_dir / "all_clear.wav"
        }
        
        # Create placeholder audio files if they don't exist
        for audio_file in audio_files.values():
            if not audio_file.exists():
                self.create_placeholder_audio(audio_file)
        
        return audio_files
    
    def create_placeholder_audio(self, file_path: Path):
        """Create placeholder audio files for testing"""
        try:
            import wave
            import struct
            
            # Create a simple sine wave alert tone
            sample_rate = 44100
            duration = 3.0  # seconds
            frequency = 880  # Hz (A5 note)
            
            # Generate sine wave
            samples = []
            for i in range(int(duration * sample_rate)):
                sample = 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate)
                samples.append(sample)
            
            # Convert to 16-bit PCM
            pcm_data = struct.pack('<' + ('h' * len(samples)), 
                                 *[int(s * 32767) for s in samples])
            
            # Write WAV file
            with wave.open(str(file_path), 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_data)
            
            self.logger.info(f"Created placeholder audio: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create placeholder audio: {e}")
    
    def initialize_hardware(self):
        """Initialize GPIO and audio hardware"""
        try:
            # Initialize GPIO
            GPIO.setmode(GPIO.BCM)
            for location, config in self.config.get('siren_locations', {}).items():
                gpio_pin = config.get('gpio_pin')
                if gpio_pin:
                    GPIO.setup(gpio_pin, GPIO.OUT)
                    GPIO.output(gpio_pin, GPIO.LOW)
                    self.siren_states[location] = {
                        'pin': gpio_pin,
                        'state': False,
                        'active_alert': None,
                        'start_time': None
                    }
            
            self.gpio_initialized = True
            self.logger.info("GPIO hardware initialized")
            
        except Exception as e:
            self.logger.error(f"GPIO initialization failed: {e}")
            self.gpio_initialized = False
        
        try:
            # Initialize audio
            pygame.mixer.init(
                frequency=self.config['audio_settings']['sample_rate'],
                size=-16,
                channels=self.config['audio_settings']['channels'],
                buffer=self.config['audio_settings']['buffer_size']
            )
            pygame.mixer.music.set_volume(self.config['audio_settings']['volume'])
            self.audio_initialized = True
            self.logger.info("Audio system initialized")
            
        except Exception as e:
            self.logger.error(f"Audio initialization failed: {e}")
            self.audio_initialized = False
    
    async def activate_siren(self, alert_data: Dict, locations: List[str] = None):
        """Activate sirens for alert"""
        try:
            alert_level = alert_data.get('alert_level', 'warning').lower()
            siren_config = self.config['siren_patterns'].get(alert_level)
            
            if not siren_config:
                self.logger.error(f"No siren configuration for alert level: {alert_level}")
                return False
            
            # If no locations specified, activate all
            if locations is None:
                locations = list(self.config['siren_locations'].keys())
            
            self.logger.info(f"Activating sirens for {alert_level} alert at locations: {locations}")
            
            # Start siren control tasks for each location
            tasks = []
            for location in locations:
                if location in self.config['siren_locations']:
                    task = asyncio.create_task(
                        self.control_siren_location(location, siren_config, alert_data)
                    )
                    tasks.append(task)
            
            # Wait for all sirens to start
            if tasks:
                await asyncio.gather(*tasks)
                self.logger.info(f"All sirens activated for {alert_level} alert")
                return True
            else:
                self.logger.warning("No valid siren locations specified")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to activate sirens: {e}")
            return False
    
    async def control_siren_location(self, location: str, siren_config: Dict, alert_data: Dict):
        """Control siren at specific location"""
        try:
            location_config = self.config['siren_locations'][location]
            gpio_pin = location_config['gpio_pin']
            pattern = siren_config['pattern']
            duration = siren_config['duration']
            audio_file = siren_config.get('audio_file')
            
            # Update siren state
            self.siren_states[location] = {
                'pin': gpio_pin,
                'state': True,
                'active_alert': alert_data.get('alert_id'),
                'start_time': datetime.now(),
                'pattern': pattern,
                'duration': duration
            }
            
            self.logger.info(f"Starting siren at {location} for {duration} seconds")
            
            # Play audio alert if available
            if audio_file and self.audio_initialized:
                await self.play_audio_alert(audio_file, location_config.get('volume', 0.8))
            
            # Control GPIO pattern
            start_time = datetime.now()
            pattern_index = 0
            
            while (datetime.now() - start_time).total_seconds() < duration:
                # Get current pattern state
                state = pattern[pattern_index % len(pattern)]
                
                # Control GPIO
                if self.gpio_initialized:
                    GPIO.output(gpio_pin, GPIO.HIGH if state else GPIO.LOW)
                
                # Wait for pattern interval (0.5 seconds per state)
                await asyncio.sleep(0.5)
                pattern_index += 1
            
            # Turn off siren
            if self.gpio_initialized:
                GPIO.output(gpio_pin, GPIO.LOW)
            
            # Update state
            self.siren_states[location]['state'] = False
            self.siren_states[location]['active_alert'] = None
            
            self.logger.info(f"Siren at {location} deactivated")
            
        except Exception as e:
            self.logger.error(f"Error controlling siren at {location}: {e}")
            # Ensure siren is turned off on error
            if self.gpio_initialized and location in self.config['siren_locations']:
                GPIO.output(self.config['siren_locations'][location]['gpio_pin'], GPIO.LOW)
    
    async def play_audio_alert(self, audio_file: str, volume: float = 0.8):
        """Play audio alert through speakers"""
        try:
            audio_path = self.audio_files.get(audio_file.replace('.wav', ''))
            if not audio_path or not audio_path.exists():
                self.logger.warning(f"Audio file not found: {audio_file}")
                return
            
            # Set volume
            pygame.mixer.music.set_volume(volume)
            
            # Load and play audio
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            self.logger.info(f"Audio alert completed: {audio_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to play audio alert: {e}")
    
    async def stop_all_sirens(self):
        """Stop all active sirens immediately"""
        try:
            self.logger.info("Stopping all sirens")
            
            # Stop audio
            if self.audio_initialized:
                pygame.mixer.music.stop()
            
            # Turn off all GPIO pins
            if self.gpio_initialized:
                for location, config in self.config['siren_locations'].items():
                    GPIO.output(config['gpio_pin'], GPIO.LOW)
            
            # Update states
            for location in self.siren_states:
                self.siren_states[location]['state'] = False
                self.siren_states[location]['active_alert'] = None
            
            self.logger.info("All sirens stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping sirens: {e}")
    
    async def stop_siren_location(self, location: str):
        """Stop siren at specific location"""
        try:
            if location not in self.siren_states:
                self.logger.warning(f"Unknown siren location: {location}")
                return
            
            if self.gpio_initialized:
                gpio_pin = self.siren_states[location]['pin']
                GPIO.output(gpio_pin, GPIO.LOW)
            
            self.siren_states[location]['state'] = False
            self.siren_states[location]['active_alert'] = None
            
            self.logger.info(f"Siren stopped at location: {location}")
            
        except Exception as e:
            self.logger.error(f"Error stopping siren at {location}: {e}")
    
    async def test_siren_system(self, location: str = None):
        """Test siren system functionality"""
        try:
            self.logger.info("Starting siren system test")
            
            test_alert = {
                'alert_id': 'test_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
                'alert_level': 'test',
                'location': 'Test Location',
                'timestamp': datetime.now().isoformat()
            }
            
            if location:
                locations = [location]
            else:
                locations = list(self.config['siren_locations'].keys())
            
            # Brief test activation
            await self.activate_siren(test_alert, locations)
            
            # Wait for test to complete
            await asyncio.sleep(10)
            
            # Stop all sirens
            await self.stop_all_sirens()
            
            self.logger.info("Siren system test completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Siren system test failed: {e}")
            return False
    
    def get_siren_status(self) -> Dict:
        """Get current siren system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'gpio_initialized': self.gpio_initialized,
            'audio_initialized': self.audio_initialized,
            'active_sirens': 0,
            'siren_states': {}
        }
        
        for location, state in self.siren_states.items():
            status['siren_states'][location] = {
                'active': state.get('state', False),
                'active_alert': state.get('active_alert'),
                'start_time': state.get('start_time'),
                'duration_remaining': 0
            }
            
            if state.get('state'):
                status['active_sirens'] += 1
                start_time = state.get('start_time')
                duration = state.get('duration', 0)
                
                if start_time and duration:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    status['siren_states'][location]['duration_remaining'] = max(
                        0, duration - elapsed
                    )
        
        return status
    
    async def emergency_override(self, command: str):
        """Emergency override commands for manual control"""
        try:
            if command == "activate_all":
                emergency_alert = {
                    'alert_id': 'emergency_override',
                    'alert_level': 'critical',
                    'location': 'ALL LOCATIONS',
                    'timestamp': datetime.now().isoformat()
                }
                await self.activate_siren(emergency_alert)
                
            elif command == "deactivate_all":
                await self.stop_all_sirens()
                
            elif command.startswith("activate_"):
                location = command.replace("activate_", "")
                emergency_alert = {
                    'alert_id': 'emergency_override',
                    'alert_level': 'critical',
                    'location': location,
                    'timestamp': datetime.now().isoformat()
                }
                await self.activate_siren(emergency_alert, [location])
                
            elif command.startswith("deactivate_"):
                location = command.replace("deactivate_", "")
                await self.stop_siren_location(location)
                
            else:
                self.logger.warning(f"Unknown emergency command: {command}")
                return False
            
            self.logger.info(f"Emergency override executed: {command}")
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency override failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup GPIO and audio resources"""
        try:
            # Stop all sirens
            asyncio.run(self.stop_all_sirens())
            
            # Cleanup GPIO
            if self.gpio_initialized:
                GPIO.cleanup()
            
            # Quit pygame
            if self.audio_initialized:
                pygame.mixer.quit()
            
            self.logger.info("Siren controller cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Web API for remote siren control
class SirenAPI:
    """REST API for remote siren control"""
    
    def __init__(self, siren_controller: SirenController, host: str = '0.0.0.0', port: int = 8081):
        self.siren_controller = siren_controller
        self.host = host
        self.port = port
        self.setup_logging()
    
    def setup_logging(self):
        """Setup API logging"""
        self.logger = logging.getLogger(__name__ + '.API')
    
    async def start_server(self):
        """Start the siren control API server"""
        from aiohttp import web
        import aiohttp_cors
        
        app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        
        # Add routes
        app.router.add_get('/', self.handle_root)
        app.router.add_get('/status', self.handle_status)
        app.router.add_post('/activate', self.handle_activate)
        app.router.add_post('/deactivate', self.handle_deactivate)
        app.router.add_post('/test', self.handle_test)
        app.router.add_post('/emergency', self.handle_emergency)
        
        # Apply CORS to all routes
        for route in list(app.router.routes()):
            cors.add(route)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"Siren API server started on {self.host}:{self.port}")
        
        return runner
    
    async def handle_root(self, request):
        """Root endpoint"""
        return web.json_response({
            'service': 'Siren Controller API',
            'version': '1.0.0',
            'endpoints': [
                '/status - Get siren status',
                '/activate - Activate sirens',
                '/deactivate - Deactivate sirens',
                '/test - Test siren system',
                '/emergency - Emergency override'
            ]
        })
    
    async def handle_status(self, request):
        """Get siren system status"""
        status = self.siren_controller.get_siren_status()
        return web.json_response(status)
    
    async def handle_activate(self, request):
        """Activate sirens for alert"""
        try:
            data = await request.json()
            
            # Validate required fields
            if 'alert_level' not in data:
                return web.json_response(
                    {'error': 'alert_level is required'}, status=400
                )
            
            locations = data.get('locations')  # Optional specific locations
            
            success = await self.siren_controller.activate_siren(data, locations)
            
            if success:
                return web.json_response({'status': 'activated'})
            else:
                return web.json_response(
                    {'error': 'Failed to activate sirens'}, status=500
                )
                
        except Exception as e:
            self.logger.error(f"Activation error: {e}")
            return web.json_response(
                {'error': 'Internal server error'}, status=500
            )
    
    async def handle_deactivate(self, request):
        """Deactivate sirens"""
        try:
            data = await request.json()
            location = data.get('location')  # Optional specific location
            
            if location:
                await self.siren_controller.stop_siren_location(location)
            else:
                await self.siren_controller.stop_all_sirens()
            
            return web.json_response({'status': 'deactivated'})
            
        except Exception as e:
            self.logger.error(f"Deactivation error: {e}")
            return web.json_response(
                {'error': 'Internal server error'}, status=500
            )
    
    async def handle_test(self, request):
        """Test siren system"""
        try:
            data = await request.json()
            location = data.get('location')
            
            success = await self.siren_controller.test_siren_system(location)
            
            if success:
                return web.json_response({'status': 'test_completed'})
            else:
                return web.json_response(
                    {'error': 'Test failed'}, status=500
                )
                
        except Exception as e:
            self.logger.error(f"Test error: {e}")
            return web.json_response(
                {'error': 'Internal server error'}, status=500
            )
    
    async def handle_emergency(self, request):
        """Emergency override commands"""
        try:
            data = await request.json()
            command = data.get('command')
            auth_token = data.get('auth_token')
            
            # Validate authentication
            if not self.authenticate_emergency(auth_token):
                return web.json_response(
                    {'error': 'Unauthorized'}, status=401
                )
            
            if not command:
                return web.json_response(
                    {'error': 'command is required'}, status=400
                )
            
            success = await self.siren_controller.emergency_override(command)
            
            if success:
                return web.json_response({'status': 'emergency_executed'})
            else:
                return web.json_response(
                    {'error': 'Emergency command failed'}, status=500
                )
                
        except Exception as e:
            self.logger.error(f"Emergency command error: {e}")
            return web.json_response(
                {'error': 'Internal server error'}, status=500
            )
    
    def authenticate_emergency(self, auth_token: str) -> bool:
        """Authenticate emergency commands"""
        # In production, this would validate against a secure token
        expected_token = "emergency_access_2024"
        return auth_token == expected_token

# Example usage
async def main():
    """Example usage of Siren Controller"""
    
    # Initialize siren controller
    siren_controller = SirenController()
    
    # Example alert
    alert_data = {
        'alert_id': 'alert_20241001123000',
        'alert_level': 'warning',
        'location': 'Mountain Valley Region',
        'timestamp': datetime.now().isoformat()
    }
    
    # Activate sirens at specific locations
    locations = ['valley_entrance', 'community_center']
    success = await siren_controller.activate_siren(alert_data, locations)
    
    if success:
        print("Sirens activated successfully")
    else:
        print("Failed to activate sirens")
    
    # Check status
    status = siren_controller.get_siren_status()
    print(f"Siren status: {status}")
    
    # Start API server
    api = SirenAPI(siren_controller)
    await api.start_server()
    
    # Keep running
    await asyncio.Event().wait()

if __name__ == '__main__':
    import math  # For placeholder audio generation
    
    asyncio.run(main())
