#!/usr/bin/env python3
"""
Gateway Controller for Cloudburst Early Warning System
Handles data aggregation from multiple edge nodes and coordinates alerts
"""

import serial
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import paho.mqtt.client as mqtt
from database import TimescaleDB
from alert_handler import AlertHandler

@dataclass
class NodeData:
    node_id: str
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    rainfall: float
    lightning_count: int
    risk_score: float
    battery_level: float
    signal_strength: int

class GatewayController:
    def __init__(self, config_file: str = "gateway_config.json"):
        self.config = self.load_config(config_file)
        self.nodes: Dict[str, NodeData] = {}
        self.alert_handler = AlertHandler()
        self.db = TimescaleDB()
        self.mqtt_client = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gateway.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.initialize_communication()
        
    def load_config(self, config_file: str) -> Dict:
        """Load gateway configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("Config file not found, using defaults")
            return {
                "lora_serial_port": "/dev/ttyUSB0",
                "lora_baudrate": 115200,
                "mqtt_broker": "localhost",
                "mqtt_port": 1883,
                "alert_threshold": 0.7,
                "critical_threshold": 0.9
            }
    
    def initialize_communication(self):
        """Initialize all communication interfaces"""
        try:
            # Initialize LoRa serial connection
            self.lora_serial = serial.Serial(
                self.config["lora_serial_port"],
                self.config["lora_baudrate"],
                timeout=1
            )
            
            # Initialize MQTT client
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(
                self.config["mqtt_broker"],
                self.config["mqtt_port"]
            )
            self.mqtt_client.loop_start()
            
            self.logger.info("Communication interfaces initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise
    
    def process_incoming_data(self):
        """Main loop to process incoming data from edge nodes"""
        self.logger.info("Starting gateway data processing loop")
        
        while True:
            try:
                # Read data from LoRa
                if self.lora_serial.in_waiting > 0:
                    raw_data = self.lora_serial.readline().decode('utf-8').strip()
                    if raw_data:
                        self.process_lora_packet(raw_data)
                
                # Check for alerts
                self.check_system_alerts()
                
                # Send aggregated data to cloud
                self.send_aggregated_data()
                
                time.sleep(1)  # Prevent CPU overload
                
            except KeyboardInterrupt:
                self.logger.info("Shutting down gateway controller")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def process_lora_packet(self, packet_data: str):
        """Process incoming LoRa packet from edge nodes"""
        try:
            data = json.loads(packet_data)
            node_id = data.get("node_id")
            
            if not node_id:
                self.logger.warning("Received packet without node ID")
                return
            
            # Create NodeData object
            node_data = NodeData(
                node_id=node_id,
                timestamp=datetime.fromisoformat(data.get("timestamp")),
                temperature=data.get("temperature"),
                humidity=data.get("humidity"),
                pressure=data.get("pressure"),
                rainfall=data.get("rainfall"),
                lightning_count=data.get("lightning_count", 0),
                risk_score=data.get("risk_score", 0.0),
                battery_level=data.get("battery_level", 100.0),
                signal_strength=data.get("signal_strength", 0)
            )
            
            # Update node data
            self.nodes[node_id] = node_data
            
            # Store in database
            self.db.store_node_data(node_data)
            
            # Publish to MQTT
            self.publish_to_mqtt(node_data)
            
            self.logger.info(f"Processed data from node {node_id}")
            
            # Check if alert needed
            if node_data.risk_score > self.config["alert_threshold"]:
                self.trigger_alert(node_data)
                
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON packet: {packet_data}")
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
    
    def trigger_alert(self, node_data: NodeData):
        """Trigger appropriate alert based on risk score"""
        alert_level = "WARNING"
        if node_data.risk_score > self.config["critical_threshold"]:
            alert_level = "CRITICAL"
        
        alert_message = {
            "level": alert_level,
            "node_id": node_data.node_id,
            "risk_score": node_data.risk_score,
            "rainfall": node_data.rainfall,
            "timestamp": node_data.timestamp.isoformat(),
            "location": self.get_node_location(node_data.node_id)
        }
        
        self.alert_handler.send_alert(alert_message)
        self.logger.warning(f"Alert triggered: {alert_message}")
    
    def check_system_alerts(self):
        """Check for system-wide alert conditions"""
        # Check if multiple nodes reporting high risk
        high_risk_nodes = [
            node for node in self.nodes.values() 
            if node.risk_score > self.config["alert_threshold"]
        ]
        
        if len(high_risk_nodes) >= 2:
            self.trigger_system_alert(high_risk_nodes)
    
    def trigger_system_alert(self, high_risk_nodes: List[NodeData]):
        """Trigger system-wide alert"""
        system_alert = {
            "level": "SYSTEM_CRITICAL",
            "affected_nodes": [node.node_id for node in high_risk_nodes],
            "average_risk_score": sum(node.risk_score for node in high_risk_nodes) / len(high_risk_nodes),
            "timestamp": datetime.now().isoformat(),
            "message": "Multiple nodes detecting cloudburst conditions"
        }
        
        self.alert_handler.send_system_alert(system_alert)
        self.logger.critical(f"System alert: {system_alert}")
    
    def publish_to_mqtt(self, node_data: NodeData):
        """Publish node data to MQTT broker"""
        if self.mqtt_client:
            topic = f"cloudburst/nodes/{node_data.node_id}"
            payload = json.dumps({
                "timestamp": node_data.timestamp.isoformat(),
                "temperature": node_data.temperature,
                "humidity": node_data.humidity,
                "pressure": node_data.pressure,
                "rainfall": node_data.rainfall,
                "risk_score": node_data.risk_score,
                "battery": node_data.battery_level
            })
            self.mqtt_client.publish(topic, payload)
    
    def send_aggregated_data(self):
        """Send aggregated data to cloud backend"""
        if self.nodes:
            aggregated = {
                "timestamp": datetime.now().isoformat(),
                "active_nodes": len(self.nodes),
                "avg_temperature": sum(node.temperature for node in self.nodes.values()) / len(self.nodes),
                "avg_humidity": sum(node.humidity for node in self.nodes.values()) / len(self.nodes),
                "max_rainfall": max(node.rainfall for node in self.nodes.values()),
                "high_risk_count": sum(1 for node in self.nodes.values() if node.risk_score > 0.7)
            }
            
            # Send to cloud via MQTT
            if self.mqtt_client:
                self.mqtt_client.publish("cloudburst/aggregated", json.dumps(aggregated))
    
    def get_node_location(self, node_id: str) -> str:
        """Get location coordinates for a node"""
        # This would typically come from a configuration file or database
        locations = {
            "node_001": "28.6139° N, 77.2090° E",
            "node_002": "28.7041° N, 77.1025° E"
        }
        return locations.get(node_id, "Unknown location")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
        if hasattr(self, 'lora_serial'):
            self.lora_serial.close()

if __name__ == "__main__":
    gateway = GatewayController()
    try:
        gateway.process_incoming_data()
    finally:
        gateway.cleanup()
