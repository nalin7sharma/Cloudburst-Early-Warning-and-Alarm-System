#!/usr/bin/env python3
"""
Cloud Backend API for Cloudburst Early Warning System
Main Flask application handling data ingestion, ML inference, and alerts
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import asyncio
import aiohttp

from database import TimescaleDB
from ml_models.anomaly_detection_model import CloudburstPredictor
from alert_system.sms_gateway import SMSAlert
from alert_system.push_notifications import PushNotification

app = Flask(__name__)
CORS(app)

# Initialize components
db = TimescaleDB()
ml_predictor = CloudburstPredictor()
sms_alerter = SMSAlert()
push_alerter = PushNotification()

# Configuration
CONFIG = {
    'high_risk_threshold': 0.7,
    'critical_risk_threshold': 0.9,
    'alert_cooldown_minutes': 30,
    'model_retrain_hours': 24
}

class CloudburstBackend:
    def __init__(self):
        self.alert_history = []
        self.last_model_retrain = datetime.now()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cloud_backend.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def process_telemetry(self, data: Dict) -> Dict:
        """Process incoming telemetry data from gateways"""
        try:
            # Store in database
            await db.store_telemetry(data)
            
            # Get recent data for ML prediction
            recent_data = await db.get_recent_node_data(
                data['node_id'], 
                hours=6
            )
            
            if len(recent_data) >= 10:  # Minimum data points for prediction
                # Prepare features for ML model
                features = self.prepare_features(recent_data)
                
                # Get ML prediction
                prediction = await ml_predictor.predict(features)
                risk_score = prediction['risk_score']
                
                # Update risk score in database
                await db.update_node_risk(data['node_id'], risk_score)
                
                # Check if alert needed
                if risk_score > CONFIG['high_risk_threshold']:
                    await self.handle_alert(data, risk_score, prediction)
                
                return {
                    'status': 'processed',
                    'risk_score': risk_score,
                    'prediction': prediction
                }
            else:
                return {
                    'status': 'insufficient_data',
                    'message': 'Need more data points for accurate prediction'
                }
                
        except Exception as e:
            self.logger.error(f"Error processing telemetry: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def prepare_features(self, data: List[Dict]) -> pd.DataFrame:
        """Prepare features for ML model from raw data"""
        df = pd.DataFrame(data)
        
        # Calculate derived features
        features = {}
        
        # Basic statistics
        features['mean_rainfall'] = df['rainfall'].mean()
        features['max_rainfall'] = df['rainfall'].max()
        features['rainfall_trend'] = self.calculate_trend(df['rainfall'])
        
        features['mean_humidity'] = df['humidity'].mean()
        features['humidity_trend'] = self.calculate_trend(df['humidity'])
        
        features['mean_pressure'] = df['pressure'].mean()
        features['pressure_trend'] = self.calculate_trend(df['pressure'])
        features['pressure_variance'] = df['pressure'].var()
        
        features['lightning_count'] = df['lightning_count'].sum()
        
        # Time-based features
        features['hour'] = datetime.now().hour
        features['is_night'] = 1 if 20 <= datetime.now().hour <= 6 else 0
        features['season'] = self.get_season()
        
        # Rate of change features
        features['rainfall_acceleration'] = self.calculate_acceleration(df['rainfall'])
        features['pressure_drop_rate'] = self.calculate_pressure_drop_rate(df['pressure'])
        
        return pd.DataFrame([features])
    
    def calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend of a time series"""
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope
    
    def calculate_acceleration(self, series: pd.Series) -> float:
        """Calculate acceleration (second derivative) of rainfall"""
        if len(series) < 3:
            return 0.0
        # Simple second difference
        return series.iloc[-1] - 2 * series.iloc[-2] + series.iloc[-3]
    
    def calculate_pressure_drop_rate(self, series: pd.Series) -> float:
        """Calculate rate of pressure drop"""
        if len(series) < 2:
            return 0.0
        recent_pressure = series.iloc[-6:]  # Last hour
        if len(recent_pressure) < 2:
            return 0.0
        return recent_pressure.iloc[-1] - recent_pressure.iloc[0]
    
    def get_season(self) -> int:
        """Get current season (0: winter, 1: spring, 2: summer, 3: fall)"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 0  # winter
        elif month in [3, 4, 5]:
            return 1  # spring
        elif month in [6, 7, 8]:
            return 2  # summer
        else:
            return 3  # fall
    
    async def handle_alert(self, data: Dict, risk_score: float, prediction: Dict):
        """Handle alert generation and distribution"""
        node_id = data['node_id']
        
        # Check alert cooldown
        if self.is_in_cooldown(node_id):
            self.logger.info(f"Alert cooldown active for node {node_id}")
            return
        
        # Determine alert level
        if risk_score > CONFIG['critical_risk_threshold']:
            alert_level = "CRITICAL"
        else:
            alert_level = "WARNING"
        
        # Create alert message
        alert_data = {
            'alert_id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'node_id': node_id,
            'location': await db.get_node_location(node_id),
            'risk_score': risk_score,
            'alert_level': alert_level,
            'rainfall': data.get('rainfall', 0),
            'humidity': data.get('humidity', 0),
            'pressure': data.get('pressure', 0),
            'prediction_confidence': prediction.get('confidence', 0),
            'estimated_impact': prediction.get('impact_area', 'localized')
        }
        
        # Store alert
        await db.store_alert(alert_data)
        self.alert_history.append(alert_data)
        
        # Send alerts
        await self.distribute_alerts(alert_data)
        
        self.logger.warning(f"Alert generated: {alert_level} for node {node_id}, risk: {risk_score:.2f}")
    
    def is_in_cooldown(self, node_id: str) -> bool:
        """Check if node is in alert cooldown period"""
        cooldown_time = datetime.now() - timedelta(minutes=CONFIG['alert_cooldown_minutes'])
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert['node_id'] == node_id and 
            datetime.fromisoformat(alert['timestamp']) > cooldown_time
        ]
        
        return len(recent_alerts) > 0
    
    async def distribute_alerts(self, alert_data: Dict):
        """Distribute alerts through multiple channels"""
        # SMS alerts to authorities
        if alert_data['alert_level'] == "CRITICAL":
            await sms_alerter.send_critical_alert(alert_data)
        
        # Push notifications to mobile app
        await push_alerter.send_alert(alert_data)
        
        # Email alerts for system administrators
        await self.send_email_alert(alert_data)
        
        # Webhook notifications for integration with other systems
        await self.send_webhook_alerts(alert_data)
    
    async def send_email_alert(self, alert_data: Dict):
        """Send email alert to administrators"""
        # Implementation would use SMTP library
        self.logger.info(f"Email alert prepared for: {alert_data}")
    
    async def send_webhook_alerts(self, alert_data: Dict):
        """Send webhook alerts to integrated systems"""
        webhooks = await db.get_webhook_configs()
        
        async with aiohttp.ClientSession() as session:
            for webhook in webhooks:
                try:
                    async with session.post(
                        webhook['url'],
                        json=alert_data,
                        headers=webhook.get('headers', {})
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Webhook delivered to {webhook['url']}")
                        else:
                            self.logger.error(f"Webhook failed: {response.status}")
                except Exception as e:
                    self.logger.error(f"Webhook error for {webhook['url']}: {e}")
    
    async def retrain_models(self):
        """Retrain ML models with new data"""
        if datetime.now() - self.last_model_retrain < timedelta(hours=CONFIG['model_retrain_hours']):
            return
        
        self.logger.info("Starting model retraining...")
        
        try:
            # Get training data from database
            training_data = await db.get_training_data(days=30)
            
            if len(training_data) > 1000:  # Minimum data points for retraining
                await ml_predictor.retrain(training_data)
                self.last_model_retrain = datetime.now()
                self.logger.info("Model retraining completed successfully")
            else:
                self.logger.info("Insufficient data for retraining")
                
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")

# Initialize backend
backend = CloudburstBackend()

# Flask Routes
@app.route('/')
def index():
    """Root endpoint - API status"""
    return jsonify({
        'status': 'running',
        'name': 'Cloudburst Early Warning System',
        'version': '2.1.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/telemetry', methods=['POST'])
async def receive_telemetry():
    """Receive telemetry data from gateways"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['node_id', 'timestamp', 'sensor_data']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Process telemetry
        result = await backend.process_telemetry(data)
        
        return jsonify(result)
        
    except Exception as e:
        backend.logger.error(f"Error in telemetry endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/alerts', methods=['GET'])
async def get_alerts():
    """Get recent alerts"""
    try:
        hours = int(request.args.get('hours', 24))
        alert_level = request.args.get('level')
        
        alerts = await db.get_recent_alerts(hours=hours, level=alert_level)
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts),
            'timeframe_hours': hours
        })
        
    except Exception as e:
        backend.logger.error(f"Error in alerts endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/nodes', methods=['GET'])
async def get_nodes():
    """Get all registered nodes with current status"""
    try:
        nodes = await db.get_all_nodes_status()
        
        return jsonify({
            'nodes': nodes,
            'total_count': len(nodes),
            'online_count': len([n for n in nodes if n.get('online', False)]),
            'high_risk_count': len([n for n in nodes if n.get('risk_score', 0) > 0.7])
        })
        
    except Exception as e:
        backend.logger.error(f"Error in nodes endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predictions', methods=['POST'])
async def get_prediction():
    """Get ML prediction for specific conditions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Use ML model to generate prediction
        prediction = await ml_predictor.predict(data)
        
        return jsonify({
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        backend.logger.error(f"Error in prediction endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/system/health', methods=['GET'])
async def system_health():
    """Get system health status"""
    try:
        health_data = {
            'status': 'healthy',
            'database': await db.check_health(),
            'ml_model': await ml_predictor.check_health(),
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - backend.start_time),
            'active_alerts': len(backend.alert_history),
            'total_requests': backend.request_count
        }
        
        return jsonify(health_data)
        
    except Exception as e:
        backend.logger.error(f"Error in health endpoint: {e}")
        return jsonify({'status': 'degraded', 'error': str(e)}), 500

# Background tasks
@app.before_request
def before_request():
    """Initialize request context"""
    backend.request_count += 1

async def background_tasks():
    """Run background tasks periodically"""
    while True:
        try:
            # Retrain models if needed
            await backend.retrain_models()
            
            # Clean up old data
            await db.cleanup_old_data(days=7)
            
            # Check system health
            await backend.check_system_health()
            
        except Exception as e:
            backend.logger.error(f"Background task error: {e}")
        
        await asyncio.sleep(3600)  # Run every hour

if __name__ == '__main__':
    # Start background tasks
    asyncio.create_task(background_tasks())
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
