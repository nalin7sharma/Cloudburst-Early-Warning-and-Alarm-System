#!/usr/bin/env python3
"""
Anomaly Detection Model for Cloudburst Prediction
Multiple ML models for detecting abnormal weather patterns
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio

class AnomalyDetector:
    """Main anomaly detection class combining multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.is_trained = False
        self.logger = self.setup_logging()
        
        # Initialize models
        self.initialize_models()
    
    def setup_logging(self):
        """Setup logging for anomaly detection"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def initialize_models(self):
        """Initialize multiple anomaly detection models"""
        
        # Isolation Forest for point anomalies
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # One-Class SVM for novelty detection
        self.models['one_class_svm'] = OneClassSVM(
            nu=0.1,
            kernel='rbf',
            gamma='scale'
        )
        
        # Random Forest for classification
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        # DBSCAN for clustering anomalies
        self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
        
        # LSTM Autoencoder for time series anomalies
        self.models['lstm_autoencoder'] = self.build_lstm_autoencoder()
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = tf.keras.layers.Normalization()
    
    def build_lstm_autoencoder(self) -> tf.keras.Model:
        """Build LSTM Autoencoder for time series anomaly detection"""
        model = Sequential([
            # Encoder
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=(None, 7)),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=False),
            Dropout(0.2),
            
            # Bottleneck
            Dense(16, activation='relu'),
            
            # Decoder
            Dense(32, activation='relu'),
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            Dense(7, activation='linear')  # Reconstruct original features
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def train_models(self, training_data: pd.DataFrame, labels: Optional[pd.Series] = None):
        """Train all anomaly detection models"""
        self.logger.info("Starting model training...")
        
        try:
            # Prepare features
            features = self.prepare_features(training_data)
            scaled_features = self.scalers['standard'].fit_transform(features)
            
            # Train Isolation Forest (unsupervised)
            self.models['isolation_forest'].fit(scaled_features)
            
            # Train One-Class SVM (unsupervised)
            self.models['one_class_svm'].fit(scaled_features)
            
            # Train Random Forest (supervised if labels available)
            if labels is not None:
                self.models['random_forest'].fit(scaled_features, labels)
            
            # Train LSTM Autoencoder
            if len(training_data) > 100:
                sequences = self.create_sequences(training_data, sequence_length=24)
                await self.train_lstm_autoencoder(sequences)
            
            # Calculate anomaly thresholds
            self.calculate_thresholds(scaled_features)
            
            self.is_trained = True
            self.logger.info("All models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            raise
    
    async def train_lstm_autoencoder(self, sequences: np.ndarray, epochs: int = 50):
        """Train LSTM Autoencoder"""
        self.logger.info("Training LSTM Autoencoder...")
        
        # The autoencoder learns to reconstruct normal patterns
        history = self.models['lstm_autoencoder'].fit(
            sequences, sequences,  # Input and target are the same for autoencoder
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.logger.info("LSTM Autoencoder training completed")
        return history
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = []
        
        # Basic sensor readings
        basic_features = ['temperature', 'humidity', 'pressure', 'rainfall']
        features.extend(data[basic_features].values)
        
        # Derived features
        data['dew_point'] = self.calculate_dew_point(data['temperature'], data['humidity'])
        data['humidity_ratio'] = data['humidity'] / 100.0
        data['pressure_trend'] = data['pressure'].diff().fillna(0)
        data['rainfall_intensity'] = data['rainfall'].rolling(6, min_periods=1).max()
        
        derived_features = ['dew_point', 'humidity_ratio', 'pressure_trend', 'rainfall_intensity']
        features.extend(data[derived_features].values)
        
        # Statistical features
        if len(data) > 10:
            data['rainfall_std'] = data['rainfall'].rolling(10, min_periods=1).std()
            data['pressure_std'] = data['pressure'].rolling(10, min_periods=1).std()
            data['temp_humidity_corr'] = data['temperature'].rolling(10, min_periods=1).corr(data['humidity'])
            
            statistical_features = ['rainfall_std', 'pressure_std', 'temp_humidity_corr']
            features.extend(data[statistical_features].values)
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_year'] = pd.to_datetime(data['timestamp']).dt.dayofyear
            data['is_night'] = ((data['hour'] >= 20) | (data['hour'] <= 6)).astype(int)
            
            time_features = ['hour', 'day_of_year', 'is_night']
            features.extend(data[time_features].values)
        
        return np.column_stack(features)
    
    def calculate_dew_point(self, temperature: float, humidity: float) -> float:
        """Calculate dew point from temperature and humidity"""
        # Magnus formula approximation
        alpha = ((17.27 * temperature) / (237.7 + temperature)) + np.log(humidity / 100.0)
        dew_point = (237.7 * alpha) / (17.27 - alpha)
        return dew_point
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 24) -> np.ndarray:
        """Create sequences for LSTM model"""
        sequences = []
        feature_cols = ['temperature', 'humidity', 'pressure', 'rainfall', 
                       'dew_point', 'pressure_trend', 'rainfall_intensity']
        
        values = data[feature_cols].values
        
        for i in range(len(values) - sequence_length):
            seq = values[i:i + sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def calculate_thresholds(self, features: np.ndarray):
        """Calculate anomaly thresholds based on training data"""
        # Get predictions from unsupervised models
        if_scores = self.models['isolation_forest'].decision_function(features)
        svm_scores = self.models['one_class_svm'].decision_function(features)
        
        # Set thresholds at 10th percentile (most anomalous)
        self.thresholds['isolation_forest'] = np.percentile(if_scores, 10)
        self.thresholds['one_class_svm'] = np.percentile(svm_scores, 10)
        
        self.logger.info(f"Anomaly thresholds calculated: {self.thresholds}")
    
    async def detect_anomalies(self, data: pd.DataFrame) -> Dict:
        """
        Detect anomalies using ensemble of models
        Returns comprehensive anomaly analysis
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before detection")
        
        try:
            # Prepare features
            features = self.prepare_features(data)
            scaled_features = self.scalers['standard'].transform(features)
            
            # Get predictions from all models
            predictions = {}
            
            # Isolation Forest
            if_scores = self.models['isolation_forest'].decision_function(scaled_features)
            predictions['isolation_forest'] = if_scores < self.thresholds['isolation_forest']
            
            # One-Class SVM
            svm_scores = self.models['one_class_svm'].decision_function(scaled_features)
            predictions['one_class_svm'] = svm_scores < self.thresholds['one_class_svm']
            
            # Random Forest (if trained with labels)
            if hasattr(self.models['random_forest'], 'predict'):
                rf_predictions = self.models['random_forest'].predict(scaled_features)
                predictions['random_forest'] = rf_predictions == 1  # Assuming 1 is anomaly
            
            # LSTM Autoencoder reconstruction error
            if len(data) >= 24:
                sequences = self.create_sequences(data.tail(24), sequence_length=24)
                if len(sequences) > 0:
                    reconstructions = self.models['lstm_autoencoder'].predict(sequences[-1:])
                    reconstruction_error = np.mean(np.square(sequences[-1] - reconstructions))
                    predictions['lstm_autoencoder'] = reconstruction_error > 0.1  # Threshold
            
            # Ensemble voting
            ensemble_score = self.ensemble_voting(predictions, data)
            
            # Feature importance for explanation
            feature_importance = await self.analyze_feature_importance(data, ensemble_score)
            
            result = {
                'is_anomaly': ensemble_score['is_anomaly'],
                'confidence': ensemble_score['confidence'],
                'anomaly_type': self.classify_anomaly_type(data),
                'severity': ensemble_score['severity'],
                'model_agreement': ensemble_score['agreement'],
                'feature_contributions': feature_importance,
                'timestamp': datetime.now().isoformat(),
                'recommended_action': self.get_recommended_action(ensemble_score)
            }
            
            self.logger.info(f"Anomaly detection completed: {result['is_anomaly']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def ensemble_voting(self, predictions: Dict, data: pd.DataFrame) -> Dict:
        """Combine predictions from multiple models"""
        model_weights = {
            'isolation_forest': 0.3,
            'one_class_svm': 0.25,
            'random_forest': 0.3,
            'lstm_autoencoder': 0.15
        }
        
        total_weight = 0
        anomaly_score = 0
        
        for model_name, weight in model_weights.items():
            if model_name in predictions and predictions[model_name] is not None:
                if isinstance(predictions[model_name], (bool, np.bool_)):
                    anomaly_score += weight * int(predictions[model_name])
                elif isinstance(predictions[model_name], (np.ndarray, list)):
                    # Use the most recent prediction for time series
                    anomaly_score += weight * int(predictions[model_name][-1] if len(predictions[model_name]) > 0 else 0)
                total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            anomaly_score /= total_weight
        
        # Consider recent trends
        trend_boost = self.analyze_recent_trends(data)
        final_score = min(1.0, anomaly_score + trend_boost)
        
        # Count model agreement
        agreement_count = sum(1 for model_name in model_weights 
                            if model_name in predictions and predictions[model_name] is not None)
        
        return {
            'is_anomaly': final_score > 0.6,  # Threshold for anomaly
            'confidence': final_score,
            'severity': self.calculate_severity(final_score, data),
            'agreement': agreement_count
        }
    
    def analyze_recent_trends(self, data: pd.DataFrame) -> float:
        """Analyze recent trends to boost anomaly score"""
        if len(data) < 10:
            return 0.0
        
        trend_score = 0.0
        
        # Rainfall acceleration
        recent_rain = data['rainfall'].tail(6).values
        if len(recent_rain) >= 3:
            rain_acceleration = recent_rain[-1] - 2 * recent_rain[-2] + recent_rain[-3]
            trend_score += min(0.3, rain_acceleration / 50.0)  # Normalize
        
        # Pressure drop rate
        recent_pressure = data['pressure'].tail(6).values
        if len(recent_pressure) >= 2:
            pressure_drop = recent_pressure[0] - recent_pressure[-1]
            trend_score += min(0.2, pressure_drop / 20.0)  # Normalize
        
        # Humidity increase
        recent_humidity = data['humidity'].tail(6).values
        if len(recent_humidity) >= 2:
            humidity_increase = recent_humidity[-1] - recent_humidity[0]
            trend_score += min(0.2, humidity_increase / 50.0)  # Normalize
        
        return trend_score
    
    def calculate_severity(self, anomaly_score: float, data: pd.DataFrame) -> str:
        """Calculate anomaly severity"""
        if anomaly_score > 0.9:
            return "CRITICAL"
        elif anomaly_score > 0.7:
            return "HIGH"
        elif anomaly_score > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def classify_anomaly_type(self, data: pd.DataFrame) -> str:
        """Classify the type of weather anomaly"""
        if len(data) < 5:
            return "UNKNOWN"
        
        recent = data.tail(5)
        
        # Check for cloudburst patterns
        high_rainfall = recent['rainfall'].max() > 30
        high_humidity = recent['humidity'].mean() > 90
        pressure_drop = recent['pressure'].iloc[0] - recent['pressure'].iloc[-1] > 5
        
        if high_rainfall and high_humidity and pressure_drop:
            return "CLOUDBURST_IMMINENT"
        elif high_rainfall and high_humidity:
            return "HEAVY_RAINFALL"
        elif pressure_drop and high_humidity:
            return "PRESSURE_SYSTEM"
        elif high_rainfall:
            return "RAINFALL_ANOMALY"
        else:
            return "WEATHER_ANOMALY"
    
    async def analyze_feature_contributions(self, data: pd.DataFrame, anomaly_score: float) -> Dict:
        """Analyze which features contributed most to anomaly detection"""
        if len(data) == 0:
            return {}
        
        recent_data = data.tail(1)
        features = self.prepare_features(recent_data)
        
        # Simple feature importance based on deviation from mean
        feature_importance = {}
        
        # Basic features
        basic_features = ['temperature', 'humidity', 'pressure', 'rainfall']
        for i, feature in enumerate(basic_features):
            if i < features.shape[1]:
                # Use z-score like measure
                importance = abs(features[0, i] - np.mean(features[:, i])) / (np.std(features[:, i]) + 1e-8)
                feature_importance[feature] = float(importance)
        
        # Normalize importance scores
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= max_importance
        
        return feature_importance
    
    def get_recommended_action(self, ensemble_score: Dict) -> str:
        """Get recommended action based on anomaly severity"""
        severity = ensemble_score['severity']
        
        actions = {
            "CRITICAL": "IMMEDIATE_EVACUATION - Cloudburst detected within 30 minutes",
            "HIGH": "SEEK_SHELTER - Heavy rainfall expected within 1 hour",
            "MEDIUM": "STAY_ALERT - Monitor weather updates closely",
            "LOW": "CONTINUE_MONITORING - No immediate action required"
        }
        
        return actions.get(severity, "CONTINUE_MONITORING")
    
    async def evaluate_model_performance(self, test_data: pd.DataFrame, true_labels: pd.Series) -> Dict:
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        predictions = []
        
        # Make predictions for test data
        for i in range(len(test_data)):
            sample_data = test_data.iloc[:i+1]
            result = await self.detect_anomalies(sample_data)
            predictions.append(result['is_anomaly'])
        
        # Calculate metrics
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': np.mean(np.array(predictions) == true_labels.values),
            'total_samples': len(test_data),
            'anomaly_rate': np.mean(predictions)
        }
        
        self.logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def save_models(self, filepath: str = 'anomaly_models'):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save sklearn models
        for name, model in self.models.items():
            if name != 'lstm_autoencoder':  # Keras model saved separately
                joblib.dump(model, f'{filepath}/{name}.pkl')
        
        # Save Keras model
        self.models['lstm_autoencoder'].save(f'{filepath}/lstm_autoencoder.h5')
        
        # Save scalers and thresholds
        joblib.dump(self.scalers, f'{filepath}/scalers.pkl')
        joblib.dump(self.thresholds, f'{filepath}/thresholds.pkl')
        
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = 'anomaly_models'):
        """Load trained models from disk"""
        try:
            # Load sklearn models
            for name in ['isolation_forest', 'one_class_svm', 'random_forest', 'dbscan']:
                self.models[name] = joblib.load(f'{filepath}/{name}.pkl')
            
            # Load Keras model
            self.models['lstm_autoencoder'] = tf.keras.models.load_model(f'{filepath}/lstm_autoencoder.h5')
            
            # Load scalers and thresholds
            self.scalers = joblib.load(f'{filepath}/scalers.pkl')
            self.thresholds = joblib.load(f'{filepath}/thresholds.pkl')
            
            self.is_trained = True
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise

# Example usage
async def main():
    """Example usage of the AnomalyDetector"""
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(70, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000),
        'rainfall': np.random.gamma(2, 2, 1000)
    })
    
    # Add some anomalies
    anomaly_indices = [200, 400, 600, 800]
    sample_data.loc[anomaly_indices, 'rainfall'] = 50  # Heavy rainfall
    sample_data.loc[anomaly_indices, 'humidity'] = 95  # Very high humidity
    
    # Create labels (1 for anomaly, 0 for normal)
    labels = pd.Series(0, index=sample_data.index)
    labels.iloc[anomaly_indices] = 1
    
    # Initialize and train detector
    detector = AnomalyDetector()
    await detector.train_models(sample_data, labels)
    
    # Test detection
    test_sample = sample_data.tail(50)
    result = await detector.detect_anomalies(test_sample)
    
    print("Anomaly Detection Result:")
    print(f"Is Anomaly: {result['is_anomaly']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Anomaly Type: {result['anomaly_type']}")
    print(f"Severity: {result['severity']}")
    print(f"Recommended Action: {result['recommended_action']}")

if __name__ == '__main__':
    asyncio.run(main())
