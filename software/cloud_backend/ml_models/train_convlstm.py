#!/usr/bin/env python3
"""
ConvLSTM Model Training for Cloudburst Prediction
Advanced deep learning model for time-series weather data
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    ConvLSTM2D, LSTM, Dense, Dropout, BatchNormalization,
    Reshape, TimeDistributed, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import joblib
import os

class ConvLSTMPredictor:
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 6):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.logger = self.setup_logging()
        
        # Model parameters
        self.input_shape = None
        self.n_features = None
        
    def setup_logging(self):
        """Setup training logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def build_model(self, input_shape: tuple) -> Sequential:
        """
        Build ConvLSTM model architecture
        Input shape: (sequences, timesteps, features, 1)
        """
        self.logger.info(f"Building ConvLSTM model with input shape: {input_shape}")
        
        model = Sequential([
            # ConvLSTM layers for spatiotemporal patterns
            ConvLSTM2D(
                filters=64,
                kernel_size=(1, 3),
                activation='relu',
                input_shape=input_shape,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            BatchNormalization(),
            
            ConvLSTM2D(
                filters=32,
                kernel_size=(1, 2),
                activation='relu',
                return_sequences=False,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            BatchNormalization(),
            
            # Dense layers for final prediction
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            # Output layer: risk score (0-1) and additional metrics
            Dense(3, activation='sigmoid')  # [risk_score, confidence, impact_scale]
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'accuracy']
        )
        
        self.logger.info("Model built successfully")
        return model
    
    def prepare_sequences(self, data: pd.DataFrame, features: list) -> tuple:
        """
        Prepare sequences for ConvLSTM training
        Converts tabular data into spatiotemporal sequences
        """
        self.logger.info("Preparing sequences for training")
        
        # Select and scale features
        feature_data = data[features].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        sequences = []
        targets = []
        
        # Create sequences
        for i in range(len(scaled_features) - self.sequence_length - self.prediction_horizon):
            # Input sequence
            seq = scaled_features[i:i + self.sequence_length]
            
            # Target: risk indicators for future horizon
            future_data = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            # Calculate target risk score based on future rainfall
            future_rainfall = future_data['rainfall'].values
            max_rainfall = np.max(future_rainfall)
            
            # Risk score calculation (simplified)
            if max_rainfall > 50:
                risk_score = 1.0
            elif max_rainfall > 30:
                risk_score = 0.7
            elif max_rainfall > 15:
                risk_score = 0.4
            else:
                risk_score = 0.1
            
            # Additional target: prediction confidence
            confidence = min(1.0, len(future_data) / self.prediction_horizon)
            
            # Impact scale based on rainfall intensity and duration
            impact_scale = max_rainfall * len(future_data) / 100
            
            target = [risk_score, confidence, min(impact_scale, 1.0)]
            
            # Reshape sequence for ConvLSTM (samples, timesteps, features, 1)
            seq_reshaped = seq.reshape((self.sequence_length, len(features), 1))
            
            sequences.append(seq_reshaped)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        self.logger.info(f"Created {len(sequences)} sequences")
        return sequences, targets
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2, epochs: int = 100):
        """
        Train the ConvLSTM model
        """
        self.logger.info("Starting model training")
        
        # Define features for training
        features = [
            'temperature', 'humidity', 'pressure', 'rainfall', 
            'wind_speed', 'lightning_count', 'dew_point'
        ]
        
        # Prepare sequences
        X, y = self.prepare_sequences(data, features)
        
        # Set input shape
        self.input_shape = (X.shape[1], X.shape[2], X.shape[3])
        self.n_features = X.shape[2]
        
        # Build model
        self.model = self.build_model(self.input_shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        self.logger.info(f"Training data: {X_train.shape}, Validation data: {X_val.shape}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(
                'best_convlstm_model.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('convlstm_cloudburst_model.h5')
        
        # Save scalers
        joblib.dump(self.feature_scaler, 'feature_scaler.pkl')
        
        self.logger.info("Model training completed")
        
        return history
    
    def predict(self, sequence: np.ndarray) -> dict:
        """
        Make prediction using trained model
        """
        if self.model is None:
            self.load_model()
        
        # Ensure sequence has correct shape
        if sequence.ndim == 2:
            sequence = sequence.reshape((1, sequence.shape[0], sequence.shape[1], 1))
        
        # Make prediction
        prediction = self.model.predict(sequence)[0]
        
        risk_score, confidence, impact_scale = prediction
        
        return {
            'risk_score': float(risk_score),
            'confidence': float(confidence),
            'impact_scale': float(impact_scale),
            'alert_level': self.get_alert_level(risk_score),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_alert_level(self, risk_score: float) -> str:
        """Convert risk score to alert level"""
        if risk_score > 0.8:
            return "CRITICAL"
        elif risk_score > 0.6:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def load_model(self, model_path: str = 'convlstm_cloudburst_model.h5'):
        """Load pre-trained model and scalers"""
        self.logger.info("Loading pre-trained model")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        self.feature_scaler = joblib.load('feature_scaler.pkl')
        
        # Infer input shape from model
        self.input_shape = self.model.input_shape[1:]
        self.n_features = self.input_shape[1]
        
        self.logger.info("Model loaded successfully")
    
    def evaluate_model(self, test_data: pd.DataFrame):
        """Evaluate model performance on test data"""
        self.logger.info("Evaluating model performance")
        
        features = [
            'temperature', 'humidity', 'pressure', 'rainfall', 
            'wind_speed', 'lightning_count', 'dew_point'
        ]
        
        X_test, y_test = self.prepare_sequences(test_data, features)
        
        if self.model is None:
            self.load_model()
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'test_samples': len(X_test)
        }
        
        self.logger.info(f"Model evaluation completed: {metrics}")
        
        return metrics, predictions, y_test
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate sample training data for demonstration
    In production, this would come from actual sensor data
    """
    np.random.seed(42)
    
    timestamps = pd.date_range(
        start='2024-01-01',
        end='2024-10-01',
        freq='H'
    )[:n_samples]
    
    data = {
        'timestamp': timestamps,
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.normal(70, 15, n_samples),
        'pressure': np.random.normal(1013, 10, n_samples),
        'rainfall': np.random.gamma(2, 2, n_samples),  # Right-skewed for rainfall
        'wind_speed': np.random.gamma(1, 3, n_samples),
        'lightning_count': np.random.poisson(0.1, n_samples),
        'dew_point': np.random.normal(18, 4, n_samples)
    }
    
    # Add some cloudburst patterns
    cloudburst_indices = np.random.choice(n_samples, size=50, replace=False)
    for idx in cloudburst_indices:
        data['rainfall'][idx:idx+6] = np.random.uniform(40, 80, 6)  # Heavy rainfall
        data['humidity'][idx:idx+6] = np.random.uniform(90, 99, 6)  # Very high humidity
        data['pressure'][idx:idx+6] = np.random.uniform(1000, 1005, 6)  # Low pressure
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Example usage
    predictor = ConvLSTMPredictor()
    
    # Generate sample data (replace with real data)
    sample_data = generate_sample_data(5000)
    
    # Train model
    history = predictor.train(sample_data, epochs=50)
    
    # Plot training history
    predictor.plot_training_history(history)
    
    # Evaluate model
    test_data = generate_sample_data(1000)
    metrics, predictions, actual = predictor.evaluate_model(test_data)
    
    print("Model training and evaluation completed!")
    print(f"Performance metrics: {metrics}")
