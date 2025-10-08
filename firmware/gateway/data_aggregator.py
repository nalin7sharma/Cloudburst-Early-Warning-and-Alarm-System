#!/usr/bin/env python3
"""
Data Aggregator for Gateway - Processes and correlates data from multiple nodes
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from dataclasses import dataclass

@dataclass
class AggregatedData:
    timestamp: datetime
    node_count: int
    spatial_risk_score: float
    rainfall_intensity: float
    pressure_trend: float
    alert_recommendation: str

class DataAggregator:
    def __init__(self, time_window_minutes: int = 30):
        self.time_window = timedelta(minutes=time_window_minutes)
        self.node_data_buffer: Dict[str, List] = {}
        self.logger = logging.getLogger(__name__)
        
    def add_node_data(self, node_id: str, data: dict):
        """Add node data to aggregation buffer"""
        if node_id not in self.node_data_buffer:
            self.node_data_buffer[node_id] = []
        
        self.node_data_buffer[node_id].append(data)
        
        # Remove old data outside time window
        cutoff_time = datetime.now() - self.time_window
        self.node_data_buffer[node_id] = [
            d for d in self.node_data_buffer[node_id] 
            if datetime.fromisoformat(d['timestamp']) > cutoff_time
        ]
    
    def calculate_spatial_risk(self) -> float:
        """Calculate spatial risk score across all nodes"""
        if not self.node_data_buffer:
            return 0.0
        
        recent_risks = []
        for node_data in self.node_data_buffer.values():
            if node_data:
                latest = node_data[-1]
                recent_risks.append(latest.get('risk_score', 0.0))
        
        return np.mean(recent_risks) if recent_risks else 0.0
    
    def detect_rainfall_pattern(self) -> Dict:
        """Detect rainfall patterns across the network"""
        rainfall_data = []
        
        for node_id, data_list in self.node_data_buffer.items():
            if data_list:
                latest = data_list[-1]
                rainfall_data.append({
                    'node_id': node_id,
                    'rainfall': latest.get('rainfall', 0.0),
                    'timestamp': latest.get('timestamp')
                })
        
        if not rainfall_data:
            return {'intensity': 0.0, 'trend': 'stable'}
        
        df = pd.DataFrame(rainfall_data)
        max_rainfall = df['rainfall'].max()
        avg_rainfall = df['rainfall'].mean()
        
        # Calculate trend (simplified)
        if max_rainfall > 50:
            trend = 'extreme'
        elif max_rainfall > 20:
            trend = 'high'
        elif max_rainfall > 10:
            trend = 'moderate'
        else:
            trend = 'low'
        
        return {
            'max_intensity': max_rainfall,
            'average_intensity': avg_rainfall,
            'trend': trend,
            'affected_nodes': len(rainfall_data)
        }
    
    def analyze_pressure_trends(self) -> Dict:
        """Analyze atmospheric pressure trends"""
        pressure_data = []
        
        for node_id, data_list in self.node_data_buffer.items():
            if len(data_list) >= 2:
                # Calculate pressure change over last hour
                recent_pressures = [d.get('pressure', 1013.25) for d in data_list[-6:]]  # Last hour
                if len(recent_pressures) >= 2:
                    pressure_change = recent_pressures[-1] - recent_pressures[0]
                    pressure_data.append({
                        'node_id': node_id,
                        'pressure_change': pressure_change,
                        'current_pressure': recent_pressures[-1]
                    })
        
        if not pressure_data:
            return {'average_change': 0.0, 'trend': 'stable'}
        
        df = pd.DataFrame(pressure_data)
        avg_change = df['pressure_change'].mean()
        
        if avg_change < -5:
            trend = 'rapid_drop'
        elif avg_change < -2:
            trend = 'dropping'
        elif avg_change > 2:
            trend = 'rising'
        else:
            trend = 'stable'
        
        return {
            'average_change': avg_change,
            'trend': trend,
            'nodes_with_drop': len([p for p in pressure_data if p['pressure_change'] < -2])
        }
    
    def generate_aggregated_report(self) -> AggregatedData:
        """Generate comprehensive aggregated report"""
        spatial_risk = self.calculate_spatial_risk()
        rainfall_analysis = self.detect_rainfall_pattern()
        pressure_analysis = self.analyze_pressure_trends()
        
        # Determine alert recommendation
        if spatial_risk > 0.8 and rainfall_analysis['max_intensity'] > 30:
            recommendation = "IMMEDIATE_ALERT"
        elif spatial_risk > 0.6 and rainfall_analysis['max_intensity'] > 20:
            recommendation = "HIGH_ALERT"
        elif spatial_risk > 0.4:
            recommendation = "WATCH"
        else:
            recommendation = "NORMAL"
        
        return AggregatedData(
            timestamp=datetime.now(),
            node_count=len(self.node_data_buffer),
            spatial_risk_score=spatial_risk,
            rainfall_intensity=rainfall_analysis['max_intensity'],
            pressure_trend=pressure_analysis['average_change'],
            alert_recommendation=recommendation
        )
    
    def get_anomaly_nodes(self) -> List[str]:
        """Get list of nodes showing anomalous behavior"""
        anomalous_nodes = []
        
        for node_id, data_list in self.node_data_buffer.items():
            if data_list:
                latest = data_list[-1]
                if latest.get('risk_score', 0.0) > 0.7:
                    anomalous_nodes.append({
                        'node_id': node_id,
                        'risk_score': latest.get('risk_score'),
                        'rainfall': latest.get('rainfall'),
                        'timestamp': latest.get('timestamp')
                    })
        
        return anomalous_nodes
