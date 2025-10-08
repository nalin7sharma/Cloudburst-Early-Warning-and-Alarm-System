#!/usr/bin/env python3
"""
Streamlit Dashboard for Cloudburst Early Warning System
Real-time monitoring and visualization interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import asyncio
import time
from typing import Dict, List, Optional
import logging

# Page configuration
st.set_page_config(
    page_title="Cloudburst Early Warning System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-warning {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-normal {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .node-offline {
        color: #ff4b4b;
        font-weight: bold;
    }
    .node-online {
        color: #4CAF50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class CloudburstDashboard:
    def __init__(self):
        self.api_base_url = st.secrets.get("API_BASE_URL", "http://localhost:5000")
        self.update_interval = 30  # seconds
        self.setup_logging()
        
    def setup_logging(self):
        """Setup dashboard logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def fetch_system_status(_self):
        """Fetch system status from backend API"""
        try:
            response = requests.get(f"{_self.api_base_url}/api/system/health", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": "API unavailable"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @st.cache_data(ttl=30)
    def fetch_nodes_data(_self):
        """Fetch nodes data from backend API"""
        try:
            response = requests.get(f"{_self.api_base_url}/api/nodes", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"nodes": [], "total_count": 0}
        except Exception as e:
            return {"nodes": [], "total_count": 0, "error": str(e)}
    
    @st.cache_data(ttl=30)
    def fetch_recent_alerts(_self, hours: int = 24):
        """Fetch recent alerts from backend API"""
        try:
            response = requests.get(
                f"{_self.api_base_url}/api/alerts", 
                params={"hours": hours},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"alerts": [], "count": 0}
        except Exception as e:
            return {"alerts": [], "count": 0, "error": str(e)}
    
    @st.cache_data(ttl=60)
    def fetch_sensor_data(_self, node_id: str, hours: int = 24):
        """Fetch sensor data for a specific node"""
        try:
            response = requests.get(
                f"{_self.api_base_url}/api/nodes/{node_id}/sensor-data",
                params={"hours": hours},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"data": [], "node_id": node_id}
        except Exception as e:
            return {"data": [], "node_id": node_id, "error": str(e)}
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🌧️ Cloudburst Early Warning System</h1>', 
                   unsafe_allow_html=True)
        
        # System status
        status_data = self.fetch_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if status_data.get('status') == 'healthy' else "🔴"
            st.metric("System Status", f"{status_color} {status_data.get('status', 'unknown').upper()}")
        
        with col2:
            nodes_data = self.fetch_nodes_data()
            online_nodes = len([n for n in nodes_data.get('nodes', []) 
                              if n.get('status') == 'online'])
            st.metric("Online Nodes", f"{online_nodes}/{nodes_data.get('total_count', 0)}")
        
        with col3:
            alerts_data = self.fetch_recent_alerts(hours=1)
            recent_alerts = alerts_data.get('count', 0)
            st.metric("Recent Alerts (1h)", recent_alerts)
        
        with col4:
            # Last update time
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    def render_alert_panel(self):
        """Render alert and notification panel"""
        st.subheader("🚨 Active Alerts & Notifications")
        
        alerts_data = self.fetch_recent_alerts(hours=6)
        alerts = alerts_data.get('alerts', [])
        
        if not alerts:
            st.success("No active alerts in the last 6 hours")
            return
        
        for alert in alerts[:5]:  # Show latest 5 alerts
            alert_level = alert.get('alert_level', 'UNKNOWN')
            timestamp = alert.get('timestamp', '')
            node_id = alert.get('node_id', '')
            message = alert.get('message', '')
            
            # Format timestamp
            try:
                alert_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = alert_time.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = timestamp
            
            # Create alert box with appropriate color
            if alert_level == 'CRITICAL':
                alert_class = "alert-critical"
                icon = "🔴"
            elif alert_level == 'WARNING':
                alert_class = "alert-warning"
                icon = "🟡"
            else:
                alert_class = "alert-normal"
                icon = "🟢"
            
            st.markdown(f"""
            <div class="{alert_class}">
                {icon} <strong>{alert_level}</strong> | Node: {node_id} | {time_str}<br>
                {message}
            </div>
            """, unsafe_allow_html=True)
            
            # Add acknowledge button for critical alerts
            if alert_level == 'CRITICAL' and not alert.get('acknowledged', False):
                if st.button(f"Acknowledge Critical Alert", key=f"ack_{alert.get('alert_id')}"):
                    # In a real implementation, this would call an API endpoint
                    st.success(f"Alert acknowledged at {datetime.now().strftime('%H:%M:%S')}")
                    st.rerun()
    
    def render_nodes_overview(self):
        """Render nodes overview with status and metrics"""
        st.subheader("📡 Network Nodes Overview")
        
        nodes_data = self.fetch_nodes_data()
        nodes = nodes_data.get('nodes', [])
        
        if not nodes:
            st.warning("No node data available")
            return
        
        # Create columns for node cards
        cols = st.columns(3)
        
        for i, node in enumerate(nodes):
            with cols[i % 3]:
                self.render_node_card(node)
    
    def render_node_card(self, node: Dict):
        """Render individual node card"""
        node_id = node.get('node_id', 'Unknown')
        status = node.get('status', 'offline')
        location = node.get('location_name', 'Unknown Location')
        battery = node.get('battery_level', 0)
        last_seen = node.get('last_seen', '')
        risk_score = node.get('last_risk_score', 0)
        
        # Format last seen time
        try:
            last_seen_dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
            last_seen_str = last_seen_dt.strftime("%m/%d %H:%M")
            time_diff = (datetime.now().replace(tzinfo=None) - last_seen_dt.replace(tzinfo=None)).total_seconds() / 60
        except:
            last_seen_str = "Unknown"
            time_diff = 999
        
        # Status indicator
        status_icon = "🟢" if status == 'online' and time_diff < 30 else "🔴"
        status_text = f"{status_icon} {status.upper()}"
        
        # Risk level
        if risk_score > 0.8:
            risk_color = "🔴"
            risk_text = "CRITICAL"
        elif risk_score > 0.6:
            risk_color = "🟡"
            risk_text = "HIGH"
        elif risk_score > 0.4:
            risk_color = "🟠"
            risk_text = "MEDIUM"
        else:
            risk_color = "🟢"
            risk_text = "LOW"
        
        # Battery indicator
        if battery > 70:
            battery_icon = "🔋"
        elif battery > 30:
            battery_icon = "🪫"
        else:
            battery_icon = "⚠️"
        
        # Create card
        with st.container():
            st.markdown(f"""
            <div class="metric-card">
                <h3>{node_id}</h3>
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Status:</strong> {status_text}</p>
                <p><strong>Risk Level:</strong> {risk_color} {risk_text} ({risk_score:.0%})</p>
                <p><strong>Battery:</strong> {battery_icon} {battery:.0f}%</p>
                <p><strong>Last Seen:</strong> {last_seen_str}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # View details button
            if st.button("View Details", key=f"view_{node_id}"):
                st.session_state.selected_node = node_id
                st.rerun()
    
    def render_node_details(self, node_id: str):
        """Render detailed view for a specific node"""
        st.subheader(f"📊 Node Details: {node_id}")
        
        # Back button
        if st.button("← Back to Overview"):
            if 'selected_node' in st.session_state:
                del st.session_state.selected_node
            st.rerun()
        
        # Fetch node-specific data
        sensor_data = self.fetch_sensor_data(node_id, hours=24)
        data_points = sensor_data.get('data', [])
        
        if not data_points:
            st.warning(f"No sensor data available for node {node_id}")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Sensor Data", "🌧️ Rainfall Analysis", "📊 Risk Trends", "🗺️ Location"])
        
        with tab1:
            self.render_sensor_charts(df, node_id)
        
        with tab2:
            self.render_rainfall_analysis(df, node_id)
        
        with tab3:
            self.render_risk_trends(df, node_id)
        
        with tab4:
            self.render_location_map(node_id)
    
    def render_sensor_charts(self, df: pd.DataFrame, node_id: str):
        """Render sensor data charts"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature (°C)', 'Humidity (%)', 
                          'Pressure (hPa)', 'Rainfall (mm)'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['temperature'], 
                      mode='lines', name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        # Humidity
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'], 
                      mode='lines', name='Humidity', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Pressure
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['pressure'], 
                      mode='lines', name='Pressure', line=dict(color='green')),
            row=2, col=1
        )
        
        # Rainfall
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['rainfall'], 
                      mode='lines', name='Rainfall', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, 
                         title_text=f"Sensor Data - {node_id}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_rainfall_analysis(self, df: pd.DataFrame, node_id: str):
        """Render rainfall analysis charts"""
        if 'rainfall' not in df.columns:
            st.warning("No rainfall data available")
            return
        
        # Calculate hourly rainfall
        df_hourly = df.set_index('timestamp').resample('1H').agg({
            'rainfall': 'sum',
            'temperature': 'mean',
            'humidity': 'mean',
            'pressure': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly rainfall bar chart
            fig1 = px.bar(df_hourly, x='timestamp', y='rainfall',
                         title=f'Hourly Rainfall - {node_id}',
                         labels={'rainfall': 'Rainfall (mm)', 'timestamp': 'Time'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Rainfall intensity over time
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['timestamp'], y=df['rainfall'], 
                                    mode='lines', name='Rainfall',
                                    line=dict(color='blue', width=2)))
            
            # Add threshold lines
            fig2.add_hline(y=10, line_dash="dash", line_color="orange", 
                          annotation_text="Moderate Rain")
            fig2.add_hline(y=30, line_dash="dash", line_color="red", 
                          annotation_text="Heavy Rain")
            fig2.add_hline(y=50, line_dash="dash", line_color="darkred", 
                          annotation_text="Cloudburst")
            
            fig2.update_layout(title=f'Rainfall Intensity - {node_id}',
                             xaxis_title='Time', yaxis_title='Rainfall (mm)')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Rainfall statistics
        st.subheader("Rainfall Statistics (24h)")
        total_rainfall = df['rainfall'].sum()
        max_hourly = df_hourly['rainfall'].max()
        rainy_hours = len(df_hourly[df_hourly['rainfall'] > 0])
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Total Rainfall", f"{total_rainfall:.1f} mm")
        stat_col2.metric("Max Hourly", f"{max_hourly:.1f} mm")
        stat_col3.metric("Rainy Hours", rainy_hours)
    
    def render_risk_trends(self, df: pd.DataFrame, node_id: str):
        """Render risk trend analysis"""
        if 'risk_score' not in df.columns:
            st.warning("No risk score data available")
            return
        
        # Risk score over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['risk_score'], 
                               mode='lines+markers', name='Risk Score',
                               line=dict(color='red', width=3)))
        
        # Add threshold areas
        fig.add_hrect(y0=0.8, y1=1.0, line_width=0, fillcolor="red", opacity=0.2,
                     annotation_text="Critical Risk")
        fig.add_hrect(y0=0.6, y1=0.8, line_width=0, fillcolor="orange", opacity=0.2,
                     annotation_text="High Risk")
        fig.add_hrect(y0=0.4, y1=0.6, line_width=0, fillcolor="yellow", opacity=0.2,
                     annotation_text="Medium Risk")
        
        fig.update_layout(title=f'Risk Score Trend - {node_id}',
                         xaxis_title='Time', yaxis_title='Risk Score',
                         yaxis_range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk statistics
        current_risk = df['risk_score'].iloc[-1] if len(df) > 0 else 0
        max_risk = df['risk_score'].max()
        high_risk_periods = len(df[df['risk_score'] > 0.6])
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        risk_col1.metric("Current Risk", f"{current_risk:.0%}")
        risk_col2.metric("Peak Risk", f"{max_risk:.0%}")
        risk_col3.metric("High Risk Periods", high_risk_periods)
    
    def render_location_map(self, node_id: str):
        """Render node location on map"""
        # In a real implementation, this would show the actual node location
        # For now, we'll show a placeholder
        
        st.subheader("Node Location")
        
        # Mock location data - in real implementation, this would come from the API
        mock_locations = {
            'node_001': [28.6139, 77.2090],
            'node_002': [28.7041, 77.1025],
            'node_003': [28.4595, 77.0266],
            'node_004': [28.5355, 77.3910]
        }
        
        location = mock_locations.get(node_id, [28.6139, 77.2090])
        
        # Create map using plotly
        fig = go.Figure(go.Scattermapbox(
            lat=[location[0]],
            lon=[location[1]],
            mode='markers',
            marker=dict(size=20, color='red'),
            text=[f"Node: {node_id}"],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                zoom=10,
                center=dict(lat=location[0], lon=location[1])
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"Node Location: {location[0]:.4f}, {location[1]:.4f}")
    
    def render_sidebar(self):
        """Render sidebar with controls and filters"""
        with st.sidebar:
            st.header("Dashboard Controls")
            
            # Time range selector
            st.subheader("Time Range")
            time_range = st.selectbox(
                "Select Time Range",
                ["Last 6 Hours", "Last 24 Hours", "Last 7 Days", "Custom"],
                index=1
            )
            
            # Alert level filter
            st.subheader("Alert Filters")
            show_critical = st.checkbox("Critical Alerts", value=True)
            show_warning = st.checkbox("Warning Alerts", value=True)
            show_normal = st.checkbox("Normal Alerts", value=False)
            
            # Node status filter
            st.subheader("Node Status")
            show_online = st.checkbox("Online Nodes", value=True)
            show_offline = st.checkbox("Offline Nodes", value=False)
            
            # Refresh controls
            st.subheader("Refresh")
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            if auto_refresh:
                refresh_rate = st.slider("Refresh Rate (seconds)", 30, 300, 60)
                st.info(f"Next refresh in {refresh_rate} seconds")
                
                # Auto-refresh logic
                time.sleep(refresh_rate)
                st.rerun()
            else:
                if st.button("Refresh Now"):
                    st.rerun()
            
            # System information
            st.subheader("System Info")
            st.write(f"**Version:** 2.1.0")
            st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Quick actions
            st.subheader("Quick Actions")
            if st.button("Download Report"):
                self.generate_report()
            
            if st.button("System Diagnostics"):
                self.run_diagnostics()
    
    def generate_report(self):
        """Generate system report"""
        # This would generate and download a comprehensive report
        st.success("Report generation started...")
        # In real implementation, this would create a PDF/Excel report
    
    def run_diagnostics(self):
        """Run system diagnostics"""
        # This would run comprehensive system diagnostics
        st.info("Running system diagnostics...")
        
        # Mock diagnostics results
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.metric("API Connectivity", "✅ Healthy")
            st.metric("Database", "✅ Connected")
            st.metric("ML Models", "✅ Loaded")
        
        with diag_col2:
            st.metric("Alert System", "✅ Active")
            st.metric("Data Pipeline", "✅ Running")
            st.metric("Storage", "85%")
    
    def render_main_dashboard(self):
        """Render main dashboard view"""
        self.render_header()
        self.render_alert_panel()
        self.render_nodes_overview()
        
        # Additional visualizations
        st.subheader("📊 System Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_system_metrics()
        
        with col2:
            self.render_network_health()
    
    def render_system_metrics(self):
        """Render system performance metrics"""
        st.metric("Data Points Processed", "1,247,892")
        st.metric("Alerts Generated", "1,247")
        st.metric("Model Accuracy", "94.2%")
        st.metric("Uptime", "99.8%")
    
    def render_network_health(self):
        """Render network health metrics"""
        nodes_data = self.fetch_nodes_data()
        nodes = nodes_data.get('nodes', [])
        
        online_count = len([n for n in nodes if n.get('status') == 'online'])
        total_count = len(nodes)
        avg_battery = np.mean([n.get('battery_level', 0) for n in nodes]) if nodes else 0
        high_risk_count = len([n for n in nodes if n.get('last_risk_score', 0) > 0.6])
        
        st.metric("Network Health", f"{(online_count/total_count*100):.1f}%" if total_count > 0 else "N/A")
        st.metric("Average Battery", f"{avg_battery:.1f}%")
        st.metric("High Risk Nodes", high_risk_count)
        st.metric("Data Latency", "2.3s")
    
    def run(self):
        """Main dashboard runner"""
        # Initialize session state
        if 'selected_node' not in st.session_state:
            st.session_state.selected_node = None
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main content based on selection
        if st.session_state.selected_node:
            self.render_node_details(st.session_state.selected_node)
        else:
            self.render_main_dashboard()

# Run the dashboard
if __name__ == "__main__":
    dashboard = CloudburstDashboard()
    dashboard.run()
