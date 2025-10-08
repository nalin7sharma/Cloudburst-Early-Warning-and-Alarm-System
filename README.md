# Cloudburst Early Warning and Alarm System

## Overview
A distributed IoT system with AI-powered anomaly detection for real-time cloudburst prediction in hilly regions. Provides early warnings to communities and authorities through multiple communication channels.

## 🚀 Key Features
- **Multi-sensor Edge Nodes** (rain, humidity, pressure, lightning, temperature)
- **Redundant Communication** (LoRa + HF/VHF backup)
- **AI/ML Anomaly Detection** using ConvLSTM models
- **Hybrid Power Management** (Solar + Wind + LiFePO₄ batteries)
- **Real-time Alerts** via SMS, mobile app, and local sirens
- **Off-grid Operation** for remote mountainous regions

## 🛠 Technology Stack
**Hardware:** Raspberry Pi, LoRa modules, Sensor array, Solar panels, Wind turbines  
**Software:** Python, TensorFlow, PyTorch, React, Streamlit, InfluxDB, MQTT

## 📁 Project Structure
Cloudburst-Early-Warning-System/
├── hardware/ # Circuit designs, 3D models, BOM
├── firmware/ # Edge controller and gateway code
├── software/ # Cloud backend, dashboard, alerts
├── docs/ # Documentation and presentations
├── tests/ # Unit and integration tests
└── data/ # Sample datasets and configs


## 🏃‍♂️ Quick Start
1. Clone the repository: `git clone https://github.com/nalin7sharma/Cloudburst-Early-Warning-System.git`
2. Check hardware requirements in `/hardware/bom/components_list.csv`
3. Follow installation guide: `docs/installation_guide.md`
4. Deploy using: `deployment_guide.md`
