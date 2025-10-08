# Edge Controller Firmware

## Overview
Firmware for the edge nodes deployed in mountainous regions. Handles sensor data collection, local anomaly detection, and communication via LoRa.

## Features
- Multi-sensor data acquisition
- Real-time anomaly detection using TensorFlow Lite
- LoRa communication with retry mechanism
- Power management with deep sleep
- OTA firmware updates
- Local data logging to SD card

## Dependencies
- PlatformIO
- Arduino Framework
- TensorFlow Lite for Microcontrollers
- LoRa Library
- BME280 Library
- AS3935 Library

## Building and Flashing

### Using PlatformIO
```bash
cd firmware/edge_controller
pio run
pio run -t upload
```
## Using Arduino IDE
Install required libraries
Open src/main.cpp
Select ESP32 Dev Module board
Compile and upload

### Configuration
Edit config.h for:
LoRa frequency
Sensor calibration
Deep sleep intervals
Threshold values

### Debugging
Serial monitor at 115200 baud
SD card log files
LED status indicators
