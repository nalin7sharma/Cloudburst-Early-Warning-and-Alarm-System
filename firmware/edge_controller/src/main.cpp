#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <SD.h>
#include <LoRa.h>
#include "sensor_reading.h"
#include "lora_communication.h"
#include "power_management.h"
#include "anomaly_detection.h"

// Sensor objects
BME280 bme;
AS3935 lightning;
RainGauge rain;

// LoRa parameters
#define LORA_FREQUENCY 915E6
#define LORA_SYNC_WORD 0x12

// File system
File dataFile;

// Data structure for sensor readings
struct SensorData {
  float temperature;
  float humidity;
  float pressure;
  float rainfall;
  int lightning_count;
  uint32_t timestamp;
};

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  Serial.println("Initializing Cloudburst Edge Node...");
  
  // Initialize power management
  setupPowerManagement();
  
  // Initialize sensors
  if (!initializeSensors()) {
    Serial.println("Sensor initialization failed!");
    sleep(10);
    ESP.restart();
  }
  
  // Initialize LoRa
  if (!initializeLoRa()) {
    Serial.println("LoRa initialization failed!");
  }
  
  // Initialize SD card
  if (!SD.begin(5)) {
    Serial.println("SD card initialization failed!");
  }
  
  Serial.println("Edge Node Ready!");
}

void loop() {
  // Read all sensors
  SensorData data = readSensorData();
  data.timestamp = millis();
  
  // Perform edge inference for anomaly detection
  bool anomaly_detected = detectAnomaly(data);
  float risk_score = calculateRiskScore(data);
  
  // Log data to SD card
  logDataToSD(data, anomaly_detected, risk_score);
  
  if (anomaly_detected || risk_score > 0.7) {
    Serial.println("Anomaly detected! Sending alert...");
    
    // Send immediate alert via LoRa
    sendLoRaAlert(data, risk_score);
    
    // If high risk, also try backup radio
    if (risk_score > 0.8) {
      activateBackupRadio();
      sendBackupAlert(data, risk_score);
    }
  } else {
    // Send regular telemetry data
    sendLoRaTelemetry(data);
  }
  
  // Check for OTA updates
  checkForUpdates();
  
  // Enter deep sleep for 5 minutes (300 seconds)
  Serial.println("Entering deep sleep for 5 minutes...");
  ESP.deepSleep(5 * 60 * 1000000);
}

SensorData readSensorData() {
  SensorData data;
  
  data.temperature = bme.readTemperature();
  data.humidity = bme.readHumidity();
  data.pressure = bme.readPressure() / 100.0;
  data.rainfall = rain.getHourlyRainfall();
  data.lightning_count = lightning.getStrikeCount();
  
  return data;
}

void logDataToSD(SensorData data, bool anomaly, float risk) {
  dataFile = SD.open("/datalog.txt", FILE_WRITE);
  if (dataFile) {
    dataFile.print(data.timestamp);
    dataFile.print(",");
    dataFile.print(data.temperature);
    dataFile.print(",");
    dataFile.print(data.humidity);
    dataFile.print(",");
    dataFile.print(data.pressure);
    dataFile.print(",");
    dataFile.print(data.rainfall);
    dataFile.print(",");
    dataFile.print(data.lightning_count);
    dataFile.print(",");
    dataFile.print(anomaly);
    dataFile.print(",");
    dataFile.println(risk);
    dataFile.close();
  }
}
