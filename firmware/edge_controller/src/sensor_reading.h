#ifndef SENSOR_READING_H
#define SENSOR_READING_H

#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <AS3935.h>

class SensorManager {
private:
    Adafruit_BME280 bme;
    AS3935 lightningSensor;
    
    // Rain gauge parameters
    const int RAIN_GAUGE_PIN = 17;
    volatile unsigned long tipCount;
    float rainfallMM;
    
public:
    SensorManager();
    bool initialize();
    
    // Sensor reading methods
    float readTemperature();
    float readHumidity();
    float readPressure();
    float readRainfall();
    int readLightningStrikes();
    float calculateDewPoint();
    float calculateHumidityIndex();
    
    // Interrupt service routine for rain gauge
    static void rainGaugeISR();
    
private:
    void resetRainfall();
    void calibrateSensors();
};

// External instance
extern SensorManager sensorManager;

// Initialize all sensors
bool initializeSensors() {
    return sensorManager.initialize();
}

#endif
