#ifndef ANOMALY_DETECTION_H
#define ANOMALY_DETECTION_H

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

class AnomalyDetector {
private:
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input;
    TfLiteTensor* output;
    
    // Model parameters
    const tflite::Model* model;
    uint8_t tensor_arena[2048];
    
    // Threshold values
    const float RAIN_THRESHOLD = 50.0; // mm/hr
    const float HUMIDITY_THRESHOLD = 95.0;
    const float PRESSURE_DROP_THRESHOLD = 5.0; // hPa/hr
    
public:
    AnomalyDetector();
    bool initialize();
    bool detect(const float* sensorData, int dataLength);
    float calculateRiskScore(const float* sensorData, int dataLength);
    bool checkThresholdViolations(float temp, float humidity, float pressure, float rain);
    
private:
    void preprocessData(const float* rawData, float* processedData);
    bool loadModel();
};

extern AnomalyDetector anomalyDetector;

bool detectAnomaly(SensorData data) {
    float sensorArray[] = {data.temperature, data.humidity, data.pressure, data.rainfall};
    return anomalyDetector.detect(sensorArray, 4);
}

float calculateRiskScore(SensorData data) {
    float sensorArray[] = {data.temperature, data.humidity, data.pressure, data.rainfall};
    return anomalyDetector.calculateRiskScore(sensorArray, 4);
}

#endif
