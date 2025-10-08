#ifndef LORA_COMMUNICATION_H
#define LORA_COMMUNICATION_H

#include <LoRa.h>

class LoRaManager {
private:
    long frequency;
    int syncWord;
    bool loraInitialized;
    
public:
    LoRaManager(long freq = 915E6, int sync = 0x12);
    bool initialize();
    bool sendData(const uint8_t* data, size_t length);
    bool receiveData(uint8_t* buffer, size_t length);
    bool sendAlertPacket(float riskScore, const char* alertType);
    bool sendTelemetryPacket(float temp, float humidity, float pressure, float rain);
    int getSignalStrength();
    void setPowerLevel(int level);
    
private:
    void encodeData(const uint8_t* rawData, uint8_t* encodedData, size_t length);
    bool waitForAck(int timeout = 3000);
};

// External instance
extern LoRaManager loraManager;

// Initialize LoRa communication
bool initializeLoRa() {
    return loraManager.initialize();
}

// Send alert via LoRa
bool sendLoRaAlert(SensorData data, float riskScore) {
    return loraManager.sendAlertPacket(riskScore, "CLOUDBURST_RISK");
}

#endif
