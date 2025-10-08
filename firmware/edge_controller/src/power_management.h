#ifndef POWER_MANAGEMENT_H
#define POWER_MANAGEMENT_H

#include <driver/rtc_io.h>

class PowerManager {
private:
    float batteryVoltage;
    float solarVoltage;
    float windVoltage;
    int batteryPercentage;
    bool powerSaveMode;
    
public:
    PowerManager();
    void begin();
    void updatePowerStatus();
    void enterDeepSleep(int seconds);
    void setPowerSaveMode(bool enable);
    bool isBatteryLow();
    bool isCharging();
    float getBatteryVoltage();
    int getBatteryPercentage();
    void logPowerConsumption();
    
    // Power optimization methods
    void disablePeripherals();
    void enablePeripherals();
    void adjustCPUFrequency(int frequency);
    
private:
    float readBatteryVoltage();
    float readSolarInput();
    float readWindInput();
    void managePowerDistribution();
};

extern PowerManager powerManager;

void setupPowerManagement() {
    powerManager.begin();
}

#endif
