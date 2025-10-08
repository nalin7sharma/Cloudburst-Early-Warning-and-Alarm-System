# Edge Node Assembly Guide

## Required Components
- Raspberry Pi 4/CM4
- LoRa RA-02 Module (915MHz)
- BME280 Sensor (Temperature/Humidity/Pressure)
- Tipping Bucket Rain Gauge
- AS3935 Lightning Detector
- 50W Solar Panel + 20W Micro Wind Turbine
- 20Ah LiFePO₄ Battery + Charge Controller
- IP67 Weatherproof Enclosure
- SD Card (32GB minimum)

## Assembly Steps

### Step 1: Sensor Mounting
1. Mount BME280 sensor on protected area of enclosure
2. Install rain gauge with clear exposure to rainfall
3. Position lightning detector away from metal interference
4. Ensure all sensors have weatherproof connections

### Step 2: Power Management
1. Connect solar panel to charge controller
2. Wire wind turbine to charge controller
3. Connect LiFePO₄ battery to charge controller
4. Wire Raspberry Pi power from battery output

### Step 3: Main Board Assembly
1. Install Raspberry Pi in enclosure
2. Connect LoRa module via SPI interface
3. Wire all sensors to appropriate GPIO pins
4. Connect SD card with pre-loaded OS

### Step 4: Weatherproofing
1. Apply silicone sealant to all cable entries
2. Use waterproof connectors for external sensors
3. Install desiccant pack inside enclosure
4. Test enclosure with water spray

### Step 5: Initial Testing
1. Power on system and check LED indicators
2. Verify sensor readings via serial monitor
3. Test LoRa transmission range
4. Validate power consumption in deep sleep

## Wiring Diagram
Solar Panel → Charge Controller → Battery → Raspberry Pi
Wind Turbine → Charge Controller ↗

Sensors:
BME280 → I2C (SDA/SCL)
Rain Gauge → GPIO 17
Lightning → GPIO 27
LoRa → SPI (MISO/MOSI/SCK/CS)


## Calibration Procedure
1. Rain Gauge: Measure known water volume
2. BME280: Compare with reference hygrometer
3. Lightning: Test with ESD simulator
4. Power: Verify charging efficiency
