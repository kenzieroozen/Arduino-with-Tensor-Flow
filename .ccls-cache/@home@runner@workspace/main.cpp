#include <Arduino.h>
#include <SPI.h>

// Pin ADS1256 ke ESP32
#define CS_PIN   5   // Chip Select
#define DRDY_PIN 4   // Data Ready

#define SCK  18
#define MISO 19
#define MOSI 23

// ===== Fungsi kirim command ke ADS1256 =====
void ADS1256_SendCommand(byte cmd) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(cmd);
  digitalWrite(CS_PIN, HIGH);
}

// ===== Baca 24-bit data dari ADS1256 =====
long ADS1256_ReadData() {
  long value = 0;
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x01); // RDATA command
  delayMicroseconds(10);

  value = ((long)SPI.transfer(0x00) << 16) |
          ((long)SPI.transfer(0x00) << 8) |
          (long)SPI.transfer(0x00);

  digitalWrite(CS_PIN, HIGH);

  if (value & 0x800000) value |= 0xFF000000; // signed 24-bit
  return value;
}

// ===== Setup =====
void setup() {
  Serial.begin(115200);
  pinMode(CS_PIN, OUTPUT);
  pinMode(DRDY_PIN, INPUT);
  SPI.begin(SCK, MISO, MOSI, CS_PIN);
  digitalWrite(CS_PIN, HIGH);

  Serial.println("ESP32 + ADS1256 Biosensor Classifier started...");
}

// ===== Fungsi klasifikasi =====
String classifyPollutant(float voltage) {
  if (voltage >= 0.45 && voltage <= 0.55) return "Copper (Cu)";
  else if (voltage >= 0.75 && voltage <= 0.85) return "Lead (Pb)";
  else if (voltage >= 1.10 && voltage <= 1.20) return "Nickel (Ni)";
  else if (voltage >= 1.40 && voltage <= 1.50) return "PFAS";
  else return "Unknown / Noise";
}

// ===== Loop utama =====
void loop() {
  if (digitalRead(DRDY_PIN) == LOW) {
    long rawValue = ADS1256_ReadData();
    float voltage = (rawValue * 2.5) / 0x7FFFFF; // contoh: Vref=2.5V

    String pollutant = classifyPollutant(voltage);

    Serial.print("Voltage: ");
    Serial.print(voltage, 6);
    Serial.print(" V --> Detected: ");
    Serial.println(pollutant);

    delay(200); // sampling interval
  }
}
