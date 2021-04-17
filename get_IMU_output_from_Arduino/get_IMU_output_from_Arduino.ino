#include <Arduino_LSM9DS1.h>

float gyroscope_sample_rate = 0.0f;

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU");
    while (1);
  }
  gyroscope_sample_rate = IMU.gyroscopeSampleRate();
}

void loop() {
  float x, y, z;
  if (IMU.gyroscopeAvailable()) {
      IMU.readGyroscope(x, y, z);

      Serial.print("x:");
      Serial.print(x);
      Serial.print(' ');
      Serial.print("y:");
      Serial.print(y);
      Serial.print(' ');
      Serial.print("z:");
      Serial.println(z);
  }
}
