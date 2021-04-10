#include <Arduino_LSM9DS1.h>

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU");
    while (1);
  }
}

void loop() {
  float x, y, z;
  if (IMU.magneticFieldAvailable()) {
      // get only magnetometer data
      IMU.readMagneticField(x, y, z);
  
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
