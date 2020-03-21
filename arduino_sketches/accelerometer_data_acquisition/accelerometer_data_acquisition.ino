#include <Arduino_LSM9DS1.h>

int sample_every_n;
int sample_skip_counter = 1;

void setup() {
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("ERROR: Failed initializing IMU");
    while (1);
  }

  float sample_rate = IMU.accelerationSampleRate();
  sample_every_n = static_cast<int>(roundf(sample_rate / 25));
}

void loop() {
  while (IMU.accelerationAvailable()) {
    float x, y, z;
    if (!IMU.readAcceleration(x, y, z)) {
      break;
    }

    if (sample_skip_counter != sample_every_n) {
      sample_skip_counter += 1;
      continue;
    }

    Serial.printf("%f,%f,%f\r\n", x, y, z);
    sample_skip_counter = 1;
  }
}
