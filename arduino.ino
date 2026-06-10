#include <Servo.h>

// Simple serial-controlled servos (4x)
// Commands (newline-terminated):
//  - "<angle>" e.g. "90" -> move SERVO 0 to 90 deg (backward compatible)
//  - "<id>:<angle>" e.g. "2:45" -> move SERVO 2 to 45 deg
//  - "<id>:<angle>,<id>:<angle>,..." e.g. "0:90,1:45,2:120,3:0" -> set multiple
//  - "D" -> detach all servos (stop holding)
//  - "D:<id>" -> detach servo <id>

// Adjust these pins to match your wiring.
// Common Arduino Uno PWM pins: 3, 5, 6, 9, 10, 11
const int SERVO_COUNT = 4;
const int SERVO_PINS[SERVO_COUNT] = {9, 10, 11, 12};

const int MIN_ANGLE = 0;
const int MAX_ANGLE = 180;

Servo servos[SERVO_COUNT];
const bool DEBUG = false;

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < SERVO_COUNT; i++) {
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(90);
  }
}

static inline int clampAngle(int angle) {
  if (angle < MIN_ANGLE) angle = MIN_ANGLE;
  if (angle > MAX_ANGLE) angle = MAX_ANGLE;
  return angle;
}

static void detachServo(int id) {
  if (id < 0 || id >= SERVO_COUNT) return;
  if (servos[id].attached()) servos[id].detach();
  if (DEBUG) {
    Serial.print("DETACHED ");
    Serial.println(id);
  }
}

static void setServoAngle(int id, int angle) {
  if (id < 0 || id >= SERVO_COUNT) return;
  angle = clampAngle(angle);
  if (!servos[id].attached()) {
    servos[id].attach(SERVO_PINS[id]);
  }
  servos[id].write(angle);
  if (DEBUG) {
    Serial.print("OK ");
    Serial.print(id);
    Serial.print(" ");
    Serial.println(angle);
  }
}

static void detachAll() {
  for (int i = 0; i < SERVO_COUNT; i++) {
    if (servos[i].attached()) servos[i].detach();
  }
  if (DEBUG) {
    Serial.println("DETACHED ALL");
  }
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) {
      return;
    }

    if (line == "D" || line == "d") {
      detachAll();
      return;
    }

    // "D:<id>" -> detach one
    if (line.length() >= 3 && (line[0] == 'D' || line[0] == 'd') && line[1] == ':') {
      int id = line.substring(2).toInt();
      detachServo(id);
      return;
    }

    // Backward-compatible: plain number -> servo 0 angle
    bool hasColon = (line.indexOf(':') >= 0);
    bool hasComma = (line.indexOf(',') >= 0);
    if (!hasColon && !hasComma) {
      int angle = line.toInt();
      setServoAngle(0, angle);
      return;
    }

    // Parse "<id>:<angle>,<id>:<angle>,..."
    int start = 0;
    while (start < (int)line.length()) {
      int end = line.indexOf(',', start);
      if (end < 0) end = line.length();

      String token = line.substring(start, end);
      token.trim();
      if (token.length() > 0) {
        int colon = token.indexOf(':');
        if (colon >= 0) {
          int id = token.substring(0, colon).toInt();
          int angle = token.substring(colon + 1).toInt();
          setServoAngle(id, angle);
        }
      }

      start = end + 1;
    }
  }
}
