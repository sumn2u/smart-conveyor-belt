import argparse
import time

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "pyserial not installed. Install with: python -m pip install pyserial"
    ) from exc

# Defaults (mirrors your other servo script style)
ANGLE_A = [28, 26, 19, 0]
ANGLE_B = [67, 63, 60, 36]
MOVE_STEP_DEG = 1
STEP_DELAY_SEC = 0.008
SETTLE_SEC = 0.20
DEFAULT_SERVOS = 4
STARTUP_HOLD_SEC = 2.0


def clamp(value, low, high):
    return max(low, min(high, value))


def _normalize_angles(angles, servos):
    if isinstance(angles, (int, float)):
        angles = [angles]

    angles = [int(clamp(a, 0, 180)) for a in angles]

    if len(angles) == 1 and servos > 1:
        return angles * servos
    if len(angles) != servos:
        raise ValueError(f"Expected {servos} angles, got {len(angles)}: {angles}")
    return angles


def send_angles(ser, angles, *, servos=1, servo_ids=None):
    """
    Send "<id0>:<a0>,<id1>:<a1>,...\\n" (the "map" protocol).
    This matches the updated arduino_servo.ino in this repo.
    """
    angles = _normalize_angles(angles, servos)
    if servo_ids is None:
        servo_ids = list(range(servos))
    if len(servo_ids) != servos:
        raise ValueError(f"Expected {servos} servo_ids, got {len(servo_ids)}")
    payload = ",".join(f"{sid}:{a}" for sid, a in zip(servo_ids, angles))
    ser.write(f"{payload}\n".encode("ascii"))


def smooth_move(ser, start_angle, target_angle, *, servos=1, servo_ids=None):
    start = _normalize_angles(start_angle, servos)
    target = _normalize_angles(target_angle, servos)

    if start == target:
        return target

    current = start[:]
    max_delta = max(abs(t - s) for s, t in zip(start, target))
    steps = max(1, (max_delta + (MOVE_STEP_DEG - 1)) // MOVE_STEP_DEG)

    for i in range(1, steps + 1):
        for idx in range(servos):
            s = start[idx]
            t = target[idx]
            current[idx] = int(round(s + (t - s) * (i / steps)))
        send_angles(ser, current, servos=servos, servo_ids=servo_ids)
        time.sleep(STEP_DELAY_SEC)

    send_angles(ser, target, servos=servos, servo_ids=servo_ids)
    time.sleep(SETTLE_SEC)
    return target


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive per-servo toggle controller.\n"
            "Type a servo ID number (e.g. 1) then Enter to toggle that servo between "
            "--angle-a and --angle-b.\n"
            "Type q then Enter to quit."
        )
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument(
        "--servos",
        type=int,
        default=DEFAULT_SERVOS,
        help="Number of servos in the angles list (default: 4).",
    )
    parser.add_argument(
        "--servo-ids",
        type=int,
        nargs="+",
        default=list(range(DEFAULT_SERVOS)),
        help="Servo IDs to address (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--angle-a",
        type=int,
        nargs="+",
        default=ANGLE_A,
        help="Angle A per servo (1 value replicated, or N values).",
    )
    parser.add_argument(
        "--angle-b",
        type=int,
        nargs="+",
        default=ANGLE_B,
        help="Angle B per servo (1 value replicated, or N values).",
    )
    parser.add_argument(
        "--startup-hold",
        type=float,
        default=STARTUP_HOLD_SEC,
        help="Seconds to hold Angle A before accepting toggle input.",
    )
    args = parser.parse_args()

    servos = int(args.servos)
    servo_ids = args.servo_ids
    if len(servo_ids) == 1 and servos > 1:
        servo_ids = servo_ids * servos
    if len(servo_ids) != servos:
        raise SystemExit(
            f"--servo-ids must have exactly {servos} values (got {len(servo_ids)})."
        )

    a_angles = _normalize_angles(args.angle_a, servos)
    b_angles = _normalize_angles(args.angle_b, servos)
    current = a_angles[:]

    # Track whether each servo is currently at B (True) or A (False).
    at_b = [False] * servos

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        # Allow Arduino to reboot on serial open
        time.sleep(2.0)

        # Force the default/reset angles even when current already equals Angle A.
        send_angles(ser, a_angles, servos=servos, servo_ids=servo_ids)
        time.sleep(args.startup_hold)
        current = a_angles[:]

        print(f"Ready. Servo IDs: {servo_ids}")
        print("Type 0/1/2/3 (or any configured servo ID) then Enter to toggle that servo.")
        print("Type q then Enter to quit.")

        while True:
            try:
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break

            try:
                servo_id = int(line)
            except ValueError:
                print("Enter a servo ID number (e.g. 1), or q to quit.")
                continue

            if servo_id not in servo_ids:
                print(f"Servo ID {servo_id} not in configured --servo-ids {servo_ids}.")
                continue

            idx = servo_ids.index(servo_id)
            target_angle = b_angles[idx] if not at_b[idx] else a_angles[idx]
            target = current[:]
            target[idx] = target_angle

            current = smooth_move(ser, current, target, servos=servos, servo_ids=servo_ids)
            at_b[idx] = not at_b[idx]

    print("Done")


if __name__ == "__main__":
    main()
