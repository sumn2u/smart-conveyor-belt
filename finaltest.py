import argparse
import threading
import time

import cv2
import lgpio
from picamera2 import Picamera2
from ultralytics import YOLO

try:
    import serial
except ImportError as exc:
    raise SystemExit(
        "pyserial not installed. Install with: python -m pip install pyserial"
    ) from exc

from arduino_servo_serial_toggle import (
    ANGLE_A,
    ANGLE_B,
    DEFAULT_SERVOS,
    _normalize_angles,
    send_angles,
    smooth_move,
)
from ultrasonic_distance import (
    ECHO_GPIO,
    MAX_DISTANCE_CM,
    READ_INTERVAL_SEC,
    SPEED_OF_SOUND_CM_PER_SEC,
    TRIG_GPIO,
    measure_distance_cm,
)


ARDUINO_PORT = "/dev/ttyACM0"
ARDUINO_BAUD = 115200
MODEL_PATH = "conveyorset.pt"
CONFIDENCE = 0.5
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 640
DETECTION_LINE_ORIENTATION = "horizontal"
DETECTION_LINE_POSITION = 140

DISTANCE_THRESHOLD_CM = 15.8
SECONDS_PER_CLOSE_FRAME = 0.5
GATE_OPEN_DURATION_MULTIPLIER = 1.3
REQUIRED_DETECTION_FRAMES = 3
REQUIRED_CLEAR_FRAMES = 6
REQUIRED_CAMERA_FRAMES = 1
STARTUP_HOLD_SEC = 2.0

PLASTIC_SERVO_ID = 0
GLASS_SERVO_ID = 1
METAL_SERVO_ID = 2
PAPER_SERVO_ID = 3
PLASTIC_OPEN_DELAY_SEC = 0.0
PLASTIC_EXTRA_OPEN_SEC = 0.4
METAL_OPEN_DELAY_SEC = 20.4
METAL_EXTRA_OPEN_SEC = 3.5
GLASS_OPEN_DELAY_SEC = 11.0
GLASS_EXTRA_OPEN_SEC = 4.5
PAPER_OPEN_DELAY_SEC = 30.9
PAPER_EXTRA_OPEN_SEC = 3.3

GLASS_CLASS_NAMES = {"glass", "glass bottle", "bottle glass"}
METAL_CLASS_NAMES = {"metal", "can", "aluminium", "aluminum", "tin", "metal can"}
PLASTIC_CLASS_NAMES = {"plastic", "plastic bottle", "bottle plastic"}
PAPER_CLASS_NAMES = {"paper", "cardboard", "carton", "paperboard"}


def is_close_detection(distance_cm, threshold_cm):
    if distance_cm is None:
        return None
    return distance_cm < threshold_cm


def read_and_print_distance(handle, timeout_sec):
    distance_cm = measure_distance_cm(handle, timeout_sec)
    if distance_cm is None:
        print("Out of range / timeout")
    else:
        print(f"Distance: {distance_cm:.1f} cm")
    return distance_cm


def normalize_class_name(name):
    return name.strip().lower().replace("_", " ").replace("-", " ")


NORMALIZED_METAL_CLASS_NAMES = {
    normalize_class_name(class_name) for class_name in METAL_CLASS_NAMES
}
NORMALIZED_PLASTIC_CLASS_NAMES = {
    normalize_class_name(class_name) for class_name in PLASTIC_CLASS_NAMES
}
NORMALIZED_PAPER_CLASS_NAMES = {
    normalize_class_name(class_name) for class_name in PAPER_CLASS_NAMES
}
NORMALIZED_GLASS_CLASS_NAMES = {
    normalize_class_name(class_name) for class_name in GLASS_CLASS_NAMES
}


def clamp(value, low, high):
    return max(low, min(high, value))


def box_touches_line(box, line_orientation, line_position):
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
    if line_orientation == "horizontal":
        return y1 <= line_position <= y2
    return x1 <= line_position <= x2


def draw_detection_line(image, line_orientation, line_position):
    height, width = image.shape[:2]
    if line_orientation == "horizontal":
        line_position = int(clamp(line_position, 0, height - 1))
        start = (0, line_position)
        end = (width - 1, line_position)
        label_position = (8, max(20, line_position - 8))
        label = f"line y={line_position}"
    else:
        line_position = int(clamp(line_position, 0, width - 1))
        start = (line_position, 0)
        end = (line_position, height - 1)
        label_position = (min(width - 115, line_position + 8), 24)
        label = f"line x={line_position}"

    cv2.line(image, start, end, (0, 255, 255), 2)
    cv2.putText(
        image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def camera_detects_materials(picam2, model, conf, line_orientation, line_position):
    frame = picam2.capture_array()
    frame = frame[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame, imgsz=640, conf=conf)
    materials = set()

    for box in results[0].boxes:
        if not box_touches_line(box, line_orientation, line_position):
            continue

        class_id = int(box.cls[0])
        class_name = normalize_class_name(model.names[class_id])
        if class_name in NORMALIZED_GLASS_CLASS_NAMES:
            materials.add("glass")
        if class_name in NORMALIZED_METAL_CLASS_NAMES:
            materials.add("metal")
        if class_name in NORMALIZED_PLASTIC_CLASS_NAMES:
            materials.add("plastic")
        if class_name in NORMALIZED_PAPER_CLASS_NAMES:
            materials.add("paper")

    annotated = results[0].plot()
    draw_detection_line(annotated, line_orientation, line_position)
    cv2.imshow("Final Test Camera", annotated)
    return materials


def move_one_servo(ser, current_angles, servo_idx, target_angle, servos, servo_ids):
    target_angles = current_angles[:]
    target_angles[servo_idx] = target_angle
    if target_angles == current_angles:
        return current_angles
    return smooth_move(
        ser,
        current_angles,
        target_angles,
        servos=servos,
        servo_ids=servo_ids,
    )


def best_material(material_frame_counts, confirmed_materials):
    seen = [
        (material_frame_counts[material], material)
        for material in confirmed_materials
        if material_frame_counts[material] > 0
    ]
    if not seen:
        return None

    top_count = max(count for count, _ in seen)
    winners = [material for count, material in seen if count == top_count]
    if len(winners) != 1:
        return None
    return winners[0]


def queue_summary(pending_open_events, material_actions):
    counts = {material: 0 for material in material_actions}
    for event in pending_open_events:
        counts[event["material"]] += 1
    return ", ".join(
        f"{material}={counts[material]}" for material in material_actions
    )


def scaled_gate_close_time(open_time, first_close_time, last_close_time, extra_open_sec):
    object_duration = max(0.0, last_close_time - first_close_time)
    return open_time + (object_duration * GATE_OPEN_DURATION_MULTIPLIER) + extra_open_sec


def ultrasonic_worker(
    handle,
    timeout_sec,
    threshold_cm,
    required_frames,
    clear_frames_required,
    state,
    state_lock,
    stop_event,
):
    close_frames = 0
    clear_frames = 0
    confirmed_object = False
    confirmation_time = None
    first_close_time = None
    last_close_time = None
    active_object_id = None
    next_object_id = 1
    last_detected = False

    while not stop_event.is_set():
        now = time.monotonic()
        distance_cm = read_and_print_distance(handle, timeout_sec)
        detected = is_close_detection(distance_cm, threshold_cm)
        if detected is None:
            detected = last_detected
            print(
                "Timeout repeated previous ultrasonic state: "
                f"{'close' if detected else 'clear'}"
            )
        else:
            last_detected = detected
        ended_event = None

        if detected:
            if close_frames == 0 and active_object_id is None:
                active_object_id = next_object_id
                next_object_id += 1
                first_close_time = now

            close_frames += 1
            clear_frames = 0
            last_close_time = now

            if close_frames >= required_frames and not confirmed_object:
                confirmed_object = True
                confirmation_time = now
                print(
                    f"Object {active_object_id} confirmed at "
                    f"{close_frames} ultrasonic close frames. Delay timer started."
                )

            print(
                f"Ultrasonic close frames: {close_frames}"
                f"{' (confirmed)' if confirmed_object else ''}"
            )
        else:
            if confirmed_object:
                clear_frames += 1
                print(f"Ultrasonic clear frames: {clear_frames}/{clear_frames_required}")

                if clear_frames >= clear_frames_required:
                    ended_event = {
                        "object_id": active_object_id,
                        "close_frames": close_frames,
                        "confirmation_time": confirmation_time,
                        "first_close_time": first_close_time,
                        "last_close_time": last_close_time,
                    }
                    close_frames = 0
                    clear_frames = 0
                    confirmed_object = False
                    confirmation_time = None
                    first_close_time = None
                    last_close_time = None
                    active_object_id = None
            else:
                close_frames = 0
                clear_frames = 0
                first_close_time = None
                last_close_time = None
                active_object_id = None

        with state_lock:
            state["latest_distance_cm"] = distance_cm
            state["distance_detected"] = detected
            state["close_frames"] = close_frames
            state["clear_frames"] = clear_frames
            state["confirmed_object"] = confirmed_object
            state["confirmation_time"] = confirmation_time
            state["first_close_time"] = first_close_time
            state["last_close_time"] = last_close_time
            state["active_object_id"] = active_object_id
            if ended_event is not None:
                state["ended_events"].append(ended_event)

        stop_event.wait(READ_INTERVAL_SEC)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use camera material detection and ultrasonic frame counting to open "
            "the matching servo after the configured delay."
        )
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--port", default=ARDUINO_PORT)
    parser.add_argument("--baud", type=int, default=ARDUINO_BAUD)
    parser.add_argument("--threshold-cm", type=float, default=DISTANCE_THRESHOLD_CM)
    parser.add_argument("--seconds-per-frame", type=float, default=SECONDS_PER_CLOSE_FRAME)
    parser.add_argument(
        "--glass-open-delay",
        type=float,
        default=GLASS_OPEN_DELAY_SEC,
        help="Seconds to wait before opening the glass servo.",
    )
    parser.add_argument(
        "--metal-open-delay",
        type=float,
        default=METAL_OPEN_DELAY_SEC,
        help="Seconds to wait before opening the metal servo.",
    )
    parser.add_argument("--plastic-open-delay", type=float, default=PLASTIC_OPEN_DELAY_SEC)
    parser.add_argument("--paper-open-delay", type=float, default=PAPER_OPEN_DELAY_SEC)
    parser.add_argument(
        "--glass-extra-open-sec",
        type=float,
        default=GLASS_EXTRA_OPEN_SEC,
        help="Extra seconds to keep the glass servo open after the scaled object duration.",
    )
    parser.add_argument(
        "--metal-extra-open-sec",
        type=float,
        default=METAL_EXTRA_OPEN_SEC,
        help="Extra seconds to keep the metal servo open after the scaled object duration.",
    )
    parser.add_argument("--startup-hold", type=float, default=STARTUP_HOLD_SEC)
    parser.add_argument("--required-frames", type=int, default=REQUIRED_DETECTION_FRAMES)
    parser.add_argument("--clear-frames", type=int, default=REQUIRED_CLEAR_FRAMES)
    parser.add_argument("--camera-frames", type=int, default=REQUIRED_CAMERA_FRAMES)
    parser.add_argument("--glass-servo-id", type=int, default=GLASS_SERVO_ID)
    parser.add_argument("--metal-servo-id", type=int, default=METAL_SERVO_ID)
    parser.add_argument("--plastic-servo-id", type=int, default=PLASTIC_SERVO_ID)
    parser.add_argument("--paper-servo-id", type=int, default=PAPER_SERVO_ID)
    parser.add_argument("--servos", type=int, default=DEFAULT_SERVOS)
    parser.add_argument(
        "--servo-ids",
        type=int,
        nargs="+",
        default=list(range(DEFAULT_SERVOS)),
        help="Servo IDs to address (default: 0 1 2 3).",
    )
    parser.add_argument("--angle-a", type=int, nargs="+", default=ANGLE_A)
    parser.add_argument("--angle-b", type=int, nargs="+", default=ANGLE_B)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument(
        "--line-orientation",
        choices=("horizontal", "vertical"),
        default=DETECTION_LINE_ORIENTATION,
        help="Only count boxes touching this horizontal or vertical line.",
    )
    parser.add_argument(
        "--line-position",
        type=int,
        default=DETECTION_LINE_POSITION,
        help="Line y-position for horizontal, or x-position for vertical.",
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

    for servo_id_arg, servo_id in (
        ("--plastic-servo-id", args.plastic_servo_id),
        ("--metal-servo-id", args.metal_servo_id),
        ("--glass-servo-id", args.glass_servo_id),
        ("--paper-servo-id", args.paper_servo_id),
    ):
        if servo_id not in servo_ids:
            raise SystemExit(f"{servo_id_arg} {servo_id} is not in --servo-ids {servo_ids}.")

    closed_angles = _normalize_angles(args.angle_a, servos)
    open_angles = _normalize_angles(args.angle_b, servos)
    material_actions = {
        "plastic": {
            "servo_id": args.plastic_servo_id,
            "servo_idx": servo_ids.index(args.plastic_servo_id),
            "open_delay": args.plastic_open_delay,
            "extra_open_sec": PLASTIC_EXTRA_OPEN_SEC,
        },
        "metal": {
            "servo_id": args.metal_servo_id,
            "servo_idx": servo_ids.index(args.metal_servo_id),
            "open_delay": args.metal_open_delay,
            "extra_open_sec": args.metal_extra_open_sec,
        },
        "glass": {
            "servo_id": args.glass_servo_id,
            "servo_idx": servo_ids.index(args.glass_servo_id),
            "open_delay": args.glass_open_delay,
            "extra_open_sec": args.glass_extra_open_sec,
        },
        "paper": {
            "servo_id": args.paper_servo_id,
            "servo_idx": servo_ids.index(args.paper_servo_id),
            "open_delay": args.paper_open_delay,
            "extra_open_sec": PAPER_EXTRA_OPEN_SEC,
        },
    }

    max_round_trip_sec = (MAX_DISTANCE_CM * 2.0) / SPEED_OF_SOUND_CM_PER_SEC
    timeout_sec = max_round_trip_sec * 1.5

    handle = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(handle, TRIG_GPIO, 0)
    lgpio.gpio_claim_input(handle, ECHO_GPIO)

    model = YOLO(args.model)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.camera_width, args.camera_height)}
    )
    picam2.configure(config)
    picam2.start()

    current_angles = closed_angles[:]

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        time.sleep(2.0)
        send_angles(ser, closed_angles, servos=servos, servo_ids=servo_ids)
        time.sleep(args.startup_hold)

        print(
            f"Final test running. Ultrasonic confirms after {args.required_frames} "
            f"frames < {args.threshold_cm:.1f} cm and ends after {args.clear_frames} "
            f"clear frames. Camera confirms material after {args.camera_frames} "
            "separate frame(s)."
        )
        print(
            f"Plastic -> servo {args.plastic_servo_id} after "
            f"{args.plastic_open_delay:.1f}s; "
            f"metal -> servo {args.metal_servo_id} after {args.metal_open_delay:.1f}s; "
            f"glass -> servo {args.glass_servo_id} after "
            f"{args.glass_open_delay:.1f}s; paper/cardboard -> servo "
            f"{args.paper_servo_id} after {args.paper_open_delay:.1f}s."
        )
        print(
            "Each servo opens after the material delay from the first confirmed "
            "camera+ultrasonic detection, then stays open for "
            f"{GATE_OPEN_DURATION_MULTIPLIER:.1f}x the ultrasonic object duration. "
            "Press q in the camera window, or Ctrl+C in the terminal, to stop."
        )
        print(
            f"Camera only counts boxes touching the {args.line_orientation} "
            f"line at {args.line_position}."
        )

        ultrasonic_state = {
            "latest_distance_cm": None,
            "distance_detected": False,
            "close_frames": 0,
            "clear_frames": 0,
            "confirmed_object": False,
            "confirmation_time": None,
            "first_close_time": None,
            "last_close_time": None,
            "active_object_id": None,
            "ended_events": [],
        }
        ultrasonic_lock = threading.Lock()
        ultrasonic_stop = threading.Event()
        ultrasonic_thread = threading.Thread(
            target=ultrasonic_worker,
            args=(
                handle,
                timeout_sec,
                args.threshold_cm,
                args.required_frames,
                args.clear_frames,
                ultrasonic_state,
                ultrasonic_lock,
                ultrasonic_stop,
            ),
            daemon=True,
        )
        ultrasonic_thread.start()

        camera_frame_counts = {}
        camera_consecutive_frames = {}
        camera_confirmed_materials = {}
        pending_open_events = []
        event_by_object_id = {}

        try:
            while True:
                now = time.monotonic()
                camera_materials = camera_detects_materials(
                    picam2,
                    model,
                    args.conf,
                    args.line_orientation,
                    args.line_position,
                )

                with ultrasonic_lock:
                    active_object_id = ultrasonic_state["active_object_id"]
                    close_frames = ultrasonic_state["close_frames"]
                    clear_frames = ultrasonic_state["clear_frames"]
                    confirmed_object = ultrasonic_state["confirmed_object"]
                    confirmation_time = ultrasonic_state["confirmation_time"]
                    first_close_time = ultrasonic_state["first_close_time"]
                    last_close_time = ultrasonic_state["last_close_time"]
                    ended_events = ultrasonic_state["ended_events"][:]
                    ultrasonic_state["ended_events"].clear()

                if active_object_id is not None:
                    if active_object_id not in camera_frame_counts:
                        camera_frame_counts[active_object_id] = {
                            material: 0 for material in material_actions
                        }
                        camera_consecutive_frames[active_object_id] = {
                            material: 0 for material in material_actions
                        }
                        camera_confirmed_materials[active_object_id] = set()

                    for material in material_actions:
                        if material in camera_materials:
                            camera_frame_counts[active_object_id][material] += 1
                            camera_consecutive_frames[active_object_id][material] += 1
                        else:
                            camera_consecutive_frames[active_object_id][material] = 0

                        if (
                            camera_consecutive_frames[active_object_id][material]
                            >= args.camera_frames
                        ):
                            camera_confirmed_materials[active_object_id].add(material)

                    counts = camera_frame_counts[active_object_id]
                    confirmed_materials = camera_confirmed_materials[active_object_id]
                    material = best_material(counts, confirmed_materials)
                    if confirmed_object and material is not None:
                        event = event_by_object_id.get(active_object_id)
                        action = material_actions[material]

                        if event is None:
                            start_time = now
                            open_time = start_time + action["open_delay"]
                            close_time = scaled_gate_close_time(
                                open_time,
                                first_close_time or last_close_time or now,
                                last_close_time or now,
                                action["extra_open_sec"],
                            )
                            event = {
                                "object_id": active_object_id,
                                "material": material,
                                "servo_id": action["servo_id"],
                                "servo_idx": action["servo_idx"],
                                "open_time": open_time,
                                "close_time": close_time,
                                "close_frames": close_frames,
                                "ended": False,
                                "opened": False,
                            }
                            pending_open_events.append(event)
                            event_by_object_id[active_object_id] = event
                            remaining = max(0.0, event["open_time"] - now)
                            print(
                                f"Object {active_object_id} confirmed as {material}. "
                                f"Queued servo {action['servo_id']} in "
                                f"{remaining:.1f}s."
                            )
                        elif event["material"] == material:
                            event["close_time"] = scaled_gate_close_time(
                                event["open_time"],
                                first_close_time or last_close_time or now,
                                last_close_time or now,
                                action["extra_open_sec"],
                            )
                            event["close_frames"] = close_frames
                        elif not event["opened"]:
                            old_material = event["material"]
                            event["material"] = material
                            event["servo_id"] = action["servo_id"]
                            event["servo_idx"] = action["servo_idx"]
                            event["open_time"] = now + action["open_delay"]
                            event["close_time"] = scaled_gate_close_time(
                                event["open_time"],
                                first_close_time or last_close_time or now,
                                last_close_time or now,
                                action["extra_open_sec"],
                            )
                            event["close_frames"] = close_frames
                            event["ended"] = False
                            print(
                                f"Object {active_object_id} material changed from "
                                f"{old_material} to {material} before opening."
                            )

                print(
                    "Camera detected: "
                    f"{', '.join(sorted(camera_materials)) if camera_materials else 'none'}"
                )
                if active_object_id is None:
                    print("Camera frames: no active ultrasonic object")
                else:
                    print(
                        f"Camera frames for object {active_object_id}: "
                        + ", ".join(
                            f"{material}={camera_frame_counts[active_object_id][material]}"
                            f"{' confirmed' if material in camera_confirmed_materials[active_object_id] else ''}"
                            for material in material_actions
                        )
                    )
                print(
                    f"Ultrasonic snapshot: close={close_frames}, clear={clear_frames}, "
                    f"confirmed={'yes' if confirmed_object else 'no'}"
                )
                print(f"Queued: {queue_summary(pending_open_events, material_actions)}")

                for ended_event in ended_events:
                    object_id = ended_event["object_id"]
                    event_close_frames = ended_event["close_frames"]
                    counts = camera_frame_counts.get(
                        object_id, {material: 0 for material in material_actions}
                    )
                    confirmed_materials = camera_confirmed_materials.get(object_id, set())
                    material = best_material(counts, confirmed_materials)
                    event = event_by_object_id.get(object_id)

                    if event is not None:
                        action = material_actions[event["material"]]
                        event["close_time"] = scaled_gate_close_time(
                            event["open_time"],
                            ended_event["first_close_time"],
                            ended_event["last_close_time"],
                            action["extra_open_sec"],
                        )
                        event["close_frames"] = event_close_frames
                        event["ended"] = True
                        remaining = max(0.0, event["close_time"] - now)
                        print(
                            f"Object {object_id} ended after {event_close_frames} "
                            f"ultrasonic close frames. Servo {event['servo_id']} "
                            f"will close in {remaining:.1f}s."
                        )
                    elif material is None:
                        print(
                            f"Object {object_id} ended after {event_close_frames} "
                            "ultrasonic close frames, but camera did not have one "
                            "clear supported material winner."
                        )
                    else:
                        action = material_actions[material]
                        open_time = ended_event["confirmation_time"] + action["open_delay"]
                        close_time = scaled_gate_close_time(
                            open_time,
                            ended_event["first_close_time"],
                            ended_event["last_close_time"],
                            action["extra_open_sec"],
                        )
                        event = {
                            "object_id": object_id,
                            "material": material,
                            "servo_id": action["servo_id"],
                            "servo_idx": action["servo_idx"],
                            "open_time": open_time,
                            "close_time": close_time,
                            "close_frames": event_close_frames,
                            "ended": True,
                            "opened": False,
                        }
                        pending_open_events.append(event)
                        event_by_object_id[object_id] = event
                        remaining = max(0.0, open_time - now)
                        print(
                            f"Object {object_id} ended after {event_close_frames} "
                            f"ultrasonic close frames as {material}. Queued servo "
                            f"{action['servo_id']} in {remaining:.1f}s for "
                            f"delayed close. Queued: "
                            f"{queue_summary(pending_open_events, material_actions)}."
                        )

                    camera_frame_counts.pop(object_id, None)
                    camera_consecutive_frames.pop(object_id, None)
                    camera_confirmed_materials.pop(object_id, None)

                pending_open_events.sort(key=lambda event: event["open_time"])
                for event in pending_open_events:
                    if event["opened"] or event["open_time"] > now:
                        continue
                    material = event["material"]
                    servo_id = event["servo_id"]
                    servo_idx = event["servo_idx"]
                    current_angles = move_one_servo(
                        ser,
                        current_angles,
                        servo_idx,
                        open_angles[servo_idx],
                        servos,
                        servo_ids,
                    )
                    print(
                        f"{material.title()} servo {servo_id} opened. It will stay "
                        f"open for {GATE_OPEN_DURATION_MULTIPLIER:.1f}x the "
                        "ultrasonic object duration. "
                        f"Queued: {queue_summary(pending_open_events, material_actions)}."
                    )
                    event["opened"] = True

                for event in pending_open_events[:]:
                    if (
                        not event["opened"]
                        or not event["ended"]
                        or time.monotonic() < event["close_time"]
                    ):
                        continue
                    servo_id = event["servo_id"]
                    pending_open_events.remove(event)
                    event_by_object_id.pop(event["object_id"], None)

                    keep_open = any(
                        other["opened"]
                        and other["servo_id"] == servo_id
                        and other["close_time"] > time.monotonic()
                        for other in pending_open_events
                    )
                    if keep_open:
                        continue

                    servo_idx = servo_ids.index(servo_id)
                    current_angles = move_one_servo(
                        ser,
                        current_angles,
                        servo_idx,
                        closed_angles[servo_idx],
                        servos,
                        servo_ids,
                    )
                    print(f"Servo {servo_id} closed.")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                time.sleep(READ_INTERVAL_SEC)

        except KeyboardInterrupt:
            pass
        finally:
            ultrasonic_stop.set()
            ultrasonic_thread.join(timeout=1.0)
            smooth_move(
                ser, current_angles, closed_angles, servos=servos, servo_ids=servo_ids
            )
            lgpio.gpio_write(handle, TRIG_GPIO, 0)
            lgpio.gpiochip_close(handle)
            picam2.stop()
            cv2.destroyAllWindows()
            print("Stopped cleanly")


if __name__ == "__main__":
    main()
