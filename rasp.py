import os
import time
import argparse
import numpy as np
import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def load_interpreter(model_path):
    intrp = Interpreter(model_path=model_path)
    intrp.allocate_tensors()
    in_det = intrp.get_input_details()[0]
    out_det = intrp.get_output_details()[0]
    return intrp, in_det, out_det

def preprocess_bgr(frame_bgr, target_size):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img[np.newaxis, ...]  # uint8 [1,H,W,3]

def dequantize(output, out_det):
    arr = output.squeeze()
    if out_det["dtype"] in (np.uint8, np.int8):
        scale, zero = out_det["quantization"]
        return float(scale * (arr.astype(np.int32) - zero))
    return float(arr.astype(np.float32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/fire_classifier_int8.tflite")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--pin", type=int, default=17)
    ap.add_argument("--streak", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--save_dir", default="captures")
    ap.add_argument("--cam_w", type=int, default=320)
    ap.add_argument("--cam_h", type=int, default=240)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    GPIO.setup(args.pin, GPIO.OUT)
    GPIO.output(args.pin, GPIO.LOW)

    intrp, in_det, out_det = load_interpreter(args.model)
    _, in_h, in_w, _ = in_det["shape"]

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (args.cam_w, args.cam_h)})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.2)

    pos_streak = 0
    neg_streak = 0
    last_save_ts = 0.0

    try:
        while True:
            frame = picam2.capture_array()
            inp = preprocess_bgr(frame, (in_w, in_h))
            intrp.set_tensor(in_det["index"], inp)
            intrp.invoke()
            out = intrp.get_tensor(out_det["index"])
            prob = dequantize(out, out_det)

            is_fire = prob >= args.threshold
            if is_fire:
                pos_streak += 1
                neg_streak = 0
            else:
                neg_streak += 1
                pos_streak = 0

            if pos_streak >= args.streak:
                GPIO.output(args.pin, GPIO.HIGH)
                now = time.time()
                if now - last_save_ts >= args.cooldown:
                    fname = time.strftime("%Y%m%d_%H%M%S") + f"_{prob:.2f}.jpg"
                    cv2.imwrite(os.path.join(args.save_dir, fname), frame)
                    last_save_ts = now
            elif neg_streak >= args.streak:
                GPIO.output(args.pin, GPIO.LOW)

            print(f"prob={prob:.2f} | state={'FIRE' if is_fire else 'normal'} | pos={pos_streak} neg={neg_streak}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        GPIO.output(args.pin, GPIO.LOW)
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()