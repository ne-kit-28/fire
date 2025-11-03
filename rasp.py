import os, time, argparse
import numpy as np, cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/fire_classifier.tflite")
    ap.add_argument("--input_size", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0)  # 1.0 для int8/uint8, 0.0039215686 для float32
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--streak", type=int, default=3)
    ap.add_argument("--pin", type=int, default=17)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--save_dir", default="captures")
    ap.add_argument("--cam_w", type=int, default=320)
    ap.add_argument("--cam_h", type=int, default=240)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    GPIO.setup(args.pin, GPIO.OUT)
    GPIO.output(args.pin, GPIO.LOW)

    net = cv2.dnn.readNetFromTFLite(args.model)

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (args.cam_w, args.cam_h)})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.2)

    pos_streak = 0
    neg_streak = 0
    last_save = 0.0

    try:
        while True:
            frame = picam2.capture_array()
            blob = cv2.dnn.blobFromImage(frame, scalefactor=args.scale, size=(args.input_size, args.input_size), swapRB=True)
            net.setInput(blob)
            out = net.forward()
            prob = float(out.reshape(-1)[0])

            is_fire = prob >= args.threshold
            if is_fire:
                pos_streak += 1; neg_streak = 0
            else:
                neg_streak += 1; pos_streak = 0

            if pos_streak >= args.streak:
                GPIO.output(args.pin, GPIO.HIGH)
                now = time.time()
                if now - last_save >= args.cooldown:
                    fn = time.strftime("%Y%m%d_%H%M%S") + f"_{prob:.2f}.jpg"
                    cv2.imwrite(os.path.join(args.save_dir, fn), frame)
                    last_save = now
            elif neg_streak >= args.streak:
                GPIO.output(args.pin, GPIO.LOW)

            print(f"prob={prob:.2f} state={'FIRE' if is_fire else 'normal'} pos={pos_streak} neg={neg_streak}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        GPIO.output(args.pin, GPIO.LOW)
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()