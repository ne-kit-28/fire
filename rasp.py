import os, time, argparse
import numpy as np
import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/fire_classifier.onnx")
    ap.add_argument("--input_size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--streak", type=int, default=3)
    ap.add_argument("--pin", type=int, default=17)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--save_dir", default="captures")
    ap.add_argument("--cam_w", type=int, default=320)
    ap.add_argument("--cam_h", type=int, default=240)
    # Если ваш SavedModel->ONNX был с --inputs-as-nchw, оставьте как есть.
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    GPIO.setup(args.pin, GPIO.OUT)
    GPIO.output(args.pin, GPIO.LOW)

    net = cv2.dnn.readNetFromONNX(args.model)
    in_w = in_h = args.input_size

    cam = Picamera2()
    # 3-канальный формат, чтобы не получить 4 канала
    cfg = cam.create_preview_configuration(main={"format": 'RGB888', "size": (args.cam_w, args.cam_h)})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)

    pos = neg = 0
    last_save = 0.0

    try:
        while True:
            frame = cam.capture_array()  # RGB, shape (H,W,3)
            # На всякий случай отрезаем альфа, если вдруг пришёл BGRA/RGBA
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Модель обучалась на RGB; blobFromImage по умолчанию ждёт BGR,
            # поэтому swapRB=False (ничего не меняем), т.к. у нас уже RGB.
            blob = cv2.dnn.blobFromImage(
                frame, scalefactor=1.0, size=(in_w, in_h), swapRB=False
            )  # -> NCHW: (1,3,H,W)
            net.setInput(blob)
            out = net.forward()
            prob = float(out.reshape(-1)[0])

            fire = prob >= args.threshold
            if fire:
                pos += 1; neg = 0
            else:
                neg += 1; pos = 0

            if pos >= args.streak:
                GPIO.output(args.pin, GPIO.HIGH)
                now = time.time()
                if now - last_save >= args.cooldown:
                    fn = time.strftime("%Y%m%d_%H%M%S") + f"_{prob:.2f}.jpg"
                    cv2.imwrite(os.path.join(args.save_dir, fn), frame[:, :, ::-1])  # сохраняем как BGR для cv2
                    last_save = now
            elif neg >= args.streak:
                GPIO.output(args.pin, GPIO.LOW)

            print(f"prob={prob:.2f} state={'FIRE' if fire else 'normal'} pos={pos} neg={neg}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        GPIO.output(args.pin, GPIO.LOW)
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()