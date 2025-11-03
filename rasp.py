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
    ap.add_argument("--cam_w", type=int, default=480)
    ap.add_argument("--cam_h", type=int, default=360)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    GPIO.setup(args.pin, GPIO.OUT)
    GPIO.output(args.pin, GPIO.LOW)

    net = cv2.dnn.readNetFromONNX(args.model)
    in_w = in_h = args.input_size

    cam = Picamera2()
    cfg = cam.create_preview_configuration(main={"format": 'RGB888', "size": (args.cam_w, args.cam_h)})
    cam.configure(cfg); cam.start(); time.sleep(0.2)

    pos = neg = 0
    last_save = 0.0

    try:
        while True:
            frame = cam.capture_array()  # RGB (H,W,3)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            H, W = frame.shape[:2]
            gy = np.linspace(0, H, args.grid + 1, dtype=int)
            gx = np.linspace(0, W, args.grid + 1, dtype=int)

            tiles = []
            for i in range(args.grid):
                for j in range(args.grid):
                    tiles.append(frame[gy[i]:gy[i+1], gx[j]:gx[j+1]])

            blob = cv2.dnn.blobFromImages(tiles, scalefactor=1.0, size=(in_w, in_h), swapRB=False)
            net.setInput(blob)
            out = net.forward()
            probs = out.reshape(-1)  # длина grid*grid, [0..1]
            probs2d = probs.reshape(args.grid, args.grid)

            fire_cells = probs2d >= args.threshold
            fire_any = bool(fire_cells.any())
            prob_max = float(probs.max())
            max_idx = int(probs.argmax())
            max_r, max_c = divmod(max_idx, args.grid)

            if fire_any:
                pos += 1; neg = 0
            else:
                neg += 1; pos = 0

            if pos >= args.streak:
                GPIO.output(args.pin, GPIO.HIGH)
                now = time.time()
                if now - last_save >= args.cooldown:
                    img = frame.copy()
                    if args.annotate:
                        # белая сетка, зелёная рамка вокруг сработавших ячеек
                        for k in range(1, args.grid):
                            cv2.line(img, (gx[k], 0), (gx[k], H), (255,255,255), 1)
                            cv2.line(img, (0, gy[k]), (W, gy[k]), (255,255,255), 1)
                        for i in range(args.grid):
                            for j in range(args.grid):
                                if fire_cells[i, j]:
                                    cv2.rectangle(img, (gx[j], gy[i]), (gx[j+1], gy[i+1]), (0,255,0), 2)
                        cv2.putText(img, f"max:{prob_max:.2f} at ({max_r},{max_c})", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    fn = time.strftime("%Y%m%d_%H%M%S") + f"_{prob_max:.2f}.jpg"
                    cv2.imwrite(os.path.join(args.save_dir, fn), img[:, :, ::-1])  # RGB->BGR
                    last_save = now
            elif neg >= args.streak:
                GPIO.output(args.pin, GPIO.LOW)

            # Краткий лог по сетке: максимум и список сработавших
            cells_on = np.argwhere(fire_cells)
            cells_on_str = ",".join([f"({r},{c})" for r, c in cells_on]) if cells_on.size else "-"
            print(f"max={prob_max:.2f} at ({max_r},{max_c}); on: {cells_on_str}; pos={pos} neg={neg}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        GPIO.output(args.pin, GPIO.LOW)
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()