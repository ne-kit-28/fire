import os, time, argparse
import numpy as np
import cv2
import RPi.GPIO as GPIO
from picamera2 import Picamera2

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

def us_to_dc(us, freq):
    period_us = 1e6 / freq
    return max(0.0, min(100.0, us / period_us * 100.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/fire_classifier3.onnx")
    ap.add_argument("--input_size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--streak", type=int, default=3)
    ap.add_argument("--alarm_pin", type=int, default=17)
    ap.add_argument("--cooldown", type=float, default=2.0)
    ap.add_argument("--save_dir", default="captures")
    ap.add_argument("--cam_w", type=int, default=320)
    ap.add_argument("--cam_h", type=int, default=240)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--annotate", action="store_true")
    # PWM (вертикальный канал подвеса)
    ap.add_argument("--pwm_pin", type=int, default=18)
    ap.add_argument("--pwm_freq", type=int, default=50)
    ap.add_argument("--pwm_min_us", type=int, default=1000)
    ap.add_argument("--pwm_max_us", type=int, default=2000)
    ap.add_argument("--pwm_center_us", type=int, default=1600)
    ap.add_argument("--pwm_step_us", type=int, default=100)
    ap.add_argument("--pwm_invert", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    GPIO.setup(args.alarm_pin, GPIO.OUT)
    GPIO.output(args.alarm_pin, GPIO.LOW)

    GPIO.setup(args.pwm_pin, GPIO.OUT)
    pwm = GPIO.PWM(args.pwm_pin, args.pwm_freq)
    pwm_cur_us = int(np.clip(args.pwm_center_us, args.pwm_min_us, args.pwm_max_us))
    pwm.start(us_to_dc(pwm_cur_us, args.pwm_freq))

    net = cv2.dnn.readNetFromONNX(args.model)
    in_w = in_h = args.input_size

    cam = Picamera2()
    cfg = cam.create_preview_configuration(main={"format": 'RGB888', "size": (args.cam_w, args.cam_h)})
    cam.configure(cfg)
    cam.start()
    time.sleep(0.2)

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
                GPIO.output(args.alarm_pin, GPIO.HIGH)
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
                    cv2.imwrite(os.path.join(args.save_dir, fn), img[:, :, ::-1])
                    last_save = now
            elif neg >= args.streak:
                GPIO.output(args.alarm_pin, GPIO.LOW)

            # Вертикальное выравнивание по строке с максимумом
            if fire_any:
                center_r = args.grid // 2
                err = max_r - center_r
                if err != 0:
                    step = args.pwm_step_us * (1 if err > 0 else -1)
                    if args.pwm_invert:
                        step = -step
                    pwm_cur_us = int(np.clip(pwm_cur_us + step, args.pwm_min_us, args.pwm_max_us))
                    pwm.ChangeDutyCycle(us_to_dc(pwm_cur_us, args.pwm_freq))

            cells_on = np.argwhere(fire_cells)
            cells_on_str = ",".join([f"({r},{c})" for r, c in cells_on]) if cells_on.size else "-"
            print(f"max={prob_max:.2f} at ({max_r},{max_c}); on: {cells_on_str}; PWM_us={pwm_cur_us}; pos={pos} neg={neg}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    finally:
        GPIO.output(args.alarm_pin, GPIO.LOW)
        pwm.ChangeDutyCycle(us_to_dc(args.pwm_center_us, args.pwm_freq))
        time.sleep(0.2)
        pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()