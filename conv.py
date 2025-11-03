# export_savedmodel_k3_safe.py
import os, time, shutil, argparse
from pathlib import Path
from keras.models import load_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras_in", required=True)
    ap.add_argument("--base_dir", default=r"C:\tmp\fire_export")
    args = ap.parse_args()

    base = Path(args.base_dir)
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)

    export_dir = base / f"saved_model_k3_{int(time.time())}"
    m = load_model(args.keras_in, compile=False)
    m.export(str(export_dir))
    print("SavedModel:", export_dir)

if __name__ == "__main__":
    main()