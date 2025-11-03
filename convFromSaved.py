import argparse, os, sys, subprocess

def ensure_onnx_path(saved_model_dir, onnx_out):
    onnx_out = os.path.normpath(onnx_out)
    if os.path.isdir(onnx_out) or onnx_out.endswith(os.sep) or onnx_out == ".":
        base = os.path.basename(os.path.normpath(saved_model_dir))
        onnx_out = os.path.join(onnx_out, f"{base}.onnx")
    if not onnx_out.lower().endswith(".onnx"):
        onnx_out += ".onnx"
    os.makedirs(os.path.dirname(onnx_out), exist_ok=True)
    return onnx_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True)
    ap.add_argument("--onnx_out", required=True)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    sm_dir = os.path.normpath(args.saved_model_dir)
    onnx_out = ensure_onnx_path(sm_dir, args.onnx_out)

    sm_pb = os.path.join(sm_dir, "saved_model.pb")
    if not os.path.isfile(sm_pb):
        raise FileNotFoundError(f"SavedModel not found: {sm_pb}")

    try:
        import tf2onnx as t2o
        # Попробуем Python API
        if hasattr(t2o, "convert") and hasattr(t2o.convert, "from_saved_model"):
            t2o.convert.from_saved_model(sm_dir, output_path=onnx_out, opset=args.opset)
        else:
            # Fallback на CLI
            raise AttributeError("tf2onnx.convert.from_saved_model not available")
    except Exception:
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", sm_dir,
            "--opset", str(args.opset),
            "--output", onnx_out
        ]
        subprocess.check_call(cmd)

    print(f"OK -> {onnx_out}")

if __name__ == "__main__":
    main()