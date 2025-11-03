import argparse, os, sys, subprocess

def ensure_onnx_path(saved_model_dir, onnx_out):
    onnx_out = os.path.normpath(onnx_out)
    if os.path.isdir(onnx_out) or onnx_out.endswith(os.sep) or onnx_out in (".", "./", ".\\"):
        base = os.path.basename(os.path.normpath(saved_model_dir))
        onnx_out = os.path.join(onnx_out, f"{base}.onnx")
    if not onnx_out.lower().endswith(".onnx"):
        onnx_out += ".onnx"
    out_dir = os.path.dirname(onnx_out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    return onnx_out

def discover_io(saved_model_dir, signature_def):
    try:
        import tensorflow as tf
        m = tf.saved_model.load(saved_model_dir)
        sig = m.signatures[signature_def]
        ins, outs = list(sig.inputs), list(sig.outputs)
        input_candidates = []
        for t in ins:
            name = t.name
            rank = getattr(t, "shape", None)
            rank_len = len(rank) if hasattr(rank, "__len__") else None
            if (rank_len == 4) or ("input" in name.lower()):
                input_candidates.append(name)
        if not input_candidates and ins:
            input_candidates = [ins[0].name]
        output_names = [outs[0].name] if outs else []
        return input_candidates, output_names
    except Exception:
        return [], []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved_model_dir", required=True)
    ap.add_argument("--onnx_out", required=True)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--signature_def", default="serving_default")
    ap.add_argument("--inputs", nargs="*", default=None)
    ap.add_argument("--outputs", nargs="*", default=None)
    ap.add_argument("--inputs_as_nchw", nargs="*", default=None)
    args = ap.parse_args()

    sm_dir = os.path.normpath(args.saved_model_dir)
    sm_pb = os.path.join(sm_dir, "saved_model.pb")
    if not os.path.isfile(sm_pb):
        raise FileNotFoundError(f"SavedModel not found: {sm_pb}")

    onnx_out = ensure_onnx_path(sm_dir, args.onnx_out)

    in_names, out_names = args.inputs, args.outputs
    if not in_names or not out_names:
        auto_in, auto_out = discover_io(sm_dir, args.signature_def)
        if not in_names and auto_in: in_names = [auto_in[0]]
        if not out_names and auto_out: out_names = [auto_out[0]]
    if not in_names or not out_names:
        raise RuntimeError("Failed to determine inputs/outputs. Provide --inputs and --outputs explicitly.")

    inputs_as_nchw = args.inputs_as_nchw or in_names

    print(f"SavedModel: {sm_dir}")
    print(f"Inputs: {in_names} | Outputs: {out_names} | Inputs-as-NCHW: {inputs_as_nchw}")
    print(f"ONNX out: {onnx_out}")

    try:
        import tf2onnx as t2o
        if hasattr(t2o, "convert") and hasattr(t2o.convert, "from_saved_model"):
            t2o.convert.from_saved_model(
                sm_dir,
                output_path=onnx_out,
                opset=args.opset,
                signature_def=args.signature_def,
                input_names=in_names,
                output_names=out_names,
                inputs_as_nchw=inputs_as_nchw,
            )
        else:
            raise AttributeError("tf2onnx.convert.from_saved_model not available")
    except Exception:
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", sm_dir,
            "--signature_def", args.signature_def,
            "--inputs", ",".join(in_names),
            "--outputs", ",".join(out_names),
            "--inputs-as-nchw", ",".join(inputs_as_nchw),
            "--opset", str(args.opset),
            "--output", onnx_out,
        ]
        subprocess.check_call(cmd)

    print(f"OK -> {onnx_out}")

if __name__ == "__main__":
    main()