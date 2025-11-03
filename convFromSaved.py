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
    import tensorflow as tf
    m = tf.saved_model.load(saved_model_dir)
    sig = m.signatures[signature_def]
    ins = [t.name for t in sig.inputs]
    outs = [t.name for t in sig.outputs]
    # фильтруем «мусор»
    good_ins = [n for n in ins if not n.lower().startswith("unknown")]
    # приоритет по подстрокам
    for key in ("input:", "serving_default_input", "input_layer", "input_"):
        cand = [n for n in good_ins if key in n]
        if cand:
            input_name = cand[0]; break
    else:
        input_name = good_ins[0] if good_ins else ins[0]
    # выход: предпочитаем Identity или prob
    good_outs = [n for n in outs if not n.lower().startswith("unknown")]
    for key in ("Identity:", "prob:", "output"):
        cand = [n for n in good_outs if key.split(":")[0] in n]
        if cand:
            output_name = cand[0]; break
    else:
        output_name = good_outs[0] if good_outs else outs[0]
    return input_name, output_name

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

    # имена I/O
    if args.inputs and args.outputs:
        in_name = args.inputs[0]
        out_name = args.outputs[0]
    else:
        try:
            in_name, out_name = discover_io(sm_dir, args.signature_def)
        except Exception as e:
            raise RuntimeError(f"Не удалось определить вход/выход: {e}\nУкажите --inputs и --outputs явно.")
    inputs_as_nchw = args.inputs_as_nchw[0] if args.inputs_as_nchw else in_name

    print(f"SavedModel: {sm_dir}")
    print(f"Input: {in_name} | Output: {out_name} | Inputs-as-NCHW: {inputs_as_nchw}")
    print(f"ONNX out: {onnx_out}")

    # попытка через API, иначе CLI
    try:
        import tf2onnx as t2o
        if hasattr(t2o, "convert") and hasattr(t2o.convert, "from_saved_model"):
            t2o.convert.from_saved_model(
                sm_dir,
                output_path=onnx_out,
                opset=args.opset,
                signature_def=args.signature_def,
                input_names=[in_name],
                output_names=[out_name],
                inputs_as_nchw=[inputs_as_nchw],
            )
        else:
            raise AttributeError("tf2onnx.convert.from_saved_model not available")
    except Exception:
        cmd = [
            sys.executable, "-m", "tf2onnx.convert",
            "--saved-model", sm_dir,
            "--signature_def", args.signature_def,
            "--inputs", in_name,
            "--outputs", out_name,
            "--inputs-as-nchw", inputs_as_nchw,
            "--opset", str(args.opset),
            "--output", onnx_out,
        ]
        subprocess.check_call(cmd)

    print(f"OK -> {onnx_out}")

if __name__ == "__main__":
    main()