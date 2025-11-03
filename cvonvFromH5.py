# py -m tf2onnx.convert --keras model/fire_classifier.h5 --opset 13 --output model/fire_classifier.onnx

# conv_onnx.py
import tensorflow as tf, tf2onnx
m = tf.keras.models.load_model("model/fire_classifier.h5", compile=False)
spec = (tf.TensorSpec((1, 64, 64, 3), tf.float32, name="input"),)
tf2onnx.convert.from_keras(m, input_signature=spec, opset=13,
                           output_path="model/fire_classifier.onnx")
print("Saved: model/fire_classifier.onnx")