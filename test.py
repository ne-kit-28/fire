import tensorflow as tf
m = tf.saved_model.load(r"C:\tmp\fire_export") # путь к вашей папке SavedModel
sig = m.signatures["serving_default"]
print("inputs:", [x.name for x in sig.inputs])
print("outputs:", [x.name for x in sig.outputs])