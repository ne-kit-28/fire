# from keras.models import load_model
# m = load_model("model/fire_classifier.keras", compile=False)
# # m.save("model/fire_classifier.h5")          # H5-формат
# # или экспортируйте SavedModel (Keras 3):
# m.export("model/saved_model")

from keras.models import load_model
m = load_model("model/fire_classifier.keras", compile=False)
m.export("test")  # корень проекта