
import argparse, tensorflow as tf
from keras import layers, models

def get_ds(root, img, bs, val_split=0.0):
    common = dict(image_size=(img, img), batch_size=bs, class_names=['no_fire','fire'], label_mode='binary', seed=42)
    if val_split>0:
        tr = tf.keras.utils.image_dataset_from_directory(root, shuffle=True, validation_split=val_split, subset='training', **common)
        va = tf.keras.utils.image_dataset_from_directory(root, shuffle=False,  validation_split=val_split, subset='validation', **common)
    else:
        tr = tf.keras.utils.image_dataset_from_directory(root, shuffle=True, **common); va=None
    aug = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.1)])
    AUT = tf.data.AUTOTUNE
    tr = tr.cache().map(lambda x,y:(aug(x),y), num_parallel_calls=AUT).prefetch(AUT)
    if va is not None: va = va.cache().prefetch(AUT)
    return tr, va

def build(img):
    inp = layers.Input(name="input", shape=(img,img,3))
    x = layers.Rescaling(1/255.0)(inp)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid", name="prob")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)          # dataset/ (no_fire/, fire/)
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val_split", type=float, default=0.2)  # можно 0.0, если val не нужен
    ap.add_argument("--out_keras", default="model/fire_conv.keras")
    ap.add_argument("--out_saved", default=r"C:\tmp\fire_export")
    args = ap.parse_args()

    tr, va = get_ds(args.data, args.img, args.bs, args.val_split)
    m = build(args.img)
    cbs = [tf.keras.callbacks.ModelCheckpoint(args.out_keras, save_best_only=True, monitor="val_auc" if va else "auc", mode="max"),
           tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc" if va else "auc", mode="max")]
    m.fit(tr, validation_data=va, epochs=args.epochs, callbacks=cbs)
    m.export(args.out_saved)  # Keras 3 export SavedModel
    print("Saved:", args.out_keras, "and", args.out_saved)

if __name__ == "__main__":
    main()