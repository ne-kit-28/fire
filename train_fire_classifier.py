# train_fire_classifier_noval.py
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def get_datasets(train_dir, img_size, batch_size, val_split):
    common = dict(image_size=(img_size, img_size),
                  batch_size=batch_size,
                  class_names=['no_fire', 'fire'],
                  label_mode='binary',
                  seed=42)
    if val_split and val_split > 0.0:
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir, shuffle=True, validation_split=val_split, subset='training', **common)
        val_ds = keras.utils.image_dataset_from_directory(
            train_dir, shuffle=False, validation_split=val_split, subset='validation', **common)
    else:
        train_ds = keras.utils.image_dataset_from_directory(
            train_dir, shuffle=True, **common)
        val_ds = None

    autotune = tf.data.AUTOTUNE
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.cache().map(lambda x, y: (aug(x), y), num_parallel_calls=autotune).prefetch(autotune)
    if val_split and val_split > 0.0:
        val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds

def build_model(img_size):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.SeparableConv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(96, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def get_rep_ds(train_dir, img_size, max_samples=200):
    ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels=None,                 # без меток
        image_size=(img_size, img_size),
        batch_size=1,
        shuffle=True,
        seed=123
    )
    ds = ds.unbatch()
    reps = []
    for img in ds.take(max_samples):
        reps.append(tf.cast(img, tf.float32))
    return reps

def export_tflite_int8(model, rep_imgs, out_path):
    def representative_data_gen():
        for img in rep_imgs:
            yield [tf.expand_dims(img, 0)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val_split", type=float, default=0.0)  # 0.0 = без val
    ap.add_argument("--model_out", default="fire_classifier.keras")
    ap.add_argument("--tflite_out", default="fire_classifier_int8.tflite")
    ap.add_argument("--rep_samples", type=int, default=200)
    args = ap.parse_args()

    train_ds, val_ds = get_datasets(args.train_dir, args.img_size, args.batch_size, args.val_split)
    model = build_model(args.img_size)

    callbacks = []
    if args.val_split and args.val_split > 0.0:
        callbacks.append(keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True, monitor="val_auc", mode="max"))
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    else:
        callbacks.append(keras.callbacks.ModelCheckpoint(args.model_out, save_best_only=True, monitor="auc", mode="max"))
        model.fit(train_ds, epochs=args.epochs, callbacks=callbacks)

    model.save(args.model_out)
    rep_imgs = get_rep_ds(args.train_dir, args.img_size, max_samples=args.rep_samples)
    export_tflite_int8(model, rep_imgs, args.tflite_out)

if __name__ == "__main__":
    main()