"""
Age & Gender Estimation - Training Script
==========================================
Run this on Google Colab (Runtime > Change runtime type > GPU) or Kaggle Notebooks.
It will NOT run usefully on a plain laptop CPU for the full dataset (it will just
be slow) - a free Colab GPU trains this in ~20-40 minutes.

STEP 1: Get the UTKFace dataset
--------------------------------
Option A (Kaggle, recommended):
    1. Create a free Kaggle account & an API token (kaggle.json)
    2. In Colab:
        from google.colab import files
        files.upload()  # upload kaggle.json
        !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/
        !chmod 600 ~/.kaggle/kaggle.json
        !kaggle datasets download -d jangedoo/utkface-new
        !unzip -q utkface-new.zip -d utkface_data

Option B: Download manually from
    https://susanqq.github.io/UTKFace/
    and upload the "UTKFace" folder to Colab / Google Drive.

The dataset is ~20,000 face images named like:
    [age]_[gender]_[race]_[date&time].jpg
    e.g. 25_0_2_20170116174525125.jpg  -> age=25, gender=0 (male)
    gender: 0 = male, 1 = female

STEP 2: Set DATA_DIR below to the folder containing the images, then run:
    python train.py
(or run each cell in a notebook - the code is written to work either way)
"""

import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model_architecture import build_model, IMG_SIZE

# ------------------------------------------------------------------
# CONFIG - edit these
# ------------------------------------------------------------------
DATA_DIR = "utkface_data/UTKFace"   # <-- change to wherever your images live
BATCH_SIZE = 64
EPOCHS = 30
MODEL_OUT_PATH = "age_gender_model.h5"

# ------------------------------------------------------------------
# 1. Build a list of (filepath, age, gender) from filenames
# ------------------------------------------------------------------
FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)_")


def load_labels(data_dir):
    filepaths, ages, genders = [], [], []
    for fname in os.listdir(data_dir):
        match = FILENAME_PATTERN.match(fname)
        if not match:
            continue  # skip any file that doesn't follow the naming convention
        age, gender = int(match.group(1)), int(match.group(2))
        if age > 100:  # drop a handful of noisy/mislabeled outliers
            continue
        filepaths.append(os.path.join(data_dir, fname))
        ages.append(age)
        genders.append(gender)
    return filepaths, np.array(ages, dtype=np.float32), np.array(genders, dtype=np.float32)


# ------------------------------------------------------------------
# 2. tf.data pipeline: decode image, resize, normalize
# ------------------------------------------------------------------
def _decode_and_preprocess(filepath, age, gender):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, {"age_output": age, "gender_output": gender}


def make_dataset(filepaths, ages, genders, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, ages, genders))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths))
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    print("Loading labels from filenames...")
    filepaths, ages, genders = load_labels(DATA_DIR)
    print(f"Found {len(filepaths)} labeled images.")

    train_fp, val_fp, train_age, val_age, train_gender, val_gender = train_test_split(
        filepaths, ages, genders, test_size=0.15, random_state=42
    )

    train_ds = make_dataset(train_fp, train_age, train_gender, BATCH_SIZE, shuffle=True)
    val_ds = make_dataset(val_fp, val_age, val_gender, BATCH_SIZE, shuffle=False)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"age_output": "mae", "gender_output": "binary_crossentropy"},
        loss_weights={"age_output": 0.5, "gender_output": 1.0},
        metrics={"age_output": "mae", "gender_output": "accuracy"},
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT_PATH, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_OUT_PATH)
    print(f"Saved trained model to {MODEL_OUT_PATH}")

    # Quick evaluation summary for your report
    val_loss = model.evaluate(val_ds)
    print("Final validation metrics:", dict(zip(model.metrics_names, val_loss)))


if __name__ == "__main__":
    main()
