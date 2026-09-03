"""
Local CPU training script for Age & Gender model.
Optimized for faster CPU training:
  - Uses a subset of the dataset (5000 images) for faster convergence
  - Reduced batch size for CPU memory
  - 20 epochs with early stopping
"""

import os
import sys
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from model_architecture import build_model, IMG_SIZE

# ------------------------------------------------------------------
# 1. Download the UTKFace dataset via kagglehub
# ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Downloading UTKFace dataset...")
print("=" * 60)

import kagglehub
path = kagglehub.dataset_download("jangedoo/utkface-new")
print(f"Dataset downloaded to: {path}")

# Find the folder with actual images
data_dir = path
for root, dirs, files in os.walk(path):
    jpg_count = sum(1 for f in files if f.lower().endswith('.jpg'))
    if jpg_count > 1000:
        data_dir = root
        break
print(f"Using image folder: {data_dir}")

# ------------------------------------------------------------------
# 2. Load labels from filenames
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Loading labels from filenames...")
print("=" * 60)

FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)_")

filepaths, ages, genders = [], [], []
for fname in os.listdir(data_dir):
    match = FILENAME_PATTERN.match(fname)
    if not match:
        continue
    age, gender = int(match.group(1)), int(match.group(2))
    if age > 100:
        continue
    filepaths.append(os.path.join(data_dir, fname))
    ages.append(age)
    genders.append(gender)

ages = np.array(ages, dtype=np.float32)
genders = np.array(genders, dtype=np.float32)
print(f"Found {len(filepaths)} total labeled images.")

# Use a subset for faster CPU training (5000 images is enough for decent results)
MAX_SAMPLES = 5000
if len(filepaths) > MAX_SAMPLES:
    indices = np.random.RandomState(42).choice(len(filepaths), MAX_SAMPLES, replace=False)
    filepaths = [filepaths[i] for i in indices]
    ages = ages[indices]
    genders = genders[indices]
    print(f"Using subset of {MAX_SAMPLES} images for CPU training.")

# ------------------------------------------------------------------
# 3. Train/val split
# ------------------------------------------------------------------
train_fp, val_fp, train_age, val_age, train_gender, val_gender = train_test_split(
    filepaths, ages, genders, test_size=0.15, random_state=42
)
print(f"Training samples: {len(train_fp)}, Validation samples: {len(val_fp)}")

# ------------------------------------------------------------------
# 4. tf.data pipeline
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Building data pipeline...")
print("=" * 60)

def _decode_and_preprocess(filepath, age, gender):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, {"age_output": age, "gender_output": gender}

BATCH_SIZE = 32  # smaller batch for CPU

def make_dataset(fps, a, g, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((fps, a, g))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(fps))
    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_fp, train_age, train_gender, BATCH_SIZE, shuffle=True)
val_ds = make_dataset(val_fp, val_age, val_gender, BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------------
# 5. Build, compile, train
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Building and compiling model...")
print("=" * 60)

MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "model", "age_gender_model.h5")

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

print("\n" + "=" * 60)
print("STEP 5: Training started! (This will take ~30-60 minutes on CPU)")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
)

model.save(MODEL_OUT_PATH)
print(f"\n{'=' * 60}")
print(f"DONE! Saved trained model to {MODEL_OUT_PATH}")
print(f"{'=' * 60}")

# Quick evaluation
val_results = model.evaluate(val_ds)
print("\nFinal validation metrics:")
for name, value in zip(model.metrics_names, val_results):
    print(f"  {name}: {value:.4f}")

print("\n✅ Training complete! You can now run the app:")
print("   .\\venv\\Scripts\\streamlit.exe run app/app.py")
