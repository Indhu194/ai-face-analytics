"""
Age & Gender Estimation - Model Architecture
==============================================
A single CNN backbone with two output heads:
  1. Gender head  -> binary classification (Male / Female), sigmoid output
  2. Age head     -> regression (predicts a single number: the age), linear output

Why one shared backbone instead of two separate models?
  - Faster to train and much smaller in size.
  - Age and gender are both learned from the same facial features
    (bone structure, skin texture, etc.), so sharing the backbone
    works well in practice and is the standard approach used in
    most published age/gender CNN papers (e.g. Levi & Hassner, 2015).

Input : 128 x 128 x 3 RGB face crop
Output: {"age_output": float, "gender_output": 0/1 probability}
"""

from tensorflow.keras import layers, models

IMG_SIZE = 128  # width = height of the input face crop


def build_model(img_size: int = IMG_SIZE):
    """Builds and returns the multi-output age+gender CNN."""

    inputs = layers.Input(shape=(img_size, img_size, 3), name="face_input")

    # ---------------- Shared Convolutional Backbone ----------------
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # ---------------- Gender Head (classification) ----------------
    gender_branch = layers.Dense(64, activation="relu")(x)
    gender_branch = layers.Dropout(0.3)(gender_branch)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(gender_branch)

    # ---------------- Age Head (regression) ----------------
    age_branch = layers.Dense(64, activation="relu")(x)
    age_branch = layers.Dropout(0.3)(age_branch)
    age_output = layers.Dense(1, activation="linear", name="age_output")(age_branch)

    model = models.Model(inputs=inputs, outputs=[age_output, gender_output], name="age_gender_cnn")
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
