import os
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 42


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "utkface_metadata.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "age_gender_cnn.keras"
)


# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD METADATA
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print()
print("Total records:", len(df))

print()
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 4. CLEAN DATA
# ============================================================

# Remove missing values
df = df.dropna(
    subset=[
        "image_path",
        "age",
        "gender"
    ]
).copy()


# Make sure age is numeric
df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

df = df.dropna(
    subset=["age"]
)


# ============================================================
# 5. CONVERT GENDER TO NUMERIC
# ============================================================

# Male   = 0
# Female = 1

df["gender_label"] = df["gender"].map({
    "Male": 0,
    "Female": 1
})


# Remove unknown gender values
df = df.dropna(
    subset=["gender_label"]
).copy()


df["gender_label"] = df["gender_label"].astype(
    np.float32
)


# Age as float
df["age"] = df["age"].astype(
    np.float32
)


print()
print("Clean records:", len(df))

print()
print("Gender distribution:")
print(df["gender"].value_counts())


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First:
# 80% training
# 20% temporary

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df["gender"]
)


# From temporary:
# 10% validation
# 10% test

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["gender"]
)


print()
print("Dataset split:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))


# ============================================================
# 7. IMAGE LOADING FUNCTION
# ============================================================

def load_image(image_path, age, gender):

    # Read image
    image = tf.io.read_file(image_path)

    # Decode JPG/JPEG/PNG
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Resize
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    # Convert pixel values:
    # 0-255 → 0-1
    image = tf.cast(
        image,
        tf.float32
    ) / 255.0


    # Outputs
    targets = {
        "age_output": age,
        "gender_output": gender
    }

    return image, targets


# ============================================================
# 8. CREATE TF.DATA DATASET
# ============================================================

def create_dataset(dataframe, shuffle=False):

    image_paths = dataframe["image_path"].values

    ages = dataframe["age"].values.astype(
        np.float32
    )

    genders = dataframe["gender_label"].values.astype(
        np.float32
    )


    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            ages,
            genders
        )
    )


    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=2000,
            seed=RANDOM_STATE
        )


    dataset = dataset.batch(
        BATCH_SIZE
    )


    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return dataset


# ============================================================
# 9. CREATE DATASETS
# ============================================================

print()
print("Preparing TensorFlow datasets...")


train_dataset = create_dataset(
    train_df,
    shuffle=True
)


val_dataset = create_dataset(
    val_df,
    shuffle=False
)


test_dataset = create_dataset(
    test_df,
    shuffle=False
)


print("TensorFlow datasets ready.")


# ============================================================
# 10. BUILD CNN MODEL
# ============================================================

print()
print("=" * 60)
print("BUILDING CNN MODEL")
print("=" * 60)


inputs = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    name="image_input"
)


# ------------------------------------------------------------
# CNN BLOCK 1
# ------------------------------------------------------------

x = layers.Conv2D(
    32,
    (3, 3),
    padding="same",
    activation="relu"
)(inputs)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)


# ------------------------------------------------------------
# CNN BLOCK 2
# ------------------------------------------------------------

x = layers.Conv2D(
    64,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)


# ------------------------------------------------------------
# CNN BLOCK 3
# ------------------------------------------------------------

x = layers.Conv2D(
    128,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)


# ------------------------------------------------------------
# CNN BLOCK 4
# ------------------------------------------------------------

x = layers.Conv2D(
    256,
    (3, 3),
    padding="same",
    activation="relu"
)(x)

x = layers.BatchNormalization()(x)

x = layers.MaxPooling2D(
    (2, 2)
)(x)


# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(
    0.4
)(x)


# ============================================================
# 11. AGE OUTPUT
# ============================================================

age_branch = layers.Dense(
    128,
    activation="relu"
)(x)

age_output = layers.Dense(
    1,
    activation="linear",
    name="age_output"
)(age_branch)


# ============================================================
# 12. GENDER OUTPUT
# ============================================================

gender_branch = layers.Dense(
    128,
    activation="relu"
)(x)

gender_output = layers.Dense(
    1,
    activation="sigmoid",
    name="gender_output"
)(gender_branch)


# ============================================================
# 13. CREATE MODEL
# ============================================================

model = models.Model(
    inputs=inputs,
    outputs=[
        age_output,
        gender_output
    ]
)


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss={
        "age_output": "mae",
        "gender_output": "binary_crossentropy"
    },

    loss_weights={
        "age_output": 1.0,
        "gender_output": 1.0
    },

    metrics={
        "age_output": [
            tf.keras.metrics.MeanAbsoluteError(
                name="age_mae"
            )
        ],

        "gender_output": [
            tf.keras.metrics.BinaryAccuracy(
                name="gender_accuracy"
            )
        ]
    }
)


# ============================================================
# 15. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 16. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001,

    verbose=1
)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("STARTING MODEL TRAINING")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]

)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print()
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print()
print("Model saved to:")

print(MODEL_PATH)


# ============================================================
# 19. EVALUATE MODEL
# ============================================================

print()
print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)


results = model.evaluate(
    test_dataset,
    return_dict=True
)


print()

for key, value in results.items():

    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# 20. SAVE TEST DATA
# ============================================================

test_csv_path = os.path.join(
    MODEL_DIR,
    "test_metadata.csv"
)

test_df.to_csv(
    test_csv_path,
    index=False
)


print()
print("Test metadata saved to:")

print(test_csv_path)


print()
print("Training pipeline completed successfully!")