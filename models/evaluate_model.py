import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import math

# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/age_gender_cnn.keras"

METADATA_PATH = "models/test_metadata.csv"

IMAGE_SIZE = (128, 128)

# Gender labels:
# 0 = Male
# 1 = Female


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD TEST METADATA
# ==========================================

df = pd.read_csv(METADATA_PATH)

print("Metadata loaded successfully!")

print("\nTotal test images:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# VARIABLES
# ==========================================

actual_ages = []
predicted_ages = []

actual_genders = []
predicted_genders = []


# ==========================================
# PREDICTION
# ==========================================

for index, row in df.iterrows():

    try:

        # --------------------------------------
        # IMAGE PATH FROM CSV
        # --------------------------------------

        image_path = row["image_path"]

        # --------------------------------------
        # ACTUAL AGE FROM CSV
        # --------------------------------------

        actual_age = float(row["age"])

        # --------------------------------------
        # ACTUAL GENDER FROM CSV
        # --------------------------------------

        actual_gender = int(row["gender_label"])

        # --------------------------------------
        # CHECK IMAGE
        # --------------------------------------

        if not os.path.exists(image_path):

            print(
                f"Image not found: {image_path}"
            )

            continue

        # --------------------------------------
        # LOAD IMAGE
        # --------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE
        )

        # --------------------------------------
        # PREPROCESS
        # --------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # --------------------------------------
        # MODEL PREDICTION
        # --------------------------------------

        age_output, gender_output = model.predict(
            image_array,
            verbose=0
        )

        # --------------------------------------
        # AGE PREDICTION
        # --------------------------------------

        predicted_age = float(
            age_output[0][0]
        )

        # --------------------------------------
        # GENDER PREDICTION
        # --------------------------------------

        gender_score = float(
            gender_output[0][0]
        )

        if gender_score >= 0.5:

            predicted_gender = 1

        else:

            predicted_gender = 0

        # --------------------------------------
        # SAVE RESULTS
        # --------------------------------------

        actual_ages.append(
            actual_age
        )

        predicted_ages.append(
            predicted_age
        )

        actual_genders.append(
            actual_gender
        )

        predicted_genders.append(
            predicted_gender
        )

    except Exception as e:

        print(
            f"Error processing row {index}: {e}"
        )


# ==========================================
# CHECK RESULTS
# ==========================================

if len(actual_ages) == 0:

    print("\nNo valid images were processed.")

    exit()


# ==========================================
# AGE METRICS
# ==========================================

actual_ages = np.array(
    actual_ages
)

predicted_ages = np.array(
    predicted_ages
)

age_errors = (
    predicted_ages - actual_ages
)

absolute_errors = np.abs(
    age_errors
)

mae = np.mean(
    absolute_errors
)

rmse = math.sqrt(
    np.mean(
        age_errors ** 2
    )
)


# ==========================================
# GENDER METRICS
# ==========================================

actual_genders = np.array(
    actual_genders
)

predicted_genders = np.array(
    predicted_genders
)

correct_gender = np.sum(
    actual_genders == predicted_genders
)

total_gender = len(
    actual_genders
)

gender_accuracy = (
    correct_gender / total_gender
) * 100


# ==========================================
# FINAL RESULTS
# ==========================================

print("\n")
print("=" * 60)

print(
    "             MODEL EVALUATION"
)

print("=" * 60)


# ==========================================
# AGE
# ==========================================

print("\nAGE MODEL")
print("-" * 60)

print(
    f"Test Images : {len(actual_ages)}"
)

print(
    f"MAE         : {mae:.2f} years"
)

print(
    f"RMSE        : {rmse:.2f} years"
)


# ==========================================
# GENDER
# ==========================================

print("\nGENDER MODEL")
print("-" * 60)

print(
    f"Correct     : {correct_gender}"
)

print(
    f"Total       : {total_gender}"
)

print(
    f"Accuracy    : {gender_accuracy:.2f}%"
)


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("\n")
print("=" * 60)

print(
    "             SAMPLE PREDICTIONS"
)

print("=" * 60)


for i in range(
    min(10, len(actual_ages))
):

    actual_gender_text = (
        "Male"
        if actual_genders[i] == 0
        else "Female"
    )

    predicted_gender_text = (
        "Male"
        if predicted_genders[i] == 0
        else "Female"
    )

    print(
        f"\nImage {i + 1}"
    )

    print(
        f"Actual Age       : {actual_ages[i]:.0f}"
    )

    print(
        f"Predicted Age    : {predicted_ages[i]:.2f}"
    )

    print(
        f"Actual Gender    : {actual_gender_text}"
    )

    print(
        f"Predicted Gender : {predicted_gender_text}"
    )


print("\n")
print("=" * 60)