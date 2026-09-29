import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/age_gender_cnn.keras"

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# IMAGE
# ==========================================

image_path = input("\nEnter image path: ")

image = Image.open(image_path).convert("RGB")

# Model expects 128 x 128
image = image.resize((128, 128))

image_array = np.array(
    image,
    dtype=np.float32
)

image_array = image_array / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)

print("\nInput shape:", image_array.shape)

# ==========================================
# PREDICTION
# ==========================================

age_output, gender_output = model.predict(
    image_array,
    verbose=0
)

# ==========================================
# AGE
# ==========================================

predicted_age = float(
    age_output[0][0]
)

# ==========================================
# AGE GROUP
# ==========================================

if predicted_age <= 4:
    age_group = "0-4"

elif predicted_age <= 9:
    age_group = "5-9"

elif predicted_age <= 14:
    age_group = "10-14"

elif predicted_age <= 19:
    age_group = "15-19"

elif predicted_age <= 24:
    age_group = "20-24"

elif predicted_age <= 29:
    age_group = "25-29"

elif predicted_age <= 34:
    age_group = "30-34"

elif predicted_age <= 39:
    age_group = "35-39"

elif predicted_age <= 44:
    age_group = "40-44"

elif predicted_age <= 49:
    age_group = "45-49"

elif predicted_age <= 54:
    age_group = "50-54"

elif predicted_age <= 59:
    age_group = "55-59"

else:
    age_group = "60+"

# ==========================================
# GENDER SCORE
# ==========================================

gender_score = float(
    gender_output[0][0]
)

# Assuming your training labels are:
# 0 = Male
# 1 = Female

female_probability = gender_score * 100

male_probability = (
    1 - gender_score
) * 100

if gender_score >= 0.5:
    predicted_gender = "Female"
else:
    predicted_gender = "Male"

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n")
print("=" * 50)
print("             PREDICTION RESULT")
print("=" * 50)

print(
    f"Estimated Age : {predicted_age:.2f} years"
)

print(
    f"Rounded Age   : {round(predicted_age)} years"
)

print(
    f"Age Group     : {age_group}"
)

print("\nGender Prediction")
print("-" * 50)

print(
    f"Male          : {male_probability:.2f}%"
)

print(
    f"Female        : {female_probability:.2f}%"
)

print("-" * 50)

print(
    f"Predicted Gender : {predicted_gender}"
)

print("=" * 50)