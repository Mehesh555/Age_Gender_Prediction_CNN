import os
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from werkzeug.utils import secure_filename


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "age_gender_cnn.keras"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

print(
    "Input shape:",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)

print(
    "Output names:",
    model.output_names
)

print("=" * 60)


# ============================================================
# ALLOWED FILE
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# AGE GROUP
# ============================================================

def get_age_group(age):

    age = int(
        round(age)
    )

    if age <= 4:
        return "0 - 4"

    elif age <= 9:
        return "5 - 9"

    elif age <= 14:
        return "10 - 14"

    elif age <= 19:
        return "15 - 19"

    elif age <= 24:
        return "20 - 24"

    elif age <= 29:
        return "25 - 29"

    elif age <= 34:
        return "30 - 34"

    elif age <= 39:
        return "35 - 39"

    elif age <= 44:
        return "40 - 44"

    elif age <= 49:
        return "45 - 49"

    elif age <= 54:
        return "50 - 54"

    elif age <= 59:
        return "55 - 59"

    else:
        return "60+"


# ============================================================
# GET DATASET GENDER LABEL
# ============================================================

# ============================================================
# GET GENDER FROM MODEL OUTPUT
# ============================================================

def get_gender_label(gender_score):

    if gender_score is None:
        return "Unknown"

    if gender_score < 0.5:
        return "Male"

    else:
        return "Female"


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    # IMPORTANT:
    # Your trained model expects 128 x 128

    image = image.resize(
        (128, 128)
    )


    image_array = np.array(
        image,
        dtype=np.float32
    )


    # Normalize

    image_array = (
        image_array / 255.0
    )


    # Add batch dimension

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return image_array


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image_path):

    image_array = preprocess_image(
        image_path
    )


    prediction = model.predict(
        image_array,
        verbose=0
    )


    print("=" * 60)

    print("RAW MODEL OUTPUT:")

    print(prediction)

    print("=" * 60)


    # ========================================================
    # AGE OUTPUT
    # ========================================================

    age_output = None


    if isinstance(
        prediction,
        list
    ):

        for name, output in zip(
            model.output_names,
            prediction
        ):

            if name == "age_output":

                age_output = output

                break


    if age_output is None:

        age_output = prediction[0]


    predicted_age = float(
        np.asarray(
            age_output
        ).reshape(-1)[0]
    )


    # Keep age between 0 and 116

    predicted_age = max(
        0,
        min(
            116,
            predicted_age
        )
    )


    rounded_age = int(
        round(
            predicted_age
        )
    )


    age_group = get_age_group(
        rounded_age
    )


    # ========================================================
    # SECOND MODEL OUTPUT
    # ========================================================

    second_output = None


    if isinstance(
        prediction,
        list
    ):

        for name, output in zip(
            model.output_names,
            prediction
        ):

            if name == "gender_output":

                second_output = output

                break


    if second_output is None:

        second_output = prediction[1]


    raw_second_output = float(
        np.asarray(
            second_output
        ).reshape(-1)[0]
    )

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    gender = get_gender_label(
        raw_second_output
    )


    return {

        "estimated_age":
            round(
                predicted_age,
                2
            ),

        "rounded_age":
            rounded_age,

        "age_group":
            age_group,

        "gender":
            gender

    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "Please select an image."

            }), 400


        file = request.files[
            "image"
        ]


        if file.filename == "":

            return jsonify({

                "success": False,

                "message":
                    "No image selected."

            }), 400


        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "success": False,

                "message":
                    "Only JPG, JPEG and PNG images are supported."

            }), 400


        # ----------------------------------------------------
        # KEEP ORIGINAL DATASET FILENAME
        # ----------------------------------------------------

        original_filename = secure_filename(
            file.filename
        )


        print(
            "Original filename:",
            original_filename
        )



        # ----------------------------------------------------
        # EXTENSION
        # ----------------------------------------------------

        extension = (
            original_filename
            .rsplit(
                ".",
                1
            )[1]
            .lower()
        )


        # ----------------------------------------------------
        # UNIQUE UPLOAD FILENAME
        # ----------------------------------------------------

        unique_filename = (
            f"{uuid.uuid4().hex}.{extension}"
        )


        file_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )


        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        file.save(
            file_path
        )


        # ----------------------------------------------------
        # PREDICT AGE
        # ----------------------------------------------------

        result = predict_image(
            file_path
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "image_url":
                f"/uploads/{unique_filename}",

            "estimated_age":
                result["estimated_age"],

            "rounded_age":
                result["rounded_age"],

            "age_group":
                result["age_group"],

            "gender":
                result["gender"]

        })


    except Exception as e:

        print("=" * 60)

        print("ERROR:")

        print(e)

        print("=" * 60)


        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# SERVE UPLOADS
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(
    filename
):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )