# =========================================
# IMPORTS
# =========================================

from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
import joblib
import requests
import os
import cv2

from PIL import Image

# =========================================
# FLASK SETUP
# =========================================

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================================
# LOAD MODELS
# =========================================

soil_model = tf.keras.models.load_model(
    "models/soil_model.keras",
    compile=False
)

crop_model = joblib.load(
    "models/crop_xgb_model.pkl"
)

print("✅ Models Loaded Successfully")

# =========================================
# SOIL CLASSES
# =========================================

soil_classes = [
    "Black_Soil",
    "Clayey_Soil",
    "Loamy_Soil",
    "Red_Soil",
    "Sandy_Soil"
]

# =========================================
# SOIL ENCODING
# =========================================

soil_encoding = {
    "Black_Soil": 0,
    "Clayey_Soil": 1,
    "Loamy_Soil": 2,
    "Red_Soil": 3,
    "Sandy_Soil": 4
}

# =========================================
# CROP LABELS
# =========================================

crop_labels = {
    0: "Barley",
    1: "Cotton",
    2: "Ground Nuts",
    3: "Maize",
    4: "Millets",
    5: "Oil seeds",
    6: "Paddy",
    7: "Pulses",
    8: "Sugarcane",
    9: "Tobacco",
    10: "Wheat",
    11: "Jowar",
    12: "Bajra",
    13: "Gram",
    14: "Soyabean",
    15: "Sunflower",
    16: "Safflower",
    17: "Linseed",
    18: "Castor seed",
    19: "Coriander",
    20: "Ragi",
    21: "Arhar",
    22: "Urad",
    23: "Moong",
    24: "Horsegram",
    25: "Cowpea",
    26: "Sesamum",
    27: "Niger seed",
    28: "Sweet potato",
    29: "Tapioca",
    30: "Moth",
    31: "Guar seed",
    32: "Coconut",
    33: "Cashewnut",
    34: "Masoor",
    35: "Peas & beans",
    36: "Potato",
    37: "Onion",
    38: "Garlic",
    39: "Ginger",
    40: "Turmeric",
    41: "Banana",
    42: "Rapeseed",
    43: "Jute",
    44: "Arecanut"
}

# =========================================
# SMART NPK
# =========================================

def get_smart_npk(soil_type, humidity, region):

    base_npk = {
        "Black_Soil": [80, 40, 40],
        "Red_Soil": [40, 30, 30],
        "Sandy_Soil": [20, 20, 20],
        "Loamy_Soil": [60, 50, 50],
        "Clayey_Soil": [70, 60, 60]
    }

    N, P, K = base_npk.get(
        soil_type,
        [50, 40, 40]
    )

    # humidity effect
    if humidity > 70:
        N -= 10
        K -= 5

    elif humidity < 40:
        N -= 5
        P -= 5

    # region effect
    if region == 0:
        N += 5

    elif region == 3:
        N -= 10
        P -= 5

    # avoid negatives
    N = max(N, 10)
    P = max(P, 10)
    K = max(K, 10)

    return N, P, K

# =========================================
# REGION DETECTOR
# =========================================

def get_region(lat):

    if lat < 15:
        return 0

    elif lat < 22:
        return 3

    elif lat < 28:
        return 2

    else:
        return 1

# =========================================
# HOME ROUTE
# =========================================

@app.route("/")

def home():

    return jsonify({
        "message": "Rythu Mitra Backend Running"
    })

# =========================================
# PREDICT ROUTE
# =========================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # =================================
        # CHECK IMAGE
        # =================================

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            })

        file = request.files["image"]
        # =================================
        # GET LOCATION FROM MOBILE APP
        # =================================
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")
        if not latitude or not longitude:
            return jsonify({
                "error":
                "Location permission required. Please enable location and try again."
            }), 400

        lat = float(latitude)
        lon = float(longitude)

        # =================================
        # IMAGE PROCESSING
        # =================================

        image = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (224, 224)
        )

        image = image.astype("float32") / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        # =================================
        # SOIL PREDICTION
        # =================================

        prediction = soil_model.predict(image)

        soil_index = np.argmax(prediction)

        soil_type = soil_classes[soil_index]

        soil_confidence = float(
            np.max(prediction) * 100
        )
        # =================================
        # CHECK IF IMAGE IS REALLY SOIL
        # =================================
        if soil_confidence < 75:
            return jsonify({
                "error":
                "This image does not appear to be soil. Please capture a clear soil image and try again.",

                "soil_confidence":
                round(soil_confidence, 2)

            }), 400

        # =================================
        # WEATHER API
        # =================================

        API_KEY = "a3ddda72a1824497fdbdbd6ed51932e5"

        try:

            weather_url = (
                f"https://api.openweathermap.org/data/2.5/weather?"
                f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            )

            weather_res = requests.get(
                weather_url,
                timeout=5
            )

            weather_data = weather_res.json()

            temperature = weather_data["main"]["temp"]

            humidity = weather_data["main"]["humidity"]

        except Exception as e:
            return jsonify({
                "error":
                "Unable to fetch weather data",

                "details":
                str(e)

            }), 400

        # =================================
        # REGION
        # =================================

        region = get_region(lat)

        # =================================
        # MOISTURE
        # =================================

        if humidity > 70:
            moisture = 70

        elif humidity > 50:
            moisture = 50

        else:
            moisture = 30

        # =================================
        # SOIL ENCODE
        # =================================

        soil_encoded = soil_encoding[soil_type]

        # =================================
        # SMART NPK
        # =================================

        N, P, K = get_smart_npk(
            soil_type,
            humidity,
            region
        )

        # =================================
        # CROP INPUT
        # =================================

        crop_input = [[
            temperature,
            humidity,
            moisture,
            soil_encoded,
            region
        ]]

        # =================================
        # CROP PREDICTION
        # =================================

        probabilities = crop_model.predict_proba(
            crop_input
        )[0]

        # =================================
        # RULE FILTERING
        # =================================

        for i, crop in crop_labels.items():

            if soil_type == "Sandy_Soil" and crop == "Paddy":
                probabilities[i] = 0

            if soil_type == "Red_Soil" and crop == "Paddy":
                probabilities[i] = 0

            if moisture < 40 and crop == "Sugarcane":
                probabilities[i] = 0

            if humidity < 50 and crop == "Paddy":
                probabilities[i] = 0

        # =================================
        # TOP 3 CROPS
        # =================================

        top3 = probabilities.argsort()[-3:][::-1]

        recommended_crops = []

        for i in top3:

            recommended_crops.append({

                "crop":
                    crop_labels[i],

                "confidence":
                    round(float(probabilities[i] * 100), 2)

            })

        # =================================
        # FINAL RESPONSE
        # =================================

        return jsonify({

            "soil_type":
                soil_type,

            "soil_confidence":
                round(soil_confidence, 2),
            
            "latitude":
                lat,

            "longitude":
                lon,


            "temperature":
                temperature,

            "humidity":
                humidity,

            "recommended_crops":
                recommended_crops

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )