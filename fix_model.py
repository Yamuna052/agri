from tensorflow.keras.models import load_model

# load old model
model = load_model("backend/models/soil_model.keras", compile=False)

# re-save in new compatible format
model.save("backend/models/soil_model_fixed.keras")

print("Model re-saved successfully!")
