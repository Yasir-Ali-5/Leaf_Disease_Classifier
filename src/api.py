from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Initialize FastAPI app
app = FastAPI(title="Leaf Disease Classifier")

# Load your model once on startup
model = load_model("models/leaf_disease_color_model.h5")

# Define class labels
class_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___Healthy",
    "Blueberry___Healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___Healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___Healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___Healthy", "Raspberry___Healthy", "Soybean___Healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___Healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy"
]

# API route to test the app
@app.get("/")
def read_root():
    return {"message": "Leaf Disease Detection API is running!"}


# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((128, 128))

        # Convert image to array
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction[0])
        predicted_label = class_labels[predicted_index]

        return JSONResponse({
            "filename": file.filename,
            "predicted_label": predicted_label,
            "confidence": float(np.max(prediction[0]))
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
