import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# Load your trained model
model = load_model("models\leaf_disease_color_model.h5")  # Make sure the file is in the same directory or provide full path

# Define class labels (same as before)
class_labels = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___Healthy",
    "Blueberry___Healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___Healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___Healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___Healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy",
    "Raspberry___Healthy",
    "Soybean___Healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___Healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy"
]

# Use file dialog to pick image
Tk().withdraw()  # Hide the root Tk window
img_path = filedialog.askopenfilename(title="Select a leaf image")

if not img_path:
    print("❌ No file selected.")
    exit()

# Load and process image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
print("Raw prediction output:", prediction)
print("Prediction shape:", prediction.shape)

predicted_index = np.argmax(prediction[0])
predicted_label = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown class"

# Show result
plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

print("✅ Predicted Disease:", predicted_label)