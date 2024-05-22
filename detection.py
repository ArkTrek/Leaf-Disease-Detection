import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
import matplotlib.pyplot as plt
from io import BytesIO
import os
from PIL import Image, ImageDraw

model = tf.keras.models.load_model("./modelFile/plantDisease.h5")
CLASSES = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy',
'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot',
'Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

# Load and preprocess the image
img_path = "./Samples/3.JPG"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Make predictions
preds = model.predict(x)
predicted_class = np.argmax(preds)
predicted_label = CLASSES[predicted_class]

# Display the image with label and rectangle
img = Image.open(img_path)
draw = ImageDraw.Draw(img)

# Add predicted label
draw.text((10, 10), f'Predicted Class: {predicted_label}', fill='red')

# Add rectangle around the predicted region (you might need to adjust these coordinates based on your needs)
rectangle_coords = [(50, 50), (150, 150)]  # Example coordinates, adjust as needed
draw.rectangle(rectangle_coords, outline='red', width=2)

# Display the image
img.show()