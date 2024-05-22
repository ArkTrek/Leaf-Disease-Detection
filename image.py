import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("./modelFile/plantDisease.h5")

# Define the classes
CLASSES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
           'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
           'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
           'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
           'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
           'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
           'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
           'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
           'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
           'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
           'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
           'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Load an input image from file
image_path = "./Samples/1.JPG"
input_image = cv2.imread(image_path)
input_image = cv2.resize(input_image, (224, 224))

# Convert the image to the format expected by the model
x = np.expand_dims(input_image, axis=0)

# Make a prediction
preds = model.predict(x)
predicted_class = np.argmax(preds)
predicted_prob = preds[0][predicted_class]

# Get the predicted class label
result = CLASSES[predicted_class]

# Check if predicted probability is above the threshold
# Draw a rectangle around the detected item with red color
cv2.rectangle(input_image, (0, 0), (input_image.shape[1], input_image.shape[0]), (0, 0, 255), 2)

# Display the result on the input image with red font color
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_size = cv2.getTextSize(result, font, font_scale, font_thickness)[0]
text_position = ((input_image.shape[1] - text_size[0]) // 2, (input_image.shape[0] + text_size[1]) // 2)
cv2.putText(input_image, result, text_position, font, font_scale, (0, 0, 255), font_thickness)

# Increase the result window size
result_window_size = (input_image.shape[1] + 200, input_image.shape[0] + 200)

# Display the input image with the result
pcv2.imshow('Leaf Disease Detection', cv2.resize(input_image, result_window_size))
cv2.waitKey(0)
cv2.destroyAllWindows()