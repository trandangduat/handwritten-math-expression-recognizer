import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import thin
from skimage.util import invert

math_exp = [ '!', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'X', 'cos', 'div', 'i', 'j', 'k', 'log', 'pi', 'sin', 'sqrt', 'tan', 'times', 'u', 'v', 'y', 'z' ];

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((45, 45))
    img_array = np.array(img) / 255.0
    img_array = invert(img_array)
    img_array = img_array > 0.5 # Convert every pixel to either 0/1 (thresholding)
    img_array = thin(img_array) # Make the thickness of the pattern 1 pixel (skeletonize)
    img_array = img_array.reshape((1, 45, 45, 1))
    return invert(img_array)

# Load the model
model = load_model("math_expression_recognizer.keras")

# Load and preprocess test image
test_image = preprocess_image("test.jpg")

# Make a prediction on a test image
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions)
print(predictions)

# Display the image and prediction
plt.imshow(test_image.reshape(45, 45), cmap='gray')
plt.title(f"Predicted: {math_exp[predicted_class]}")
plt.show()
