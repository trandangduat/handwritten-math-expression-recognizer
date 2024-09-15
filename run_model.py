import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.morphology import thin
from skimage.util import invert
import cv2

math_exp = [ '!', '(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'X', 'cos', 'div', 'i', 'j', 'k', 'log', 'pi', 'sin', 'sqrt', 'tan', 'times', 'u', 'v', 'y', 'z' ];

def preprocess_image(img): # preprocess image with single expression
    img = Image.fromarray(img) # convert cv2 to PIL
    img = img.convert('L')
    img = img.resize((45, 45))
    img_array = np.array(img) / 255.0
    img_array = invert(img_array)
    img_array = img_array > 0.5 # Convert every pixel to either 0/1 (thresholding)
    img_array = thin(img_array) # Make the thickness of the pattern 1 pixel (skeletonize)
    img_array = img_array.reshape((1, 45, 45, 1))
    return invert(img_array)

def show_images(images):
    fig = plt.figure()
    n = len(images)
    for i in range(n):
        fig.add_subplot(1, n, i+1)
        plt.imshow(images[i], cmap='gray')
    plt.show()
    return 0

def extract_expressions(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundingBoxes = []
    expressions = []

    # Get bounding boxes of contours
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        boundingBoxes.append((x,y,w,h))

    boundingBoxes = sorted(boundingBoxes, key=lambda x: x[0]) # sort by x position

    # Find all the cases where equal (=) splits into two minus (-) and merge them
    for i in range(1, len(boundingBoxes)):
        x, y, w, h = boundingBoxes[i]
        prev_x, prev_y, prev_w, prev_h = boundingBoxes[i-1]
        if (x < prev_x + prev_w // 2):
            boundingBoxes[i-1] = (prev_x, min(y, prev_y), x + w - prev_x, max(y + h, prev_y + prev_h) - min(y, prev_y))
            boundingBoxes[i] = (0, 0, 0, 0)

    for box in boundingBoxes:
        x,y,w,h = box
        if (w == 0 or h == 0):
            continue
        cropped_expression = image[y:y+h, x:x+w]
        # Padding the cropped image to make it square
        max_dim = max(w, h)
        expression = np.ones((max_dim, max_dim, 3), np.uint8) * 255 # create white bg with size max_dim
        expression[(max_dim-h)//2:(max_dim-h)//2+h, (max_dim-w)//2:(max_dim-w)//2+w] = cropped_expression # Put the cropped image in the center of the white bg
        expressions.append(expression)

    show_images(expressions)
    return expressions


def main():
    # Load the model
    model = load_model("google_colab_model/math_expression_recognizer.keras")

    image = cv2.imread('test2.png')
    expressions = extract_expressions(image)

    for expr in expressions:
        # Load and preprocess test image
        test_image = preprocess_image(expr)

        # Make a prediction on a test image
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions)
        print(predictions)

        # Display the image and prediction
        plt.imshow(test_image.reshape(45, 45), cmap='gray')
        plt.title(f"Predicted: {math_exp[predicted_class]}")
        plt.show()

main()
