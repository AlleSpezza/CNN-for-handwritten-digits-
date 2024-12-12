import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

# Load the model
model = load_model('mnist_model.h5')

# Upload the image and convert it in a greyscale because the CNN needs in input grayscale images
image = cv.imread("data/7_rotated.png", cv.IMREAD_GRAYSCALE)

# Resize the input image because the CNN needs a 28x28 input images
image_resized = cv.resize(image,(28,28))

# Pixel normalization
img_normalized = image_resized / 255.0

plt.imshow(img_normalized, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.show()

# Add an extra dimension for the batch becuse the keras deep learning model accept in input
# only img with 4 dimensions: (batch_size, height, width, channels)
img_input = img_normalized.reshape(1, 28, 28, 1)

# Predictions
predictions = model.predict(img_input)

# Find the class with the highest probability
predicted_class = np.argmax(predictions)

# Show the result 
print(f"the number shown in the image is: {predicted_class}")

# # Mostra le probabilità per tutte le classi
# print("Probabilità per ciascuna classe:")
# for i, prob in enumerate(predictions[0]):
#     print(f"Classe {i}: {prob:.4f}")






