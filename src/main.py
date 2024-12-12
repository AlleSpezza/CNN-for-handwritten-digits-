import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Activation

# –––––––– 1. Upload the dataset and understand how it is composed ––––––––

# Upload MNIST dataset
# x_train = includes the images used for training 
# x_train = includes the labels (numbers images form 0 to 9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Print dataset dimensions
print(f"Training set: {x_train.shape}, {y_train.shape}")
print(f"Test set: {x_test.shape}, {y_test.shape}")

# Analyze the train labels to verify the balance of the dataset classes
train_class_distribution = pd.Series(y_train).value_counts().sort_index()
print("Class distribution in the traning set: ",train_class_distribution)

# Analyze the test labels to verify the balance of the dataset classes
test_class_distribution = pd.Series(y_test).value_counts().sort_index()
print("Class distribution in the traning set: ",test_class_distribution)

# Show a database image to get an idea about the dataset 
# plt.imshow(x_train[0], cmap='gray')
# plt.title(f"Label: {y_train[0]}")
# plt.show()

# To identify the mean intensity value of the pixels 
mean_pixel_value = np.mean(x_train)
print(f"Mean pixel values for the training set: {mean_pixel_value}")

# To understand if the intensity values of the pixels changes a lot with respect the mean value 
std_pixel_value = np.std(x_train)
print(f"Std pixel values for the training set: {std_pixel_value}")

# –––––––– 2. Preprocessing phase –––––––

# Normalize the train and the test set 
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Check if the normalization is done in a good way
# print(f"Min value for train set: {x_train_normalized.min()}")
# print(f"Max value for train set: {x_train_normalized.max()}")
# print(f"Min value for test set: {x_test_normalized.min()}")
# print(f"Max value for test set: {x_test_normalized.max()}")

# Reshape to add a channel dimension (28, 28, 1) for CNN input
x_train_normalized = x_train_normalized.reshape(-1, 28, 28, 1)
x_test_normalized = x_test_normalized.reshape(-1, 28, 28, 1)

# One Hot Encoding, because "categorical_crossentropy" loss function required that the labels are represented in an one hot encoding way
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Check if the one hot encoding is done in a good way
# print(f"Original label: {y_train[0]}")
# print(f"One-hot encoded: {y_train_one_hot[0]}")

# Convert data in tf.float32
x_train_normalized = tf.convert_to_tensor(x_train_normalized, dtype=tf.float32)
x_test_normalized = tf.convert_to_tensor(x_test_normalized, dtype=tf.float32)
y_train_one_hot = tf.convert_to_tensor(y_train_one_hot, dtype=tf.float32)
y_test_one_hot = tf.convert_to_tensor(y_test_one_hot, dtype=tf.float32)

# Data augmentation to improve the quality of the dataset 
datagen = ImageDataGenerator(
    rotation_range=30,         # Randomly rotate images by up to 10 degrees
    width_shift_range=0.1,     # Randomly shift images horizontally
    height_shift_range=0.1,    # Randomly shift images vertically
    zoom_range=0.1             # Randomly zoom images
)

train_generator = datagen.flow(
    x_train_normalized,
    y_train_one_hot,
    batch_size=32
)


# –––––––– 3. Model architecture –––––––

# CNN model creation
model = Sequential()

# Convulutional layer to detected image features (edges, texture, angles, ecc..)
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())

# Polling layer to reduce the dimension of the feature map
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout to avoid the overfitting. It turns off casually a specified percentage of neurons during the training
model.add(Dropout(0.25))

# Flatten layer to convert the 2D feature maps in a 1D vector 
model.add(Flatten())

# Fully Connected Layers 
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())

# Dropout to avoid the overfitting
model.add(Dropout(0.5))

# Predict the probability that the number in the image belongs to a specific class.
# 10 because we have 10 classes (from 0 to 9)
# softmax to produce a probability for each class
model.add(Dense(10, activation='softmax'))  

# Compile the model
model.compile(optimizer='adam', # adam = useful for classifications problems 
              loss='categorical_crossentropy', # categorical_crossentropy = useful for multi-class classifications problems 
              metrics=['accuracy']) # we are interested in the accuracy 

# Model summary
# model.summary()

# –––––––– 4. Model training  –––––––

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with augmented data
history = model.fit(
    train_generator,  # Use data generator
    validation_data=(x_test_normalized, y_test_one_hot),  # Validation set remains static
    epochs=15, 
    callbacks=[early_stopping]
)

# Model evaluation
test_loss, test_accuracy = model.evaluate(x_test_normalized, y_test_one_hot)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# –––––––– 5. Results visualization  –––––––

# Accuracy plot
# Accuracy on training set for each epoch
plt.plot(history.history['accuracy'], label='Train Accuracy')
# Accuracy on validation set for each epoch
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
# Loss on training set for each epoch
plt.plot(history.history['loss'], label='Train Loss')
# Loss on validation set for each epoch
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# –––––––– 6. Saving the model –––––––

model.save('mnist_model.h5')
print("Model saved successfully!")









