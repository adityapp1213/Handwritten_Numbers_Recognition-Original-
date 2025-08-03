# Handwritten_Numbers_Recognition-Original
uses tensors(Keras), OpenCV, and MNIST dataset.  to predict the number feeded into the model 
This project implements a simple neural network using TensorFlow to recognize handwritten digits from the MNIST dataset. Additionally, it can predict digits from custom .png images stored in the digits/ directory.

bash
pip install numpy opencv-python tensorflow matplotlib
Project Structure:
project/
|-- digits/
│   |- digit1.png
│   |-digit2.png
│   | ...
|- model.py
|- README.md
File: model.py
This file is the main script that trains and evaluates the neural network model.

Step-by-step Explanation
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
Imports all required libraries:

os: for file path operations

cv2: to read image files

numpy: for matrix operations

tensorflow: for creating the neural network

matplotlib: for displaying predicted images

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")
train_new_model = True
Prints a welcome message and defines whether to train a new model or load an existing one.

Loading and Preprocessing Data:

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)
MNIST dataset is loaded.

All image values are normalized between 0 and 1 along axis 1 (pixel-wise normalization).

Creating the Model:

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
A sequential model is created.

First layer flattens the 28x28 pixel image into a 784-length vector.

Two hidden dense layers with 128 neurons each and ReLU activation.

Output layer with 10 neurons (digits 0–9) and softmax activation.

Compiling and Training:

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3)
The model is compiled with the Adam optimizer.

Loss function: sparse_categorical_crossentropy (since labels are integers, not one-hot).

Accuracy is tracked as a metric.

The model is trained for 3 epochs.

Evaluating and Saving:

val_loss, val_acc = model.evaluate(X_test, y_test)
print(val_loss)
print(val_acc)
model.save('handwritten_digits.model')
The model is evaluated on the test set.

Loss and accuracy are printed.

The model is saved to handwritten_digits.model.

Loading Saved Model and Predicting from Custom Images:

model = tf.keras.models.load_model('handwritten_digits.model')
This section runs if training is skipped and loads the saved model.

Predicting Custom Images:

image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    ...
Loads each custom image from the digits/ folder (digit1.png, digit2.png, ...).

Converts to grayscale, inverts the image (if black background), and reshapes it.

Feeds it into the model for prediction.

Prints the predicted number and shows the image.

"Notes":
Make sure your custom digit images are 28x28 pixels and in the digits/ folder.

The model expects grayscale images with white digits on a black background.

Inversion helps match the MNIST format if your custom images are the opposite.

With the default MNIST dataset and this architecture, you can expect ~97% accuracy.

Results may vary slightly depending on random initializations and system conditions.
