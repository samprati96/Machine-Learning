#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:32:40 2024

@author: sampratigawande
"""

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


# Generate some dummy data
np.random.seed(0)
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(2, size=1000)  # Binary targetx

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
Scaling is like making sure all your ingredients are in the right proportions when you're baking a cake.

Imagine you're baking a cake, and you have ingredients like flour, sugar, and baking powder. If you use a huge amount of flour but only a tiny pinch of baking powder, it would be difficult to mix them properly, and the cake might not turn out well.

Similarly, in machine learning, your data might have features (ingredients) with different ranges. For example, one feature might be the number of bedrooms in a house (ranging from 1 to 5), and another might be the house price (ranging from thousands to millions). If these features are on very different scales, it can make it harder for the model to learn effectively.

Scaling adjusts these features so that they are in the same range. It’s like making sure all your ingredients are in similar proportions so that they mix well and create a better cake (model).

Advantages of Scaling
Improves Model Performance:

Scaling helps the model learn more efficiently. When all features are on a similar scale, the model can adjust its weights more effectively during training.
Speeds Up Training:

When features are scaled, the optimization process (the process of finding the best weights for the model) becomes faster. This is because the model can make more consistent updates during training.
Prevents Dominance of Large Scale Features:

Without scaling, features with larger ranges can dominate the learning process, making it harder for the model to learn from features with smaller ranges. Scaling ensures all features contribute equally.
Improves Model Stability:

Models, especially those involving gradient descent (a common optimization technique), can become more stable and converge faster when the input features are scaled.
"""

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled.shape)
X_test_scaled = scaler.transform(X_test)

# Define a simple neural network with regularization
#Sequential([...]): This creates a new neural network model where you stack layers in a sequence.
"""
Dense(64, input_dim=10, activation='relu', kernel_regularizer=l2(0.01)):

Dense(64, ...): This means the first layer of the network has 64 neurons (think of neurons as units that process information).
input_dim=10: This specifies that the input to this layer will have 10 features (like 10 pieces of data).
activation='relu': This is an activation function called ReLU (Rectified Linear Unit). It helps the network learn complex patterns by adding non-linearity.
kernel_regularizer=l2(0.01): This adds a penalty to the loss function to prevent the model from overfitting by keeping weights small.
"""
model = Sequential([
    Dense(64, input_dim=10, activation='relu', kernel_regularizer=l2(0.01)),  # l2 regularization
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])

"""
model.compile(...): This sets up how the model will learn from the data.

optimizer='adam':

optimizer: Think of this as the method the model uses to adjust its settings (weights) to get better at making predictions.
'adam': This is a popular and efficient method for optimization. It helps the model find the best settings by tweaking them in a smart way.
loss='binary_crossentropy':

loss: This is a measure of how far off the model's predictions are from the actual results. Lower loss means better performance.
'binary_crossentropy': This is a specific type of loss function used for problems where there are only two possible outcomes (like yes/no or true/false). It helps the model understand how to improve its predictions for binary classification tasks.
metrics=['accuracy']:

metrics: These are measures used to evaluate how well the model is performing.
'accuracy': This measures the percentage of correct predictions the model makes. For binary classification, it tells you how often the model's guesses match the actual results.
"""

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
