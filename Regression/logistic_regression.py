#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:59:23 2024

@author: sampratigawande
"""

import numpy as np

#The sigmoid function is used to map predictions to probabilities.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Step 3: Define the Logistic Regression Class

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = np.zeros(1)

        # Gradient descent
        for i in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

#Step 4: Load and Prepare the Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Create synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Step 5: Train and Evaluate the Model

# Initialize and train the model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.accuracy(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


"""
Logistic Regression is a type of mathematical model used for predicting the likelihood of an event happening. Think of it as a smart yes-or-no question machine.

Here's an analogy:
Imagine you are organizing a picnic and you want to decide whether to carry an umbrella or not. You might look at several factors like the weather forecast, cloud cover, humidity, and wind speed. Based on these factors, you decide whether it's likely to rain (yes) or not (no).

Logistic Regression works similarly:

Inputs (Factors): It takes several input factors (like weather conditions).
Processing: It processes these inputs to calculate the probability of an event happening (like rain).
Output (Decision): It then gives you a probability score between 0 and 1. If the probability is higher than a certain threshold (usually 0.5), it predicts 'yes' (rain). If it's lower, it predicts 'no' (no rain).
Key Points:
Prediction: It helps predict the likelihood of a particular outcome.
Binary Outcomes: It's typically used for situations where there are two possible outcomes (yes/no, true/false, rain/no rain).
Probability: It gives a score that represents how likely it is for the event to occur.
So, Logistic Regression is like having a decision-making assistant that looks at various factors, processes them, and then tells you how likely it is that a certain event will happen. If the likelihood is high, it suggests 'yes', and if it's low, it suggests 'no'.
"""
