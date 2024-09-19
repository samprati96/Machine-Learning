#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:21:39 2024

@author: sampratigawande
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

"""
Think of this as collecting information about different types of flowers. Each flower has measurements like petal length, petal width, etc., and we also know which type of flower it is (like Iris-setosa, Iris-versicolor, or Iris-virginica).
"""

# Load the Iris dataset
#It consists of 150 samples of iris flowers, with four features and three target classes.
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
#Here, 20% of the data is used for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
We adjust the measurements so they are on a similar scale. 
For example, if one measurement is in centimeters and another is in millimeters, we convert them to be in the same unit. This helps the model learn better.
"""

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the neural network
#an instance of MLPClassifier with a simple architecture of two hidden layers, each with 10 neurons
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
#We then feed the training data to this neural network so it can learn the patterns. This is similar to how our brain learns by looking at many examples.
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
#We use a confusion matrix and a classification report to get detailed information on its performance, like how many flowers it classified correctly and where it made mistakes.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

