#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:49:05 2024

@author: sampratigawande
"""

import numpy as np

#Activation: We use a function (sigmoid) to squash this sum into a range between 0 and 1, giving us a probability.
#The sigmoid function is used to map predictions to probabilities.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class Perceptron:
    #Inputs: Each student has two scores (e.g., Math and English). These scores are the inputs to our network.
    #Weights: We assign a weight to each score to determine its importance in predicting the outcome.
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = sigmoid(linear_output)
                
                # Calculate the error
                error = y[i] - y_pred
                
                # Update weights and bias
                self.weights += self.learning_rate * error * X[i] * sigmoid_derivative(y_pred)
                self.bias += self.learning_rate * error * sigmoid_derivative(y_pred)
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_output)
        return [1 if p > 0.5 else 0 for p in y_pred]


# Create synthetic dataset
np.random.seed(42)
X = np.random.rand(1000, 2)
y = np.array([[1 if x1 + x2 > 1 else 0] for x1, x2 in X]).flatten()


#Train and Evaluate the Model

# Initialize and train the perceptron
input_size = X.shape[1]
perceptron = Perceptron(input_size)
#Training: We adjust the weights based on how wrong our predictions are so that the network gets better over time.
perceptron.train(X, y)

# Make predictions
predictions = perceptron.predict(X)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f'Accuracy: {accuracy * 100:.2f}%')



"""
Forward Propagation

Imagine you have a task of guessing the type of fruit based on some characteristics like color, size, and weight. Here's how forward propagation works:

Input Layer:

You start with the characteristics of the fruit (color, size, weight). These characteristics are like pieces of information that you feed into the system.
Hidden Layers:

Inside the system, there are several layers (like steps in a process) that help process this information. Think of these layers as a series of workers who refine and transform the information step by step.
Each worker (neuron) in a layer does a simple calculation and then passes the result to the next layer.
Output Layer:

The final worker (neuron) in the last layer gives you an output, which is the system's guess about the type of fruit (e.g., apple, banana, or orange).
So, in forward propagation, you feed the input data (characteristics of the fruit) through the network, and it processes the data step by step to produce an output (the guess).
"""


"""
Back-Propagation

Now, let's say you find out whether the guess was right or wrong. If it's wrong, you need to improve the system's guessing ability. Here's how back-propagation helps:

Calculate Error:

You compare the system's guess (output) to the actual type of fruit. This comparison gives you an error (a measure of how wrong the guess was).
Send Error Backwards:

You then send this error backwards through the network, layer by layer, to figure out how each worker (neuron) contributed to the error.
Adjust Weights:

Each worker (neuron) has settings (weights) that determine how it processes information. Based on the error, you adjust these settings slightly to reduce the error for the next time.
Think of it like giving feedback to each worker on how to do their job better, so they make fewer mistakes in the future.
Repeat:

This process of adjusting settings happens many times with many examples, gradually making the system more accurate in its guesses.
So, in back-propagation, the system learns from its mistakes by adjusting how each worker (neuron) processes information, improving its guessing ability over time.

"""
