#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:56:28 2024

@author: sampratigawande
"""

import numpy as np

def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true - y_predicted)**2)/len(y_true)
    return cost

def gradient_descent(x, y, iteration=100, learning_rate=0.001, threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01
    n = float(len(x))
    
    print(f"n: {n}")

    costs = []
    weights = []
    previous_cost = None
    for i in range(iteration):
        y_predicted = (current_weight * x) + current_bias
        current_cost = mean_squared_error(y, y_predicted)
        
        if previous_cost and abs(previous_cost - current_cost) <= threshold:
            break

        previous_cost = current_cost
        costs.append(current_cost)  # y-axis
        weights.append(current_weight)  # x-axis

        weight_derivative = -(2 / n) * sum(x * (y - y_predicted))
        bias_derivative = -(2 / n) * sum(y - y_predicted)

        current_weight = current_weight - (learning_rate * weight_derivative)  # GD formula
        current_bias = current_bias - (learning_rate * bias_derivative)
        
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
		{current_weight}, Bias {current_bias}")

    return current_weight, current_bias


def main():
    # Sample values
    x = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787, 55.14218841,
                  52.21179669, 39.29956669, 48.10504169, 52.55001444, 45.41973014, 54.35163488,
                  44.1640495, 58.16847072, 56.72720806, 48.95588857, 44.68719623, 60.29732685,
                  45.61864377, 38.81681754])
    y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513, 78.21151827,
                  79.64197305, 59.17148932, 75.3312423, 71.30087989, 55.16567715, 82.47884676,
                  62.00892325, 75.39287043, 81.43619216, 60.72360244, 82.89250373, 97.37989686,
                  48.84715332, 56.87721319])

    current_weight, current_bias = gradient_descent(x, y, iteration=1000, learning_rate=0.0001, threshold=1e-6)

    print(current_weight, current_bias)

    # once bias and weight is done, then do the predictions

    y_pred = current_weight * x + current_bias
    print(f"y_pred : {y_pred}")

if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        