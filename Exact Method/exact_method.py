#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:12:01 2024

@author: sampratigawande
"""

import pandas as pd 
import numpy as np

x = pd.read_csv("/Users/sampratigawande/Documents/Machine Learning/spyder_ml_workspace/Classwork/HousingData/X.csv") 
y = pd.read_csv("/Users/sampratigawande/Documents/Machine Learning/spyder_ml_workspace/Classwork/HousingData/Y.csv") 

x_in = x.to_numpy()
y_in = y.to_numpy()

print(x_in)

x_in_t = np.transpose(x_in)

xt_x = np.dot(x_in_t , x_in)
xt_y = np.dot(x_in_t , y_in)
inverse = np.linalg.inv(xt_x)

#inverse(x_in_t * x_in) * (x_in_t * y_in)
b = np.dot(inverse,  xt_y)

print(b)