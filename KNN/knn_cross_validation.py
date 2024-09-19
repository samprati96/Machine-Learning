#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 00:30:20 2024

@author: sampratigawande
"""

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv("/Users/sampratigawande/Documents/ML/spyder_ml_workspace/TumorData/Tumor_classification_CSV.csv") 

# Define the features and the target variable
y = df['Class']
x = df.drop('Class', axis=1)

# Initialize lists to store the results
k_cv_score = {}

# Loop through different values of k
for k in range(2, 5):
    clf = KNeighborsClassifier(n_neighbors=k)
    
    # Perform cross-validation
    scores = cross_val_score(clf, x, y, cv=5)  # 5-fold cross-validation
    
    # Store the mean of the cross-validation scores
    k_cv_score[k] = scores.mean()


# Print the results
for k, score in k_cv_score.items():
    print(f'k = {k}, Cross-Validation Score: {score:.4f}')

"""
cross_val_score

Iteration 1: Train on [2, 3, 4, 5], Validate on [1]
Iteration 2: Train on [1, 3, 4, 5], Validate on [2]
Iteration 3: Train on [1, 2, 4, 5], Validate on [3]
Iteration 4: Train on [1, 2, 3, 5], Validate on [4]
Iteration 5: Train on [1, 2, 3, 4], Validate on [5]

"""