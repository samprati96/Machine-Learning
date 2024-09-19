#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:20:43 2024

@author: sampratigawande
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("/Users/sampratigawande/Documents/ML/spyder_ml_workspace/TumorData/Tumor_classification_CSV.csv") 

y = df['Class']
x = df.drop('Class', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

K = []
train = []
test = []
scores = {}

for k in range(2, 5):
    clf = KNeighborsClassifier(n_neighbors= k)
    clf.fit(x_train, y_train)
    train_score = clf.score(x_train, y_train)
    test_score = clf.score(x_test, y_test)
    K.append(k)
    train.append(train_score)
    test.append(test_score)
    scores[k] = [train_score, test_score]

for key, value in scores.items():
    print(key, " : ", value)
