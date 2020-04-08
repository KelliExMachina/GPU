# Standard
import pandas as pd
import numpy as np

# GPU
from numba import jit, cuda

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Timing
from timeit import default_timer as timer

X_train, X_test, y_train, y_test = train_test_split(df_X_numeric, y, test_size=.2, random_state=42)
print('X_train: {}'.format(len(X_train)))
print('y_train: {}'.format(len(y_train)))
print('X_test: {}'.format(len(X_test)))
print('y_test: {}'.format(len(y_test)))

