import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(101)
tf.set_random_seed(101)

# # Estimator workflow:
# Create list of feature columns
# Create an estimator
# Train/ Test Split
# Create input functions
# Train
# Evaluate

x_data = np.linspace(0.0, 10.0, 1000000)
# Make data points noisy
noise = np.random.randn(len(x_data))

# This is the form of my function
# y = mx + b
# b = 5
y_true = (0.5 * x_data) + 5 + noise

# Make a list of the feature columns
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
# Initialize a linear regression estimator
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# We now split our data
from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

# Input function 1
input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)
# Input function 2
train_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_train}, y_train, batch_size=8, num_epochs=1000,
                                                      shuffle=False)
# Input function 3
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval, batch_size=8, num_epochs=None, shuffle=True)

estimator.train(input_fn=input_func, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)
print('TRAINING DATA MATRIX')
print(train_metrics)
print('EVAL METRICS')
print(eval_metrics)
