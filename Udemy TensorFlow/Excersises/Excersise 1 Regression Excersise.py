import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# The model that is about to be trained is not a good fit and
# this file is for exercise purposes only

data_set = pd.read_csv('data/cal_housing_clean.csv')
# print(data_set.info())
# print(data_set.head(2))

# Preview of data:
#    housingMedianAge  totalRooms  totalBedrooms  population  households  medianIncome  medianHouseValue
# 0              41.0       880.0          129.0       322.0       126.0        8.3252          452600.0
# 1              21.0      7099.0         1106.0      2401.0      1138.0        8.3014          358500.0

# Seperate labels from data
X = data_set.drop(['medianHouseValue'], axis=1)
y = data_set['medianHouseValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Scale the data
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(X_train)
X_train_scaled_pd = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled_pd = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Create input function
housingMedianAge = tf.feature_column.numeric_column('housingMedianAge')
totalRooms = tf.feature_column.numeric_column('totalRooms')
totalBedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
medianIncome = tf.feature_column.numeric_column('medianIncome')
feat_cols = [housingMedianAge, totalRooms, totalBedrooms, population, households, medianIncome]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train_scaled_pd, y=y_train, batch_size=10, num_epochs=1000,
                                                 shuffle=True)

# Initialize a model
dnn_model = tf.estimator.DNNRegressor(hidden_units=[6,6,6], feature_columns=feat_cols)
dnn_model.train(input_fn=input_func, steps=1000)

# Evaluate
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test_scaled_pd, batch_size=10, num_epochs=1000,
                                                         shuffle=False)
pred_gen = dnn_model.predict(predict_input_func)

# Extract predictions:
predictions = list(pred_gen)
# print(predictions)

predictions_list = []
for pred in predictions:
    predictions_list.append(pred['predictions'])

# Calculate mean squared error
#MSE = mean_squared_error(y_test, predictions_list) ** 0.5

print("hi")
