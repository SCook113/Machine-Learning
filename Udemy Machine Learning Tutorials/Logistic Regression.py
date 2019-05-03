import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Read in data
advertisting_data = pd.read_csv('advertising.csv')

# Create a histogram of Age
advertisting_data['Age'].plot.hist(bins=35)
plt.show()

# Explore relationship between age and area income
sns.jointplot(x='Age', y='Area Income', data=advertisting_data)
plt.show()

# Explore relationship between age and daily time spent on site
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=advertisting_data)
plt.show()

# Explore relationship between daily time spent on site and daily internet usage
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=advertisting_data)
plt.show()


# Logistic Regression

# Seperate data in input and output values
X = advertisting_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = advertisting_data['Clicked on Ad']


# Split data to test- and training-data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Instantiate a instance of a logistic regression model
logmodel = LogisticRegression()

# Train the model with the training data
logmodel.fit(X_train,y_train)

# Feed test data to the trained model and make predictions
predictions = logmodel.predict(X_test)

# Look at out results
print(classification_report(y_test,predictions))