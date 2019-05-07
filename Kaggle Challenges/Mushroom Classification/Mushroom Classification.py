import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 4000)

##########################################################################################
# Found this tutorial at:
# https://www.kaggle.com/nirajvermafcb/comparing-various-ml-models-roc-curve-comparison/notebook
##########################################################################################

data = pd.read_csv("data/mushrooms.csv")
# Look at the data set
# print(data.head())

# Check for null values
# print(data.isnull().sum())

# What classification classes do we have
# print(data['class'].unique())

# Since we only have string values we
# make integers out of them
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

# print(data.head())
# print(data.groupby('class').size())

X = data.iloc[:, 1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only

# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
################################################################
# Logistic Regression (Default)
################################################################
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
y_prob = model_LR.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print(model_LR.score(X_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

auc_roc = metrics.roc_auc_score(y_test, y_pred)
print(auc_roc)

################################################################
# Logistic Regression (Tuned)
################################################################
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

LR_model = LogisticRegression()

tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'penalty': ['l1', 'l2']
                    }
from sklearn.model_selection import GridSearchCV

LR = GridSearchCV(LR_model, tuned_parameters, cv=10)
LR.fit(X_train, y_train)
print(LR.best_params_)

y_prob = LR.predict_proba(X_test)[:, 1]  # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0)  # This will threshold the probabilities to give class predictions.
print(LR.score(X_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)
auc_roc = metrics.roc_auc_score(y_test, y_pred)
print(auc_roc)
