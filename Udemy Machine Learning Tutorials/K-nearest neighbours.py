import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Read in data
data = pd.read_csv('data/Classified Data', index_col=0)

# print(data.head())

# Standardize to a similair scale so larger values won't have bigger effect than smaller ones
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
#Fit the scaler to everything except target class
scaler.fit(data.drop('TARGET CLASS', axis=1))

#
scaled_features = scaler.transform(data.drop('TARGET CLASS',axis=1))
# print(scaled_features)

# Make a data frame of the scaled features and remove 'TARGET CLASS'
data_feat = pd.DataFrame(scaled_features, columns=data.columns[:-1] )
# print(data_feat.head())


X = scaled_features
y = data['TARGET CLASS']
# Split data
X_train, X_test, y_train, y_test = train_test_split(data_feat,data['TARGET CLASS'],test_size=0.30)

# Instantiate KNeighborsClassifier Object
knn = KNeighborsClassifier(n_neighbors=1)

# Fit the model
knn.fit(X_train,y_train)

# Predict
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

err_rate = []

# Iterate many models using many different k-values and plot their error-rate and see which hast the lowest error-rate
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    # Add mean of average error_rate
    err_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), err_rate, color='blue', linestyle='dashed', marker='o')
plt.show()

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# NOW WITH K=23
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))