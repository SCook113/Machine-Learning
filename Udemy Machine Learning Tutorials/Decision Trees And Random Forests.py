import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv('data/kyphosis.csv')

# Get some info
print(df.head())

# Explore relationchip of all features to each other
sns.pairplot(df, hue='Kyphosis', palette='Set1')

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

########################################
# Decision tree classifier
########################################
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# # Lets make our classifier predict on data
# predictions = dtree.predict(X_test)

# print(confusion_matrix(y_test,predictions))
# print(classification_report(y_test,predictions))

########################################
# Random forest
########################################
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))