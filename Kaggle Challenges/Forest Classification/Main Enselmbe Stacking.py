import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import random

# Seed so we always get same results when filling missing values
random.seed()
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
######################################################################################################
# This is an attempt to improve on the Script "Main.py" in this folder and trying some stacking
# with different models.
# A lot of code has been copied from the previous script.
# Some parameters for the models have been taking from this:
#
######################################################################################################

# Get Data
data = pd.read_csv("data/train.csv")
training_labels = data['Cover_Type']
training_data = data.drop(['Cover_Type', 'Id'], axis=1)
# Eliminate unneccessary feature
gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.25, max_features=2, max_depth=2, random_state=0)
gb.fit(training_data, training_labels)
y_true, y_pred = training_labels, gb.predict(training_data)
# Drop columns that contribute less than 1%
rounded = np.round(gb.feature_importances_, decimals=3)
feat_imp = zip(training_data.columns, rounded)
with_percentages = list(map(lambda a: (a[0], a[1] / 1), feat_imp))
sorted = sorted(with_percentages, key=lambda feature: feature[1], reverse=True)
columns_to_delete = list(filter(lambda a: a[1] < 0.01, sorted))
column_names = map(lambda a: a[0], columns_to_delete)
data = training_data.drop(column_names, axis=1)
scaler = StandardScaler()
data2 = scaler.fit_transform(data)

######################################################################################################
######################################################################################################
# Train
X_train, X_test, y_train, y_test = train_test_split(data2, training_labels, stratify=training_labels, test_size=0.25)

# Xtratrees
single_preds = []
for run in range(0, 1):
    print("Run ", run + 1, "of ExtraTrees")
    clf = ExtraTreesClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    single_preds.append(y_pred)
    summary = np.zeros(shape=(len(single_preds[0]), len(single_preds)))
    # loop through all predictions
    for j in range(len(single_preds[0])):
        # loop through number of predictor models
        for i in range(len(single_preds)):
            summary[j][i] = single_preds[i][j]
final_preds_xtratrees = []
for i in range(0, len(summary)):
    final_preds_xtratrees = np.append(final_preds_xtratrees, Counter(summary[i].tolist()).most_common(1)[0][0])
# print(final_preds_xtratrees)


print(classification_report(y_test, final_preds_xtratrees, labels=[1, 2, 3, 4, 5, 6, 7]))

# GradientBoostingClassifier
single_preds = []
for run in range(0, 1):
    print("Run ", run + 1, "of GradientBoostingClassifier")
    clf = GradientBoostingClassifier(max_depth=7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    single_preds.append(y_pred)
    summary = np.zeros(shape=(len(single_preds[0]), len(single_preds)))
    # loop through all predictions
    for j in range(len(single_preds[0])):
        # loop through number of predictor models
        for i in range(len(single_preds)):
            summary[j][i] = single_preds[i][j]
final_preds_GradientBoostingClassifier = []
for i in range(0, len(summary)):
    final_preds_GradientBoostingClassifier = np.append(final_preds_GradientBoostingClassifier,
                                                       Counter(summary[i].tolist()).most_common(1)[0][0])
# print(final_preds_knn)
print(classification_report(y_test, final_preds_GradientBoostingClassifier, labels=[1, 2, 3, 4, 5, 6, 7]))

# GradientBoostingClassifier
single_preds = []
for run in range(0, 1):
    print("Run ", run + 1, "of Randomforest")
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    single_preds.append(y_pred)
    summary = np.zeros(shape=(len(single_preds[0]), len(single_preds)))
    # loop through all predictions
    for j in range(len(single_preds[0])):
        # loop through number of predictor models
        for i in range(len(single_preds)):
            summary[j][i] = single_preds[i][j]
final_preds_RandomForest = []
for i in range(0, len(summary)):
    final_preds_RandomForest = np.append(final_preds_RandomForest, Counter(summary[i].tolist()).most_common(1)[0][0])
# print(final_preds_knn)
print(classification_report(y_test, final_preds_RandomForest, labels=[1, 2, 3, 4, 5, 6, 7]))

# all = np.concatenate((final_preds_xtratrees, final_preds_xtratrees), axis=1)
final_preds_xtratrees = final_preds_xtratrees.reshape((-1, 1))
final_preds_RandomForest = final_preds_RandomForest.reshape((-1, 1))
final_preds_GradientBoostingClassifier = final_preds_GradientBoostingClassifier.reshape((-1, 1))
new_training = np.concatenate((final_preds_RandomForest, final_preds_xtratrees, final_preds_GradientBoostingClassifier),
                              axis=1)
