import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import warnings
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
######################################################################################################
# Get Data

# Load Data Set
data = pd.read_csv("data/train.csv")
training_labels = data['Cover_Type']
training_data = data.drop(['Cover_Type', 'Id'], axis=1)

# print(data.info())
# print(training_labels.value_counts())
# Every class is represented equally

# # Check for highly correlated features
# correlations = data.corr()
# for column in correlations.columns.values:
#     correlations[column] = correlations[column].apply(lambda x: 1 if x > 0.98 or x < -0.98 else 0)
# print(correlations)
# # No highly correlated values

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
# for learning_rate in learning_rates:
#     gb = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2,
#                                     random_state=0)
#     gb.fit(training_data, training_labels)
#     print("Learning rate: ", learning_rate)
#     print("Accuracy score (training): {0:.3f}".format(gb.score(training_data, training_labels)))
#     print()

# Set the parameters by cross-validation
# tuned_parameters = [{'learning_rate': [0.05, 0.1, 0.25, 0.5, 0.75, 1],
#                      'max_features': [2, 8],
#                      'max_depth': [2],
#                      'random_state': [0]}]
#
# print("# Tuning hyper-parameters for recall")
# print()
#
# clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring='accuracy',verbose=5)
# clf.fit(training_data, training_labels)
#
# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
# print()
#
# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# y_true, y_pred = training_labels, clf.predict(training_data)
# print(classification_report(y_true, y_pred, labels=[1,2,3,4,5,6,7]))
# print()
######################################################################################################
######################################################################################################
# Eliminate unneccessary feature

gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.25, max_features=2, max_depth=2, random_state=0)
gb.fit(training_data, training_labels)
y_true, y_pred = training_labels, gb.predict(training_data)
# print(classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6, 7]))
# Drop columns that contribute less than 1%
rounded = np.round(gb.feature_importances_, decimals=3)
feat_imp = zip(training_data.columns, rounded)
with_percentages = list(map(lambda a: (a[0], a[1] / 1), feat_imp))
sorted = sorted(with_percentages, key=lambda feature: feature[1], reverse=True)
columns_to_delete = list(filter(lambda a: a[1] < 0.01, sorted))
column_names = map(lambda a: a[0], columns_to_delete)
data = training_data.drop(column_names, axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data2 = scaler.fit_transform(data)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


######################################################################################################
######################################################################################################
# Train
X_train, X_test, y_train, y_test = train_test_split(data2, training_labels, stratify=training_labels, test_size=0.25)

percent = list()
for bagging_run in range(0,30):

    # train 3 classifiers
    final_preds = []
    for x in range(0, 3):
        clf = ExtraTreesClassifier(n_estimators =100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        final_preds.append(y_pred)
        # print(classification_report(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6, 7]))

    summary = np.zeros(shape=(len(final_preds[0]), len(final_preds) + 1))


    # Create an array with all predictions in a row
    # with the correct prediction at the end
    # summary = pred pred pred ... correct_y
    # loop through all predictions
    for j in range(len(final_preds[0])):
        # loop through number of predictor models
        for i in range(len(final_preds)):
            summary[j][i] = final_preds[i][j]
        # Append correct pred
        summary[j][summary[0].shape[0] - 1] = y_test.iloc[j]

    # Create voted List
    summed_votes = np.zeros(shape=y_pred.shape)
    for it in range(0,y_pred.shape[0]):
        lll = summary[it][:-1].tolist()
        summed_votes[it] = max(set(lll), key=lll.count)
    # Compare summed votes to last classifiers result
    print(classification_report(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6, 7]))
    print("Accuracy score: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, summed_votes, labels=[1, 2, 3, 4, 5, 6, 7]))
    print("Accuracy score: ", accuracy_score(y_test, summed_votes))


    # Stats
    not_all_same = list()
    for row in summary:
        if row[0] == row[1] == row[2]:
            pass
        else:
            not_all_same.append(row)
    print(len(not_all_same), " of ", len(y_test), " were not classified exactly")

    rows_with_dups = list()
    count_would_correct = 0
    count_would_miss = 0
    # Only prepared to handle 3 classifiers
    if not_all_same[0].shape[0] == 4:
        # durch alle unkorrekten
        for row in not_all_same:
            # duplicate values speichern
            dups = list()
            # make a row to list
            l = row.tolist()
            # iterate over all elements except true val
            for x in np.nditer(row[:-1]):
                # if there is a duplicate in the pred values
                if l[:-1].count(x) > 1:
                    dups.append(x)

            if dups != list():
                # if there are duplicates found and the duplicate is the correct predition
                # increase count of "would be predicted correctly"
                if dups[0] == row[-1]:
                    count_would_correct = count_would_correct + 1
                    # print(count_would_correct)
                # Append to rows with dups array
                rows_with_dups.append(row)
                if dups[0] != row[-1]:
                    count_would_miss = count_would_miss + 1

        # Rows with duplicates to array
        r_w_d = np.zeros(shape=(len(rows_with_dups), 4))
        for i in range(0 , len(rows_with_dups)):
            r_w_d[i] = rows_with_dups[i]
        print('#'*10,"Run: " , bagging_run,'#'*10)
        print("from all ", len(not_all_same)," ambiguous rows, ", len(rows_with_dups), " had duplicates")
        print("from all ",len(rows_with_dups)," rows with duplicates, ", count_would_correct, " would have been corrected")
        print("and " , count_would_miss , " would have been missclassified")
        print("We would have classifieed ", count_would_correct - count_would_miss , " correctly")
        percent.append(len(not_all_same) / (count_would_correct - count_would_miss))
        print("That is: ", len(not_all_same) / (count_would_correct - count_would_miss) , "%")

print("On average thats " , sum(percent) / len(percent))

