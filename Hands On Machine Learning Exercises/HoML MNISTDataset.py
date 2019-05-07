# fetch_mldata is deprecated, ignore for now
import warnings
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

mnist = fetch_mldata("MNIST original")
# DESCR: Description, data: data,  target: labels
print('Describe: ', mnist['DESCR'], ' data: ', mnist['data'], ' target: ', mnist['target'])
print(type(mnist))

X, y = mnist["data"], mnist["target"]
# The data consists of 70000 datasets with 784 features
# One picture is 28x28px
print(X.shape)

#################################################################
# Binary Classification
# Let's pick out one digit and plot it
#################################################################
some_digit = X[36000]
print(some_digit)
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
# It looks like a five, let's check the label:
print(y[36000])

#################################################################
# Let's split the set
#################################################################

# Set is already split with the first 60000 being the training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# We shuffle the training set so all numbers are represented in
# the validation folds
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#################################################################
# For now we focus on training a binaray classifier
# We will create a '5' dectector
#################################################################
# Set '5' labels to true
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# We choose a stochastic gradient descent classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

# We train our parameters to the instances of "true" in the labels
# This causes the Classifier to define a function that recognizes 5s
sgd_clf.fit(X_train, y_train_5)

# Now we feed the trained Classifier an example
print(sgd_clf.predict([some_digit]))

#################################################################
# Let's evaluate the models performance
#################################################################

# Test Accuracy
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))


# The model has a accuracy of over 95% wich looks very good

# Let's check if this is a good result:
# This is a classifier that categorizes everything as "not 5"
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
# This classifier also has 90% accuracy => accuracy is no good performance measure
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# We will now use a confusion matrix
# We need some predicted labels to compare them to the original ones
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Let's look at the confusion matrix
# Result is structured like this:
# [[correctly classified as non-fives,  wrongly classified as fives]
# # [ wrongly classifiedas as non-fives,   correctly classified as fives]]
# # In other words:
# # [[true negatives,  false positives]
# # [ false negatives,   true positives]]
print(confusion_matrix(y_train_5, y_train_pred))

# Here we have functions to tell us about the precision and the recall
# How often is it correct when it predictsa a five:
print(precision_score(y_train_5, y_train_pred))
# How many fives does it detect:
print(recall_score(y_train_5, y_train_pred))

# Compute F1 score
f1_score(y_train_5, y_train_pred)

# There is a tradeoff between precision and recall
# and the decision threshhold controls this
# so now we want to find a good threshhold to use:
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# Let's plot these:
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# Plot precision against recall
plt.plot(recalls[:-1], precisions[:-1], "b--", label="Precision")
plt.xlabel("recall")
plt.ylabel("precision")
plt.ylim([0, 1])
plt.xlim([0, 1])
plt.show()

# We play with using a decision treshold of 70000 = 90% precision
y_train_pred_90 = (y_scores > 70000)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# plot_roc_curve(fpr, tpr)
# plt.show()

# Get the roc score
roc_auc_score(y_train_5, y_scores)

# We now train a Random Forest Classifier on our data
# to compare the different models
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

#################################################################
# Multiclass Classification
#################################################################

# Scikitlearn detects that we want to train a linear classiifer
#  for a multiclass problem, and automatically runs OvA
sgd_clf.fit(X_train, y_train)  # y_train, not y_train_5

# # Gives the prediction:
print(sgd_clf.predict([some_digit]))

some_digit_scores = sgd_clf.decision_function([some_digit])
# Gives back the decision scores
print(some_digit_scores)

# Let's evaluate the model
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# We now improve the algorithms performance by scaling the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

#################################################################
# Error analysis
#################################################################

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.xlabel("Predictions")
plt.ylabel("Classes")
plt.show()
