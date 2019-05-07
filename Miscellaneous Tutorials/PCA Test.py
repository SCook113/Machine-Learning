# Load in our libraries
import warnings

import pandas as pd

warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

##############################################
# In this script I wanted to test how doing a PCA on the data before training it
# may affect the models accuracy
#######################
for dimension in range(3, 12):
    data = pd.read_csv('data/heart.csv')
    pca = PCA(n_components=dimension)
    # Info
    # print(data.head())
    # print(data.info())
    mean = list()
    X = data.drop(['target'], axis=1)
    X = pca.fit_transform(X)
    y = data['target']
    for iteration in range(0, 500):
        # We create a holdout set for testing in the end
        X_train, testing_X_test, y_train, testing_y_test = train_test_split(X, y, test_size=0.10)

        ###################################################
        # SVC
        ###################################################
        svc = SVC(kernel='linear', C=0.25)

        svc.fit(X_train, y_train)

        svc_predictions = svc.predict(testing_X_test)
        mean.append(accuracy_score(testing_y_test, svc_predictions))
        # print("SVC accuracy: ",accuracy_score(testing_y_test, svc_predictions))
    print(dimension, " dimensional, mean: ", sum(mean) / len(mean))
