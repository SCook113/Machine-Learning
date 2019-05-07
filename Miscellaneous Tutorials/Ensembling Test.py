# Load in our libraries
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.model_selection import KFold

data = pd.read_csv('data/heart.csv')
# Info
# print(data.head())
# print(data.info())
data = data

preds_after_all_runs = np.zeros([31, ], dtype=int)

X = data.drop(['target'], axis=1)
y = data['target']
# We create a holdout set for testing in the end
X_train, testing_X_test, y_train, testing_y_test = train_test_split(X, y, test_size=0.10)

for iteration in range(0, 4):
    kf_X_train, kf_X_test, kf_y_train, kf_y_test = train_test_split(X_train, y_train, test_size=0.20)
    kf_X_train = kf_X_train.reset_index(drop=True)
    kf_y_train = kf_y_train.reset_index(drop=True)

    kf = KFold(n_splits=3)

    ###################################################
    # First level classifier 1 (RandomForestClassifier)
    ###################################################
    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                 max_depth=3, min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0, n_estimators=3000,
                                 oob_score=False, verbose=0, warm_start=False)

    x_train = kf_X_train.values

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = kf_y_train[train_index]
        # Train on fold
        rfc.fit(x_tr, y_tr)

    rfc_predictions = rfc.predict(kf_X_test)
    from sklearn.metrics import accuracy_score

    print(accuracy_score(kf_y_test, rfc_predictions))
    # Evaluation
    print("RFC")
    print(confusion_matrix(kf_y_test, rfc_predictions))
    print(classification_report(kf_y_test, rfc_predictions))

    ###################################################
    # First level classifier 2 1 (SVC)
    ###################################################
    svc_X_train, svc_X_test, svc_y_train, svc_y_test = train_test_split(X, y, test_size=0.20)
    svc_X_train = svc_X_train.reset_index(drop=True)
    svc_y_train = svc_y_train.reset_index(drop=True)
    svc = SVC(kernel='linear', C=0.025)
    x_train = svc_X_train.values
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = svc_y_train[train_index]
        # Train on fold
        svc.fit(x_tr, y_tr)
    svc_predictions = svc.predict(svc_X_test)
    print("SVC accuracy: ", accuracy_score(svc_y_test, svc_predictions))

    # # Evaluation
    # print("SVC")
    # print(confusion_matrix(svc_y_test, svc_predictions))
    # print(classification_report(svc_y_test, svc_predictions))

    ###################################################
    # First level classifier 2 2  (SVC)
    ###################################################
    svc2_X_train, svc2_X_test, svc2_y_train, svc2_y_test = train_test_split(X, y, test_size=0.20)
    svc2_X_train = svc2_X_train.reset_index(drop=True)
    svc2_y_train = svc2_y_train.reset_index(drop=True)
    svc2 = SVC(kernel='linear', C=0.025)
    x_train = svc2_X_train.values
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = svc2_y_train[train_index]
        # Train on fold
        svc2.fit(x_tr, y_tr)
    svc2_predictions = svc2.predict(svc2_X_test)
    print("SVC2 accuracy: ", accuracy_score(svc2_y_test, svc2_predictions))

    # # Evaluation
    # print("SVC2")
    # print(confusion_matrix(svc2_y_test, svc2_predictions))
    # print(classification_report(svc2_y_test, svc2_predictions))

    ###################################################
    # First level classifier 2 3  (SVC)
    ###################################################
    svc3_X_train, svc3_X_test, svc3_y_train, svc3_y_test = train_test_split(X, y, test_size=0.20)
    svc3_X_train = svc3_X_train.reset_index(drop=True)
    svc3_y_train = svc3_y_train.reset_index(drop=True)
    svc3 = SVC(kernel='linear', C=0.025)
    x_train = svc3_X_train.values
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = svc3_y_train[train_index]
        # Train on fold
        svc3.fit(x_tr, y_tr)
    svc3_predictions = svc3.predict(svc3_X_test)
    print("SVC3 accuracy: ", accuracy_score(svc3_y_test, svc3_predictions))

    # # Evaluation
    # print("SVC3")
    # print(confusion_matrix(svc3_y_test, svc3_predictions))
    # print(classification_report(svc3_y_test, svc3_predictions))

    ###################################################
    # First level classifier 3 (Adaboost)
    ###################################################
    ada_X_train, ada_X_test, ada_y_train, ada_y_test = train_test_split(X, y, test_size=0.20)
    ada_X_train = ada_X_train.reset_index(drop=True)
    ada_y_train = ada_y_train.reset_index(drop=True)
    ada = AdaBoostClassifier(n_estimators=1000, learning_rate=0.025)
    x_train = ada_X_train.values
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = ada_y_train[train_index]
        # Train on fold
        ada.fit(x_tr, y_tr)
    ada_predictions = ada.predict(ada_X_test)
    print("Ada accuracy: ", accuracy_score(ada_y_test, ada_predictions))

    # # Evaluation
    # print("ada")
    # print(confusion_matrix(ada_y_test, ada_predictions))
    # print(classification_report(ada_y_test, ada_predictions))

    ###################################################
    # First level classifier 4 (Extra Trees)
    ###################################################
    extra_trees_X_train, extra_trees_X_test, extra_trees_y_train, extra_trees_y_test = train_test_split(X, y,
                                                                                                        test_size=0.20)
    extra_trees_X_train = extra_trees_X_train.reset_index(drop=True)
    extra_trees_y_train = extra_trees_y_train.reset_index(drop=True)
    extra_trees = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, max_depth=5, min_samples_leaf=2)

    x_train = extra_trees_X_train.values
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = extra_trees_y_train[train_index]
        # Train on fold
        extra_trees.fit(x_tr, y_tr)
    extra_trees_predictions = extra_trees.predict(extra_trees_X_test)

    # Evaluation
    # print("extra_trees")
    # print(confusion_matrix(extra_trees_y_test, extra_trees_predictions))
    # print(classification_report(extra_trees_y_test, extra_trees_predictions))

    ###################################################
    # First level classifier 5 (GaussianNB)
    ###################################################
    NB_X_train, NB_X_test, NB_y_train, NB_y_test = train_test_split(X, y, test_size=0.20)
    NB_X_train = NB_X_train.reset_index(drop=True)
    NB_y_train = NB_y_train.reset_index(drop=True)
    NB = GaussianNB()

    x_train = NB_X_train.values
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # All training indexes for a fold
        x_tr = x_train[train_index]
        # All prediction  indexes for a fold
        y_tr = NB_y_train[train_index]
        # Train on fold
        NB.fit(x_tr, y_tr)
    NB_predictions = NB.predict(NB_X_test)
    # # Evaluation
    # print("NB")
    # print(confusion_matrix(NB_y_test, NB_predictions))
    # print(classification_report(NB_y_test, NB_predictions))

    #############################################################
    # Test data on the holdout set we generated in the beginning
    #############################################################
    NB_predictions_final = NB.predict(testing_X_test)
    extra_trees_final = extra_trees.predict(testing_X_test)
    ada_predictions_final = ada.predict(testing_X_test)
    svc_predictions_final = svc.predict(testing_X_test)
    svc2_predictions_final = svc2.predict(testing_X_test)
    svc3_predictions_final = svc3.predict(testing_X_test)
    rfc_predictions_final = rfc.predict(testing_X_test)
    # Evaluation
    # NB_predictions_final
    all = np.column_stack((extra_trees_final, ada_predictions_final, svc_predictions_final, rfc_predictions_final,
                           svc2_predictions_final, svc3_predictions_final))
    # Set all predictions to "1" if mean of all predictions is above 0.5
    mean = all.mean(axis=1)
    end_results = (mean > 0.5).astype(int)
    if iteration == 0:
        preds_after_all_runs = end_results
    else:
        preds_after_all_runs = np.column_stack((preds_after_all_runs, end_results))
    print("Iteration: ", iteration)
    # print("in loop", preds_after_all_runs)
    # print(classification_report(end_results, testing_y_test))

# print(preds_after_all_runs)
mean = preds_after_all_runs.mean(axis=1)
mean_end_results = (mean > 0.5).astype(int)
print(preds_after_all_runs)
print("End")
print(confusion_matrix(mean_end_results, testing_y_test))
print(classification_report(mean_end_results, testing_y_test))
print("with preds: ")
with_preds = np.column_stack((preds_after_all_runs, testing_y_test))
print(with_preds)
