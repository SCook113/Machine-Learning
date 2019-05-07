import tensorflow as tf
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.INFO)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
import warnings

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
random.seed(57)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
########################################################################
# In this script I got a data set from kaggle.com and tried out different things
########################################################################

data = pd.read_csv('data/heart.csv')
# print(data['target'].value_counts())
# numeric_data.loc[(numeric_data['age'] > 20) & (numeric_data['age'] <= 30), 'age'] = 2

# Get some info
# print(data.head(50))
# print(data.info())
# print(data.describe())

#######################################
# Wrangle the data
#######################################
# Convert everything to float values
numeric_data = data.astype(dtype='float64')
numeric_data['target'] = numeric_data['target'].apply(int)
# Normalize columns: trestbps, chol, thalach
column_list = {'trestbps', 'chol', 'thalach'}
min_max_scaler = MinMaxScaler(copy=False)
numeric_data[['trestbps', 'chol', 'thalach']] = min_max_scaler.fit_transform(
    numeric_data[['trestbps', 'chol', 'thalach']])
# Bin values of age in 5 discrete values
# numeric_data['age_band'] = pd.cut(numeric_data['age'], 6)
# Replace actual age with a mapping value depending on ageband
# numeric_data.loc[numeric_data['age'] < 20, 'age'] = 1
# numeric_data.loc[(numeric_data['age'] > 20) & (numeric_data['age'] <= 30), 'age'] = 2
# numeric_data.loc[(numeric_data['age'] > 30) & (numeric_data['age'] <= 40), 'age'] = 3
# numeric_data.loc[(numeric_data['age'] > 40) & (numeric_data['age'] <= 50), 'age'] = 4
# numeric_data.loc[(numeric_data['age'] > 50) & (numeric_data['age'] <= 60), 'age'] = 5
# numeric_data.loc[(numeric_data['age'] > 60) & (numeric_data['age'] <= 70), 'age'] = 6
# numeric_data.loc[(numeric_data['age'] > 70) & (numeric_data['age'] <= 80), 'age'] = 7
# Remove ageband feature
# train_df = train_df.drop(['ageBand'], axis=1)
# print(numeric_data.head(30))
numeric_data.drop('sex', axis=1)
numeric_data.drop('age', axis=1)

y = numeric_data['target']

X = numeric_data.drop('target', axis=1)
print(numeric_data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.30)
print(X.info())

########################################################################
# Here I was playing around with the estimator API
########################################################################
# Feature columns for estimator
feat_cols = []
for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
accuracy_scores = []
for numeration in range(0, 10):
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=30, num_epochs=150, shuffle=True)
    ADAM = tf.train.AdamOptimizer(learning_rate=0.01)
    he_init = tf.contrib.layers.variance_scaling_initializer()
    classifier = tf.estimator.DNNClassifier(hidden_units=[40, 2], n_classes=2, feature_columns=feat_cols,
                                            batch_norm=True)

    classifier.train(input_fn=input_func, steps=100000)

    pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)

    predictions = list(classifier.predict(input_fn=pred_fn))

    final_preds = []
    for pred in predictions:
        final_preds.append(pred['class_ids'][0])
    # print(classification_report(y_test, final_preds))
    acc = accuracy_score(y_test, final_preds)
    accuracy_scores.append(acc)
    print("Accuracy: ", accuracy_score(y_test, final_preds))
print("all: ", accuracy_scores)
print("avg: ", sum(accuracy_scores) / len(accuracy_scores))
