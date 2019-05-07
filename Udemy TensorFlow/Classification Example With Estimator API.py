import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

np.random.seed(101)
tf.set_random_seed(101)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################
# Get data
################################
diabetes = pd.read_csv('data/pima-indians-diabetes.csv')
print(diabetes.head(10))
# print(diabetes.columns)

# Dataset:
#######################################################
#    Number_pregnant  Glucose_concentration  Blood_pressure   Triceps   Insulin       BMI  Pedigree  Age  Class Group
# 0                6               0.743719        0.590164  0.353535  0.000000  0.500745  0.234415   50      1     B
# 1                1               0.427136        0.540984  0.292929  0.000000  0.396423  0.116567   31      0     C
# 2                8               0.919598        0.524590  0.000000  0.000000  0.347243  0.253629   32      1     B


################################
# Wrangle data
################################
# Normalise numeric values
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
                'Insulin', 'BMI', 'Pedigree']
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Now we create all these feature columns in tf
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Handle categorical features
# Vocab list:
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
# With a hashbucket
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

# diabetes['Age'].hist(bins=4)
# plt.show()

# Convert the 'Age' column to a bucketized column
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

# List of all columns
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, assigned_group, age_buckets]

################################
# Create training / test data
################################

# Seperate input from output
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']

# Create a train test split
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.33, random_state=101)

################################
# Train a model
################################

# Input functions are functions that return a tf.data.Dataset object
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

# Instantiate a model and give it our feature columns
# Linear Classifier
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

# Train model
model.train(input_fn=input_func, steps=1000)

# Evaluation function
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
# Get results
results = model.evaluate(eval_input_func)
# Predict something
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(pred_input_func)
print(list(predictions))

# Dense neural network classifier
embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)
feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, bmi, diabetes_pedigree, embedded_group_column,
             age_buckets]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)

# Evaluate
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
eval = dnn_model.evaluate(eval_input_func)

print(eval)
