import random as rnd

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import random

# Seed so we always get same results when filling missing values
random.seed(57)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#######################################
# Tutorial from:
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
#######################################

# Info about the data frame we will extract as 'train_df'
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# Name           891 non-null object
# Sex            891 non-null object
# Age            714 non-null float64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# Ticket         891 non-null object
# Fare           891 non-null float64
# Cabin          204 non-null object
# Embarked       889 non-null object
#
# Sample of train_df.head(5)
#    PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
# 0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
# 2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
# 3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
# 4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S
#

# Loading the data
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

#######################################
# Analyse the data by describing it
#######################################

# # Note feature names for manipulating them later on
# print(train_df.columns.values)
#
# # Which categories are available
# print(train_df.columns.values)

#
# What does the data look like
# #
# # We want to answer following questions:
# # Which features are categorical?
# # Which features are numerical?
# # Which features are mixed data types?
# # Which features may contain errors or typos?
# # Which features contain blank, null or empty values?
# # What are the data types for various features?
# print(train_df.head(50))
# # What are the data types for various features?
# train_df.info() # 891 entries
# print('_'*40)
# test_df.info() # 418 entries

# # What is the distribution of numerical feature values across the samples?
# print(train_df.describe())

# What is the distribution of categorical features?
# list of dtypes to include
# include =['object', 'float', 'int']
# print(train_df.describe(include=['O']))

#######################################
# Testing assumptions about the data
#######################################

# # What is the percentage of people that survived depending on their Pclass
# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# # What is the percentage of people that survived depending on their sex
# print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# # What is the percentage of people with a certain number of siblings that survived
# print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# # What is the percentage of people with a certain number of parents/children that survived
# print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Parch', ascending=False))

# # Plot the age distribution of survived and not survived
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()
# ----> Consider 'Age' as a feature

# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()
# ----> Consider 'Pclass' as a feature

# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# plt.show()
# ----> Consider 'Sex' as a feature

# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()
# ----> Consider 'Sex' as a feature

#######################################
# Wrangle the data
#######################################

# We decide to drop the feature 'Ticket' and 'Cabin'
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

# We extract the titles from names and create a new column
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# # Check survival rates based on titles
# print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# We now replace the titles with integers for computation
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# We map values in column 'sex' to integer values for computation
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Age feature correction
########################

# We fill in the missing age values with the mean of ages depending on pclass
guess_ages = np.zeros((2, 3))

# Iterate through data sets
for dataset in combine:
    # Iterate through possible values for Sex (0 or 1)
    for i in range(0, 2):
        # Iterate through possible values for Pclass (1 2 3)
        for j in range(0, 3):
            # Drop missing observations where age is missing
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()
            # Values in range of standard deviation
            age_mean = guess_df.mean()
            age_std = guess_df.std()
            age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    # Fill in the calculated values
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_ages[
                i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# Age
########################
# Create age bands and determine  correlations with survived

# Bin values of age in 5 discrete values
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
# Replace actual Age with a mapping value depending on ageband
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']
# Remove ageband feature
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# Parch and SibSp
# Combine Parch and SibSp to a separate feature in order to drop both features
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create feature isalone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Drop the features Parch, SibSp and FamilySize for feature 'IsAlone'
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# Create artificial feature combining pclass and age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# Correcting embarked feature

# Impute mode for missing values
# Get mode
freq_port = train_df.Embarked.dropna().mode()[0]
# Impute
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Convert to numeric values
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Correcting Fare feature

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

#######################################
# Model, predict and solve
#######################################

# First we need to choose a model to train
# We are trying to solve a regression/classification problem

# Prepare test and train data
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Logistic Regression model
#######################################

# We train a logistic regression model on the data
# Initialise Model
logreg = LogisticRegression()
# Fit model
logreg.fit(X_train, Y_train)
# Get predictions on test data
Y_pred = logreg.predict(X_test)
# Get the mean accuracy on the given test data and labels.
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# # Check every feature for its coefficient value in the
# # logistic regression model to see which feature impacts the
# # outcome the most
# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
# coeff_df.sort_values(by='Correlation', ascending=False)
# print(logreg.coef_)

# SVM
#######################################
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# K-Nearest neighbours
#######################################
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
#######################################
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Linear SVC
#######################################
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
#######################################
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
#######################################
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
#######################################
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#######################################
# Evaluate the models
#######################################
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))
