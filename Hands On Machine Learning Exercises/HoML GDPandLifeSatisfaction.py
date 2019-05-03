import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import pandas.tools.plotting as ptool
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the data
housing_data = pd.read_csv("datasets/housing.csv")

#################################################
#################################################
# Inspecting data
#################################################
#################################################

# get info about data
print(housing_data.head())
housing_data.info()

# print all where longitude = -122.22
print(housing_data.loc[housing_data['longitude'] != -122.22])
# What different values does 'longitude' column contains and how many of them
print(housing_data['longitude'].value_counts())
# Get a description of each column
print(housing_data.describe())
# Or just of one single column
print(housing_data['median_house_value'].describe())
# plot a histogram for each numerical attribute
housing_data.hist(bins=50, figsize=(20, 15))
plt.show()

#################################################
#################################################
# Creating a Test Set
#################################################
#################################################

# # Splitting the data
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
# Create an income category attribute
housing_data["income_cat"] = np.ceil(housing_data["median_income"] / 1.5)
# # Merge all categories greater than 5 to 5.0
housing_data["income_cat"].where(housing_data["income_cat"] < 5, 5.0, inplace=True)
print(housing_data["income_cat"])
housing_data["income_cat"].hist(bins=50, figsize=(20, 15))
plt.show()

# Stratified sampling based on income category
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

# Check if stratified sampling worked:
# Check proportions in original dataset
# print(housing_data["income_cat"].value_counts() / len(housing_data))
# Check proportions in test dataset
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

# Remove the attribute income_cat and put the data in its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#################################################
#################################################
# Exploring the data
#################################################
#################################################

# Create a copy so original data won't be changed
housing_copy = strat_train_set.copy()

# Visualize the data
# Alpha is set to show where data is the densest
housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()

# Option s = radius of circle represents population
# Option c = price is represented in colors
housing_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                  s=housing_copy["population"] / 100, label="population", figsize=(10, 7),
                  c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, )
plt.legend()
plt.show()

#################################################
#################################################
# Looking for correlations
#################################################
#################################################

# Process every standard correlation coefficient for every attribute
corr_matrix = housing_copy.corr()

# Only plot most promising attributes
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
ptool.scatter_matrix(housing_copy[attributes], figsize=(12, 8))
plt.show()

# Create new attributes
housing_copy["rooms_per_household"] = housing_copy["total_rooms"] / housing_copy["households"]
housing_copy["bedrooms_per_room"] = housing_copy["total_bedrooms"] / housing_copy["total_rooms"]
housing_copy["population_per_household"] = housing_copy["population"] / housing_copy["households"]
corr_matrix = housing_copy.corr()
# Check if the attributes have a high correlation
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Seperate labels from data
housing_copy = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#################################################
#################################################
# Data cleaning
#################################################
#################################################

# In case there is a column with missing fields
# there are three options:
# 1. Get rid of the corresponding lines
housing_copy.dropna(subset=["total_bedrooms"])
# 2. Get rid of the whole attribute
housing_copy.drop("total_bedrooms", axis=1)
# 3. Set the values to some value (zero, the mean, the median, etc.)
median = housing_copy["total_bedrooms"].median()  # option 3
housing_copy["total_bedrooms"].fillna(median, inplace=True)

# Another way fill in the median is with an instance of imputer
imputer = Imputer(strategy="median")

# Median can only be calculated for numerical values
# so ocean_proximity has to go
housing_numerical = housing_copy.drop("ocean_proximity", axis=1)

# 'Train' the imputer
imputer.fit(housing_numerical)

# Set the median for missing values in set
X = imputer.transform(housing_numerical)

# The data is now in a Numpy array
# We transform it back to a DataFrame:
housing_tr = pd.DataFrame(X, columns=housing_numerical.columns)

#################################################
# Handeling string values
#################################################

# 1. Eliminate strings by encoding them:
########################################
# An encoder can make numerical values of strings by mapping them
encoder = LabelEncoder()
# Isolate ocean_proximity
housing_cat = housing_copy["ocean_proximity"]
# Train encoder
housing_cat_encoded = encoder.fit_transform(housing_cat)

# See what happened:
print(encoder.classes_)
print(housing_cat_encoded)

# 2. Apply one-hot-enoding:
########################################
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# This returns a sparse matrix to save memory
# If you want to fill the empty field transorm to array:
housing_cat_1hot.toarray()

# Steps 1 and 2 can be done with one step using the LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)


# Create a Transformer for our Pipeline
housing_copy = strat_train_set.drop("median_house_value", axis=1)
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Create another Transformer for our Pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


from sklearn.base import TransformerMixin  # gives fit_transform method for free


# Had to implement my own transformer because the standard one did not work
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)


num_attribs = list(housing_numerical)
cat_attribs = ["ocean_proximity"]

# Numericals
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
# Categories
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
])
# Join both pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

# Run the whole pipeline:
prepared_data = full_pipeline.fit_transform(housing_copy)
# print(prepared_data.shape)

#################################################
#################################################
# Training a model
#################################################
#################################################

# A linear regression model is chosen
# and trained
lin_reg = LinearRegression()
trained_model = lin_reg.fit(prepared_data, housing_labels)

#################################################
#################################################
# Testing our trained model
#################################################
#################################################
# # Testing on a sample
# some_data = housing_copy.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# # The model works but it is very inaccurate:
# predictions = trained_model.predict(some_data_prepared)
# print("Predictions:", predictions)
# print("Predictions:", list(some_labels))

# # Here we calculate the mse
# housing_predictions = lin_reg.predict(prepared_data)
# housing_predictions = lin_reg.predict(prepared_data)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)

# # In this case the mse represents the amount of dollars we miscalculated
# # We miscalculate (on average) by 68,629 dollars. With housing prices
# # ranging between 120,000 and 265,000 that is a lot
# print(lin_rmse)

#################################################
#################################################
# Trained model was not accurate enough =>
# Train different model
#################################################
#################################################

tree_reg = DecisionTreeRegressor()
tree_reg.fit(prepared_data, housing_labels)
housing_predictions = tree_reg.predict(prepared_data)
housing_predictions = tree_reg.predict(prepared_data)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
# # The rmse was 0.0 this is clearly indicating that the model overfit
# # we need a different way to validate the model

#################################################
#################################################
# We use scikit-learns cross_val_score to validate model
#################################################
#################################################

scores = cross_val_score(tree_reg, prepared_data, housing_labels, scoring="neg_mean_squared_error", cv=10)

# cross_val_score expects a utility function
tree_rmse_scores = np.sqrt(-scores)

print("Scores: ", tree_rmse_scores, " Mean: ", tree_rmse_scores.mean(), "Standard deviation: ", tree_rmse_scores.std())
# The results are not good.


# We test a random forest regressor model
forest_reg = RandomForestRegressor()
forest_reg.fit(prepared_data, housing_labels)
housing_predictions = forest_reg.predict(prepared_data)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
scores = cross_val_score(forest_reg, prepared_data, housing_labels, scoring="neg_mean_squared_error", cv=10)
# cross_val_score expects a utility function
forest_rmse_scores = np.sqrt(-scores)

print("Scores: ", forest_rmse_scores, " Mean: ", forest_rmse_scores.mean(), "Standard deviation: ",
      forest_rmse_scores.std())

# Try Grid Search to find best parameters
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(prepared_data, housing_labels)
# print(grid_search.best_estimator_.feature_importances_)

#################################################
#################################################
# Evaluate System on the test set
#################################################
#################################################

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
