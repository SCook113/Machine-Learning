#######################################
# This is a tutorial from
# https://www.datacamp.com/community/tutorials/categorical-data
# "Handling Categorical Data in Python"
#######################################
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1200)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from pyspark import SparkContext
from pyspark.sql import SparkSession as spark
from pyspark.ml.feature import StringIndexer, OneHotEncoder

#######################################
# Fetching and exploring the data
#######################################
df_flights = pd.read_csv('data/flights.csv')

df_flights.info()
print(df_flights.columns.values)
print(df_flights.head(30).to_string())
print(df_flights.describe())

# Since we are only interested in the categorical data
# we copy only the columns with categorical data and explore further
cat_df_flights = df_flights.select_dtypes(include=['object']).copy()

print(cat_df_flights.head(5))
print(cat_df_flights.describe())
print(cat_df_flights.info())

# Check for sum null values and their distribution among columns
print(cat_df_flights.isnull().values.sum())
print(cat_df_flights.isnull().sum())

# All the null values are in the column 'tailnum'. We now fill
# the missing values with the most frequent value in the column
cat_df_flights = cat_df_flights.fillna(cat_df_flights['tailnum'].value_counts().index[0])

# Explore carrier column
print(cat_df_flights['carrier'].value_counts())
print(cat_df_flights['carrier'].value_counts().count())

# Vizualise frequency distribution od the carrier column
carrier_count = cat_df_flights['carrier'].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of Carriers')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Carrier', fontsize=12)
plt.show()

# Vizualise as pie chart
labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
counts = cat_df_flights['carrier'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()

#######################################
# Encoding categorical data
#######################################


################
# Replacing
################

# Replacing with numbers in a dictionary that we create
# Categories are: ['AA', 'AS', 'B6', 'DL', 'F9', 'HA', 'OO', 'UA', 'US', 'VX', 'WN']
replace_map = {
    'carrier': {'AA': 1, 'AS': 2, 'B6': 3, 'DL': 4, 'F9': 5, 'HA': 6, 'OO': 7, 'UA': 8, 'US': 9, 'VX': 10, 'WN': 11}}

# Get list of categories
labels = cat_df_flights['carrier'].astype('category').cat.categories.tolist()
# Generate new column
replace_map_comp = {'carrier': {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}

print(replace_map_comp)

# Make a copy
cat_df_flights_replace = cat_df_flights.copy()
# Replace
cat_df_flights_replace.replace(replace_map_comp, inplace=True)

print(replace_map_comp)
print('#' * 60)
print('New Dataframe:')
print(cat_df_flights_replace.head())

################
# Label Encoding
################

# Each label gets a number assigned
cat_df_flights_lc = cat_df_flights.copy()

# Cast columns to type 'category' so operations are faster
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].astype('category')
cat_df_flights_lc['origin'] = cat_df_flights_lc['origin'].astype('category')

# Encode categories
cat_df_flights_lc['carrier'] = cat_df_flights_lc['carrier'].cat.codes

# Give line a code depending on value in a different column
cat_df_flights_specific = cat_df_flights.copy()
cat_df_flights_specific['US_code'] = np.where(cat_df_flights_specific['carrier'].str.contains('US'), 1, 0)
print(cat_df_flights_specific.head(5))

# You can do the label encoding with sklearn's label encoder also
cat_df_flights_sklearn = cat_df_flights.copy()
encoder = LabelEncoder()
cat_df_flights_sklearn['carrier_code'] = encoder.fit_transform(cat_df_flights['carrier'])
print(cat_df_flights_sklearn.head())

################
# One-Hot encoding
################

cat_df_flights_onehot = cat_df_flights.copy()
cat_df_flights_onehot = pd.get_dummies(cat_df_flights_onehot, columns=['carrier'], prefix=['carrier'])
print(cat_df_flights_onehot.head())

# You can do the this with sklearn's LabelBinarizer() also
cat_df_flights_onehot_sklearn = cat_df_flights.copy()
lb = LabelBinarizer()
lb_results = lb.fit_transform(cat_df_flights_onehot_sklearn['carrier'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)
print(lb_results_df.head())

################
# Binary encoding
################

cat_df_flights_ce = cat_df_flights.copy()

# This method creates less additional columns than One-Hot encoding
encoder = ce.BinaryEncoder(cols=['carrier'])
df_binary = encoder.fit_transform(cat_df_flights_ce)
print(df_binary.head())

################
# Backward Difference Encoding
################

encoder = ce.BackwardDifferenceEncoder(cols=['carrier'])
df_bd = encoder.fit_transform(cat_df_flights_ce)
df_bd.head()

################
# Miscellaneous Features
################

# Example of how to prepare a feature that has ranges as values

dummy_df_age = pd.DataFrame({'age': ['0-20', '20-40', '40-60', '60-80']})

dummy_df_age['start'], dummy_df_age['end'] = zip(*dummy_df_age['age'].map(lambda x: x.split('-')))
print(dummy_df_age.head())


# Alternatively you could enter it's mean
def split_mean(x):
    split_list = x.split('-')
    print(split_list)
    mean = (float(split_list[0]) + float(split_list[1])) / 2
    return mean


dummy_df_age['age_mean'] = dummy_df_age['age'].apply(lambda x: split_mean(x))
print(dummy_df_age.head())

################
# Dealing with Categorical Features in Big Data with Spark
################

sc = SparkContext()
sc.setLogLevel("ERROR")
spark = spark.builder.appName("DataTutorial").getOrCreate()

from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)

# Load the csv file
spark_flights = spark.read.format("csv").option('header', True).load('data/flights.csv', inferSchema=True)

# # Inspect data
# spark_flights.show(3)
# spark_flights.printSchema()

# # Should be empty
# print(spark.catalog.listTables())
# # Register data
# spark_flights.createOrReplaceTempView("flights_temp")
# # Data should appear
# print(spark.catalog.listTables())


################
# Spark StringIndexer()
################

# Only load carrier column
carrier_df = spark_flights.select("carrier")
# carrier_df.show(5)
# Spark method of indexing string values with numerical values
# Set up StringIndexer()
carr_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
# Transform data
carr_indexed = carr_indexer.fit(carrier_df).transform(carrier_df)
# carr_indexed.show(7)

# Do a OneHotEncoder first and then add StringIndexer
carrier_df_onehot = spark_flights.select("carrier")

stringIndexer = StringIndexer(inputCol="carrier", outputCol="carrier_index")
model = stringIndexer.fit(carrier_df_onehot)
indexed = model.transform(carrier_df_onehot)
encoder = OneHotEncoder(dropLast=False, inputCol="carrier_index", outputCol="carrier_vec")
encoded = encoder.transform(indexed)

encoded.show(7)
