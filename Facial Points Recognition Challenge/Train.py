import warnings

import pandas as pd
import tensorflow as tf
from sklearn.exceptions import DataConversionWarning
from tensorflow import keras as keras

import general_helper_functions as help
import project_helper_functions as project_helpers

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load the data set:
data = pd.read_csv('data/training.csv')

# To get a overview of the data I write all the infos to a file
# help.write_df_infos_to_file(data, "training_data_small_info.txt")

# There are a lot of missing values in the set so we will delete all lines with missing values for now

# Drop rows with missing values
data_copy = data.copy().dropna()

# Seperate features from images
points, pictures = help.seperate_a_target_column(data_copy, 'Image')

# Extract pictures into list of numpy arrays
pictures = project_helpers.extract_pictures_to_np_array_list(pictures)

# Extract features
points = project_helpers.extract_points_to_np_array_list(points)

# Train a fully connected neural net
model = keras.Sequential([keras.layers.Flatten(input_shape=(96, 96, 1)),
                          keras.layers.Dense(128, activation="relu"),
                          keras.layers.Dropout(0.1),
                          keras.layers.Dense(64, activation="relu"),
                          keras.layers.Dense(30)
                          ])
# Compile model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mse',
              metrics=['mae'])

# Train model
model.fit(pictures, points, epochs=500)
model.save("models/model_fully_trained")
