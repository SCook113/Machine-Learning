import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from tensorflow import keras as keras

import general_helper_functions as help
import project_helper_functions as project_helpers

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

model_to_use = "model_fully_trained"

# Load the data set:
print("# Load the data set:")
data = pd.read_csv('data/training.csv')
# Drop rows with missing values
data_copy = data.copy().dropna()

# Seperate features from images
points, pictures = help.seperate_a_target_column(data_copy, 'Image')

# Extract pictures into list of numpy arrays
pictures = project_helpers.extract_pictures_to_np_array_list(pictures)

# Extract features
print("# Extract features")
points = project_helpers.extract_points_to_np_array_list(points)

model = keras.models.load_model("models/" + model_to_use)

# Get predictions
print("# Get predictions")
predictions = model.predict(pictures[1:2])

# Save predictions to files
print("# Save predictions to files")
for i in range(0, pictures.shape[0]):
    # Get image
    img = pictures[i].reshape((96, 96))
    # Put predicted pixels on picture
    preds_as_list = predictions[0].tolist()
    for index in (range(0, len(preds_as_list), 2)):
        # Only draw points if they are in the dimensions of the picture
        if 0 < preds_as_list[index + 1] <= 96 and 0 < preds_as_list[index + 1] <= 96:
            img[int(preds_as_list[index + 1]), int(preds_as_list[index])] = 255
    np.save("predictions_" + model_to_use + "/" + str(i) + "_prediction", img)
