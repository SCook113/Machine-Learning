import warnings

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import DataConversionWarning
from tensorflow import keras as keras

import general_helper_functions as help
import project_helper_functions as project_helpers

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

max_index = 2139  # Complete training data set is used

print("Please enter an index number between 0 and " + str(max_index))
provided_index_number = int(input())

if 0 <= provided_index_number <= max_index:
    print("Ok thanks, wait a second...")
    # Load the data set:
    data = pd.read_csv('data/training.csv')
    # Drop rows with missing values
    data_copy = data.copy().dropna()
    max_index = str(data_copy.shape[0])
    # Seperate features from images
    points, pictures = help.seperate_a_target_column(data_copy, 'Image')

    # Extract pictures into list of numpy arrays
    pictures = project_helpers.extract_pictures_to_np_array_list(pictures)

    # Extract features
    points = project_helpers.extract_points_to_np_array_list(points)

    model = keras.models.load_model("models/model_fully_trained")

    # Get predictions
    predictions = model.predict(pictures[1:2])

    # Get image
    img = pictures[provided_index_number].reshape((96, 96))
    # Put predicted pixels on picture
    preds_as_list = predictions[0].tolist()
    for index in (range(0, len(preds_as_list), 2)):
        # Only draw points if they are in the dimensions of the picture
        if 0 < preds_as_list[index + 1] <= 96 and 0 < preds_as_list[index + 1] <= 96:
            img[int(preds_as_list[index + 1]), int(preds_as_list[index])] = 255

    plt.imshow(img)
    plt.show()
else:
    print("Please try again and provide an index value between 0 and " + str(max_index))
