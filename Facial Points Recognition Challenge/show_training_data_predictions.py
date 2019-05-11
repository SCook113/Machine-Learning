import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

max_index = 2139  # Complete training data set is used

print("Please enter an index number between 0 and " + str(max_index))
provided_index_number = int(input())

if 0 <= provided_index_number <= max_index:
    img = np.load("predictions_model_fully_trained/" + str(provided_index_number) + "_prediction.npy")
    plt.imshow(img)
    plt.show()
else:
    print("Please try again and provide an index value between 0 and " + str(max_index))
