import numpy as np

def extract_pictures_to_np_array_list(pictures):
    '''
    Extracts the pictures from dataframe, formats them correctly and puts them in a
    list on numpy arrays
    :param pictures: Dataframe column with pictures in them
    :return:
    '''
    # Extract pictures into list of numpy arrays
    IMAGE_WIDTH = 96
    IMAGE_HEIGHT = 96
    x_train = []
    picture_index = 0
    # Loop through the data frame and get index number
    for index, row in pictures.iterrows():
        # Create an numpy array for the image we are about to extract
        img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float)
        # Turn picture into a list
        picture_as_list = row["Image"].split(" ")
        # Counter for pixels in the picture_as_list to keep track
        # which pixel we have already saved
        i = 0
        for x_coord in range(0, 96):
            for y_coord in range(0, 96):
                img[x_coord, y_coord, 0] = picture_as_list[i]
                i += 1
        # Append Array to a list
        x_train.append(img)
    return np.array(x_train)

def extract_points_to_np_array_list(points):
    y_train = []
    for index , row in points.iterrows():
        points = []
        for i in row:
            points.append(float(i))
        y_train.append(points)
    return np.array(y_train, dtype=np.float)