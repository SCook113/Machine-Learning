import general_helper_functions as help
import pandas as pd

data = pd.read_csv('../data/training.csv')
help.write_df_infos_to_file(data, "training_data_info.txt")