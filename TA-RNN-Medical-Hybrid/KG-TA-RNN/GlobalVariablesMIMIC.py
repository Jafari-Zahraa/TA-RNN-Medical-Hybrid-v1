import pandas as pd
import numpy as np

pathMatrix = '../KG-TA-RNN/SNOMED/DataScience/snomed_embedding_matrix.npy'
embedding_matrix = np.load(pathMatrix)


print("Type:", type(embedding_matrix))
print("Shape:", embedding_matrix.shape)  # (num_codes, embedding_dim)

print("First 5 rows:\n", embedding_matrix[:5])

print("min:", np.min(embedding_matrix))
print("max:", np.max(embedding_matrix))
print("mean:", np.mean(embedding_matrix))
print("std:", np.std(embedding_matrix))

# Longitudinal training data
file_name = '../KG-TA-RNN/MIMIC/DataScience/longitudinal_data_train.pkl'
lon_data_train = pd.read_pickle(file_name)

# Labels of traing data
file_name = '../KG-TA-RNN/MIMIC/DataScience/label_train.pkl'
label_train = pd.read_pickle(file_name)

# Demographic training data
file_name = '../KG-TA-RNN/MIMIC/DataScience/demographic_data_train.pkl'
dem_data_train = pd.read_pickle(file_name)

# elapsed time training data
file_name = '../KG-TA-RNN/MIMIC/DataScience/elapsed_data_train.pkl'
time_train = pd.read_pickle(file_name)

# Longitudinal test data
file_name = '../KG-TA-RNN/MIMIC/DataScience/longitudinal_data_test.pkl'
lon_data_test = pd.read_pickle(file_name)

# Labels of test data
file_name = '../KG-TA-RNN/MIMIC/DataScience/label_test.pkl'
label_test = pd.read_pickle(file_name)

# Demographic test data
file_name = '../KG-TA-RNN/MIMIC/DataScience/demographic_data_test.pkl'
dem_data_test = pd.read_pickle(file_name)

# elapsed time test data
file_name = '../KG-TA-RNN/MIMIC/DataScience/elapsed_data_test.pkl'
time_test = pd.read_pickle(file_name)

# lon_last data
file_name = '../KG-TA-RNN/MIMIC/DataScience/lon_data_last.pkl'
lon_data_last = pd.read_pickle(file_name)
# time_last data
file_name = '../KG-TA-RNN/MIMIC/DataScience/elapsed_last.pkl'
time_data_last = pd.read_pickle(file_name)

# demo_last data
file_name = '../KG-TA-RNN/MIMIC/DataScience/demo_data_last.pkl'
demo_data_last = pd.read_pickle(file_name)

# patient_idx
file_name = '../KG-TA-RNN/MIMIC/DataScience/rid_last.pkl'
rid_last = pd.read_pickle(file_name)

# # Remove age from lon_data_train and lon_data_test
# for i in range(len(lon_data_train)):
#     lon_data_train[i] = lon_data_train[i][:,:,1:]
#     lon_data_test[i] = lon_data_test[i][:,:,1:]

# This represents number of visits (time points) will be used in the training.
time_steps = lon_data_test[0].shape[1]

# This represents number of future visits ahead to predict
future_time_s = label_test[0].shape[1]

# This represents how many featutes in each visit (longitudinal).
num_features_in_each_time_step = lon_data_test[0].shape[2]

# This represents how many demographic featutes (cross sectional).
demographic_features = len(dem_data_test[0][0])

print(f"time_steps = {time_steps}")
print(f"num_features_in_each_time_step = {num_features_in_each_time_step}")



