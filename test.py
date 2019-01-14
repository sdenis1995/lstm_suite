import numpy as np
from LSTM import *
from sklearn.externals import joblib
import os

# all required paths and variables
data_dir = "./LSTM_new"
scaler_path = "./scaler.pkl"
model_dir = "./LSTM"
test_stacked = False

if test_stacked:
    lstm_model = StackedLSTM.load_model(model_dir)
else:
    lstm_model = LSTM.load_model(model_dir)
scaler = joblib.load(scaler_path)

# loading data, same as in train.py, just for testing
data = []
labels = []

files = os.listdir(data_dir)
for file in files:
    fp = open("{}/{}".format(data_dir, file))
    lines = fp.readlines()
    if len(lines) == 0:
        continue
    job_data = []
    job_labels = []
    for line in lines:
        job_data.append([float(x) for x in line.split(";")])
        job_labels.append(int(job_data[-1][-1]) - 1) # classes must start at 0 (in our case it was 1)
        job_data[-1] = job_data[-1][:-1]
    # we tried smoothing the time series, but it did not affect the results
    #job_data = smoothen(job_data, neighbours=3, leave_last=True)
    data.append(scaler.transform(job_data))
    labels.append(job_labels)

lstm_types = lstm_model.transform_predict(data)
labels = lstm_model.transform_Y(time_series=labels, dtype=np.int)

# just to see, whats the score
print(len([i for i, j in zip(labels, lstm_types) if i == j]) / len(lstm_types))



