import os
from sklearn.externals import joblib
from LSTM import *

# all required paths and variables
data_dir = "./LSTM_new"
scaler_path = "./scaler.pkl"
save_dir = "./LSTM"
n_smooth = 3
series_length = 20

train_stacked = False

# we use scaler from sklearn to scale features to be on average from -1 to 1
scaler = joblib.load(scaler_path)

# data load
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

if train_stacked:
    lstm_model = StackedLSTM(series_length=series_length, feature_count=len(data[-1][-1]))
    lstm_model.transform_fit(data, labels, test_size=0.05, epochs=20, batch_size=2048, verbose=2)
    lstm_model.save_model(save_dir)
else:
    lstm_model = LSTM(series_length=series_length, feature_count=len(data[-1][-1]))
    lstm_model.transform_fit(data, labels, test_size=0.05, epochs=20, batch_size=2048, verbose=2)
    lstm_model.save_model(save_dir)