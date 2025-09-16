import numpy as np

data = np.load(r"C:\Users\despo\Documents\Thesis2025\HarProject\results\cnn_lstm\Walking\Walking_Y_pred_all.npy")
data1 = np.load(r"C:\Users\despo\Documents\Thesis2025\HarProject\results\cnn_lstm\Walking\Walking_timestamps_all.npy")
print(len(data))
print(len(data1))
