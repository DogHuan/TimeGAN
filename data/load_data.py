
import numpy as np
from os.path import dirname, abspath


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def real_data_loading(data_path, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    ori_data = np.loadtxt(dirname(dirname(abspath(__file__))) + '/data/01SZ_with_tag.csv', delimiter=",", skiprows=1)
    # Flip the data to make chronological data
    ori_data = ori_data[::-1]
    # Normalize the data
    ori_data = MinMaxScaler(ori_data)

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    no = len(data)
    output = np.array(data)
    time = []

    # For each uniq id
    for i in range(no):
        # Extract the time-series data with a certain admissionid

        curr_data = data[i]

        # Normalize data
        curr_data = MinMaxScaler(curr_data)

        # Extract time and assign to the preprocessed data (Excluding ID)
        curr_no = len(curr_data)

        # Pad data to `max_seq_len`
        if curr_no >= seq_len:
            output[i, :, 1:] = curr_data[:seq_len, 1:]  # Shape: [1, max_seq_len, dim]
            time.append(seq_len)
        else:
            output[i, :curr_no, 1:] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
            time.append(curr_no)

    return output, time


def load_data(data_path, max_seq_len):
    # Data loading
    ori_data, time = real_data_loading(data_path, max_seq_len)  # list: 3661; [24,6]

    return ori_data[:, :, 1:], time
