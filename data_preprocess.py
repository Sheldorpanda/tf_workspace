import numpy as np

# Slice time domain input data into chunks, fft into frequency domain
def data_process(raw_datafile, dir_raw, stride):
    data = np.genfromtxt(dir_raw + raw_datafile, delimiter=',')
    n = len(data) // stride
    l = []
    for i in range(n):
        data_batch = data[i * stride : (i+1) * stride]
        data_f = np.fft.fft(data_batch, axis=0)
        l.append(data_f)
    l = np.concatenate(l)
    return l

# Label averaged for each tensor
def label_process(raw_label_datafile, dir_label, stride):
    labels = np.genfromtxt(dir_label + raw_label_datafile, delimiter=',')
    n = len(labels) // stride # num batches
    m = len(labels[0]) # dim
    l = []
    for i in range(n):
        label_batch = []
        chunk = labels[i * stride: (i + 1) * stride]
        for j in range(m):
            chunk_dim = [entry[j] for entry in chunk]
            label_batch.append([np.average(chunk_dim)] * stride)
        label_batch = np.dstack(label_batch).squeeze()
        l.append(label_batch)
    l = np.concatenate(l)
    return l

# API for deepSense
def preprocess(raw_data_files, raw_label_file, processed_file_name, dir_raw, dir_label, stride=16):
    l = [data_process(raw_data_file, dir_raw, stride=stride) for raw_data_file in raw_data_files]
    # Check calibration among data, can be optimized to O(nlogn)
    for i in range(len(l)):
        for j in range(len(l)):
            if l[i].shape[0] != l[j].shape[0]:
                raise Exception('Data not calibrated')
    labels = label_process(raw_label_file, dir_label, stride=stride)
    # Check calibration with labels
    if l[0].shape[0] != labels.shape[0]:
        raise Exception('Data and label not calibrated')
    l.append(labels)
    # Write to a new file for DeepSense
    processed_file = DIR_PROCESSED + processed_file_name
    processed_data = np.dstack(np.concatenate(l, axis=1)).squeeze().transpose()
    np.savetxt(processed_file, processed_data, delimiter=',')
    print('Preprocession finished')
    return stride


raw_data_files = ['rel_angle.csv']
raw_label_file = 'curr_state.csv'
preprocess(raw_data_files, raw_label_file, 16)


