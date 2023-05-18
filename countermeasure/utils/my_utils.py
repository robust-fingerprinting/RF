import numpy as np
import os
import multiprocessing as mp
import pandas as pd
import torch
from matplotlib import pyplot as plt
from countermeasure.utils.const_exp import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)

def get_cam_dict(data_path):
    data_dict = np.load(data_path, allow_pickle=True).item()
    return data_dict

def extract_cam(input_tensor: torch.Tensor, input_cam: np.ndarray, threshold: float, adaptive=False):
    inp_cam = input_cam.copy()
    inp_array = input_tensor.clone().detach().cpu().numpy()
    ans = []
    if not adaptive:
        inp = np.where(inp_cam >= threshold, 1, 0)
        inp = np.append(np.array([0]), inp)
        inp = np.append(inp, np.array([0]))
        inp = inp[1:] - inp[0:-1]
        idx_le = np.where(inp == 1)[0].tolist()
        idx_ri = np.where(inp == -1)[0].tolist()
        assert len(idx_le) == len(idx_ri)

        for le, ri in zip(idx_le, idx_ri):
            ans.append(inp_array[int(le):int(ri)])
    else:
        return ans
    
    return ans

def show_cam(gray_cam, target_label, nums, out_idx=None, fig_path=None):
    for k in range(nums):
        if out_idx is not None:
            idx = out_idx
        else:
            idx = k
        cam_len = gray_cam.shape[-1]
        plt.figure(figsize=(10, 5))
        channels = gray_cam.shape[1]
        fig, ax = plt.subplots(channels, 1)
        if channels == 1:
            ax = [ax]
        plt.suptitle(f"idx:{idx} label:{int(target_label[k])}", x=0.05, y=0.5, va="center", rotation=90)
        for channel in range(channels):
            cam = gray_cam[k, channel, :]

            cam_max = cam.max()
            cam_min = cam.min()
            if cam_max == 0.:
                print(f"idx:{idx} will divided by zero!")
            for i in range(cam_len - 1):
                ax[channel].plot(np.array([i, i + 2]), cam[i: i + 2],
                           color=plt.cm.jet(int(255 * (cam[i] - cam_min) / (cam_max - cam_min + 1e-7))))
            if fig_path is not None:
                plt.savefig(fig_path + f'{idx}.png')

        plt.show()


def get_cam_set(data_path):
    data = np.load(data_path, allow_pickle=True).item()

    camset, labels = data['camset'], data['label']

    return camset, labels


def get_dataset(data_path):
    data = np.load(data_path, allow_pickle=True).item()

    dataset, labels = data['dataset'], data['label']

    dataset = torch.from_numpy(dataset).type(torch.FloatTensor).to(device)
    if len(dataset.shape) == 3:
        dataset = dataset.unsqueeze(1)
    labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)

    print(dataset.shape)
    print(labels.shape)
    return dataset, labels


def extract_new_feature(inp):
    size = inp.shape[0]
    feature = np.zeros((1, 2, max_matrix_len))

    for i in range(size):
        if inp[i, 1] > 0:
            if inp[i, 0] >= maximum_load_time:
                feature[0][0][-1] += 1
            else:
                idx = int(inp[i, 0] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][0][idx] += 1
        elif inp[i, 1] < 0:
            if inp[i, 0] >= maximum_load_time:
                feature[0][1][-1] += 1
            else:
                idx = int(inp[i, 0] * (max_matrix_len - 1) / maximum_load_time)
                feature[0][1][idx] += 1

    return feature


def parallel(file_list, n_jobs=20):
    pool = mp.Pool(n_jobs)
    data_dict = pool.map(extract_feature, file_list)

    return data_dict


def extract_feature(file_path):
    file_name = file_path.split('/')[-1]

    with open(file_path, 'r') as f:
        tcp_dump = f.readlines()

    features = pd.Series(tcp_dump[:5000]).str.slice(0, -1).str.split('\t', expand=True).astype('float')
    features = np.array(features)

    features = extract_new_feature(features)

    if '-' in file_name:
        label = int(file_name.split('-')[0])
    else:
        label = 95

    return features, label

def dump(trace, output_path, file_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, file_name), 'w') as fo:
        for i in range(len(trace)):
            fo.write("{}".format(trace[i][0]) + '\t' + "{}".format(int(trace[i][1])) + '\n')
