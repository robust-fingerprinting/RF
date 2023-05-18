import random
import numpy as np
import os
from os.path import join

import const_rf as const
import multiprocessing as mp
import pandas as pd
from importlib import import_module
import tqdm


def parallel(para_list, n_jobs=1):
    pool = mp.Pool(n_jobs)
    data_dict = tqdm.tqdm(pool.imap(extract_feature, para_list), total=len(para_list))
    pool.close()
    return data_dict


def extract_feature(para):
    f, feature_func = para
    file_name = f.split('/')[-1]

    with open(f, 'r') as f:
        tcp_dump = f.readlines()

    seq = pd.Series(tcp_dump[:const.max_trace_length]).str.slice(0, -1).str.split(const.split_mark, expand=True).astype(
        "float")
    times = np.array(seq.iloc[:, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")
    fun = import_module('FeatureExtraction.' + feature_func)
    feature = fun.fun(times, length_seq)
    if '-' in file_name:
        label = file_name.split('-')
        label = int(label[0])
    else:
        label = const.MONITORED_SITE_NUM

    return feature, label


def process_dataset():
    output_dir = const.output_dir + defence + '-' + feature_func

    para_list = []

    for i in range(const.MONITORED_SITE_NUM):
        for j in range(const.MONITORED_INST_NUM):
            if os.path.exists(traces_path + str(i) + '-' + str(j)):
                para_list.append(traces_path + str(i) + '-' + str(j))

    if const.OPEN_WORLD:
        for i in range(const.UNMONITORED_SITE_NUM):
            para_list.append(join(traces_path, str(i)))

    random.shuffle(para_list)

    data_dict = {'dataset': [], 'label': []}
    raw_data_dict = parallel(para_list, n_jobs=15)
    features, label = zip(*raw_data_dict)

    features = np.array(features)
    if len(features.shape) < 3:
        features = features[:, np.newaxis, :]
    labels = np.array(label)

    print("dataset shape:{}, label shape:{}".format(features.shape, labels.shape))
    data_dict['dataset'], data_dict['label'] = features, labels
    np.save(output_dir, data_dict)

    print('save to %s' % (output_dir + ".npy"))


if __name__ == '__main__':

    defence = 'Undefence'
    traces_path = ''            # TODO: change to your dataset path
    feature_func = 'packets_per_slot'

    process_dataset()
