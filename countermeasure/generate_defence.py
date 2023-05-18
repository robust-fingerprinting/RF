import os
import numpy as np
import random

import tqdm
import multiprocessing as mp
from utils.my_utils import get_cam_dict, dump
from utils.const_exp import *

device = 'cuda:0'

def fill_cam(cam_vec, fill_num):
    pruned_cam = np.trim_zeros(cam_vec)
    filled_cam = np.where(pruned_cam > fill_num, pruned_cam, fill_num)
    return filled_cam


def patch_delay(ori_vec, cam_list, fill_num):
    defended_vec = np.copy(ori_vec)
    cam_len = len(cam_list)
    cnt_cam = 0
    idx = 0
    total_cum = 0
    total_sum = 0
    total_add = 0
    cum = 0

    while idx < max_matrix_len - 1:
        if ori_vec[idx] > 1 and cnt_cam < cam_len:
            cam_vec = cam_list[cnt_cam]
            cam_vec = fill_cam(cam_vec, fill_num)

            total_sum += np.sum(cam_vec)
            cnt_cam += 1

            max_cam_len = min(len(cam_vec), max_matrix_len - idx - 1)
            for i in range(max_cam_len):
                now_real = ori_vec[idx + i] + cum
                now_cam = cam_vec[i]
                max_now = np.ceil((1 + delta_up) * now_cam)
                min_now = np.ceil((1 - delta_down) * now_cam)

                if min_now <= now_real <= max_now:
                    defended_vec[idx + i] = now_real
                    cum = 0
                elif now_real < min_now:
                    total_add += (now_cam - now_real)
                    defended_vec[idx + i] = now_cam
                    cum = 0
                else:
                    defended_vec[idx + i] = now_cam
                    cum = now_real - max_now
                    total_cum += (now_real - max_now)

            idx += max_cam_len
        elif ori_vec[idx] <= 1 and cum >= D and cnt_cam < cam_len:
            cam_vec = cam_list[cnt_cam]
            cam_vec = fill_cam(cam_vec, fill_num)

            total_sum += np.sum(cam_vec)
            cnt_cam += 1

            max_cam_len = min(len(cam_vec), max_matrix_len - idx - 1)
            for i in range(max_cam_len):
                now_real = ori_vec[idx + i] + cum
                now_cam = cam_vec[i]
                max_now = np.ceil((1 + delta_up) * now_cam)
                min_now = np.ceil((1 - delta_down) * now_cam)

                if min_now <= now_real <= max_now:
                    defended_vec[idx + i] = now_real
                    cum = 0
                elif now_real < min_now:
                    total_add += (now_cam - now_real)
                    defended_vec[idx + i] = now_cam
                    cum = 0
                else:
                    defended_vec[idx + i] = now_cam
                    cum = now_real - max_now
                    total_cum += (now_real - max_now)

            idx += max_cam_len
        elif ori_vec[idx] <= 1 and cum > 0:
            num_to_add = min(cum, random.randint(1, U))
            defended_vec[idx] = ori_vec[idx] + num_to_add
            cum -= num_to_add
            idx += 1
        else:
            defended_vec[idx] = ori_vec[idx]
            idx += 1

    if cum > 0:
        defended_vec[idx] = ori_vec[idx] + cum

    return defended_vec


def sample_information_regions(ori_label):
    upper_list = []
    down_list = []

    cnt_upper = 0
    cnt_down = 0

    tar_label = random.randint(0, num_classes - 1)
    num = len(data_dict_upper[tar_label])
    while tar_label == ori_label or num < 1:
        tar_label = random.randint(0, num_classes - 1)
        num = len(data_dict_upper[tar_label])

    while cnt_upper < N:
        upper_list.append(random.sample(data_dict_upper[tar_label], 1)[0])
        cnt_upper += 1

    while cnt_down < N:
        down_list.append(random.sample(data_dict_down[tar_label], 1)[0])
        cnt_down += 1

    return upper_list, down_list, tar_label


def get_dataset(data_path):
    data = np.load(data_path, allow_pickle=True).item()

    dataset, labels = data['dataset'], data['label']

    if len(dataset.shape) == 3:
        dataset = dataset[:, np.newaxis, :, :]

    return dataset, labels


def trans_to_packets(matrix):
    packets = np.empty((0, 2), dtype=np.float32)
    for i in range(max_matrix_len - 1):
        upper_num = matrix[0, i]
        down_num = matrix[1, i]
        total_num = upper_num + down_num

        time_stamps = np.random.uniform(i * time_slot, (i + 1) * time_slot, int(total_num))
        time_stamps = np.sort(time_stamps)

        for j in range(int(upper_num)):
            packets = np.append(packets, np.array([[time_stamps[j], 1]]), axis=0)

        for j in range(int(upper_num), int(total_num)):
            packets = np.append(packets, np.array([[time_stamps[j], -1]]), axis=0)

    return packets


def parallel(para_list, n_jobs=15):
    pool = mp.Pool(n_jobs)
    data_dict = tqdm.tqdm(pool.imap(get_fake, para_list), total=len(para_list))
    pool.close()
    return data_dict


def get_fake(para):
    te_x, te_y, upper_append_list, down_append_list, fake_label, file_name = para
    ori_band = np.sum(te_x).item()

    defended_upper = patch_delay(te_x[0, 0, 0, :], upper_append_list, fill_num_up)
    defended_down = patch_delay(te_x[0, 0, 1, :], down_append_list, fill_num_down)

    defended_data = np.stack((defended_upper, defended_down), 0).reshape(2, -1)

    defended_band = np.sum(defended_data).item()

    ori_vector = te_x[0, 0, 0, :] + te_x[0, 0, 1, :]
    tar_vector = defended_data[0, :] + defended_data[1, :]
    ori_slot = np.nonzero(ori_vector)[-1][-1]
    tar_slot = np.nonzero(tar_vector)[-1][-1]

    if ori_slot > tar_slot:
        tar_slot = ori_slot

    defended_time = (tar_slot + 1) * time_slot
    real_time = (ori_slot + 1) * time_slot

    '''
        You can dump the packets by uncommenting the following lines
        Please make sure that the 'output_path' exists !
    '''
    # packets = trans_to_packets(defended_data)
    # dump(packets, output_path, file_name)

    return defended_data, te_y, defended_band / ori_band - 1., defended_time / real_time - 1.


if __name__ == '__main__':

    data_dict_upper = get_cam_dict(
        r'cam_dataset/informative_regions/informative_regions_upper_' + str(threshold) + '.npy')

    data_dict_down = get_cam_dict(
        r'cam_dataset/informative_regions/informative_regions_down_' + str(threshold) + '.npy')

    test_data, test_label = get_dataset(f'dataset/Undefence-{suffix}-packets_per_slot.npy')

    process_data = []

    lines = None

    with open(os.path.join('list', f'Index_{suffix}.txt')) as f:
        lines = f.readlines()

    for idx in range(total_trace):
        upper_append_list, down_append_list, fake_label = sample_information_regions(test_label[idx:idx + 1])
        file_name = lines[idx].strip()
        process_data.append(
            (test_data[idx:idx + 1], test_label[idx], upper_append_list, down_append_list, fake_label, file_name))

    raw_data_dict = parallel(process_data, n_jobs=15)

    features, label, bandwidth, time = zip(*raw_data_dict)
    features = np.array(features)
    labels = np.array(label)

    data_dict = {'dataset': features, 'label': labels}

    print('bandwidth:{}'.format(np.median(np.array(bandwidth))))
    print('time:{}'.format(np.median(np.array(time))))

    if save:
        np.save('dataset/countermeasure_{}'.format(suffix), data_dict)
