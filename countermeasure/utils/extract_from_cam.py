import os
from const_exp import *
from my_utils import get_cam_set, extract_cam, get_dataset, show_cam
import numpy as np

if __name__ == '__main__':
    cam_datas, cam_labels = get_cam_set(r'../cam_dataset/cam_score.npy')
    train_data, train_label = get_dataset(r'../dataset/Undefence-train-packets_per_slot.npy')

    data_dict_upper = {}
    data_dict_down = {}

    for idx, (cam_data, cam_label) in enumerate(zip(cam_datas, cam_labels)):
        label = int(cam_label)
        # show_cam(cam_data[None, :], [label], 1, idx)
        real_tensor = train_data[idx]

        if not (label in data_dict_upper.keys()):
            data_dict_label = {label: []}
            data_dict_upper.update(data_dict_label)
            data_dict_down.update(data_dict_label)

        extract_part_upper = extract_cam(train_data[idx, 0, 0], cam_data[0], threshold, adaptive=False)
        extract_part_down = extract_cam(train_data[idx, 0, 1], cam_data[1], threshold, adaptive=False)

        data_dict_upper[label] += extract_part_upper
        data_dict_down[label] += extract_part_down
        
        if idx % 1000 == 0:
            print("idx: {} complete!".format(idx))

    for key in sorted(data_dict_upper.keys()):
        print("label: {} nums: {}".format(key, len(data_dict_upper[key])))

    # The informative_regions_*.npy is used for generating the defended traces
    np.save(os.path.join('../cam_dataset/informative_regions/', "informative_regions_upper_" + str(threshold)), data_dict_upper)
    np.save(os.path.join('../cam_dataset/informative_regions/', "informative_regions_down_" + str(threshold)), data_dict_down)