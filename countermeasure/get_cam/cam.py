import numpy as np
import torch
import torch.utils.data as Data

from countermeasure.utils.my_utils import get_dataset
from countermeasure.models.RF_CAM import *
from countermeasure.get_cam.base_cam import BaseCAM
from countermeasure.utils.const_exp import *


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)

class CAM(BaseCAM):
    def __init__(self, model, target_layers, use_gradient=False, scale=False):
        super(CAM, self).__init__(model, target_layers, use_gradient, scale)

    def get_cam_weights(self, input_tensor, target_layers, target_label, activations, grads):

        assert hasattr(self, "weight_layer")
        weights_target = self.model._modules.get(self.weight_layer).weight.data[target_label, :]

        return weights_target.cpu().detach().data.numpy()


if __name__ == '__main__':
    model = getRF_CAM(num_classes)
    model.load_state_dict(torch.load(r'../pretrained/Undefence-train-packets_per_slot_CAM.pth'))
    model.to(device)
    model.eval()

    # TODO: run RF/extract_list.py to get the dataset and copy it to ../dataset/
    train_data, train_label = get_dataset(r'../dataset/')
    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=False)

    data_dict = {'camset': [], 'label': []}
    cam_sets = np.array([])
    cam_labels = np.array([])
    target_layers = [model.features[-1]]

    for idx, (input_tensor, target_label) in enumerate(train_loader):
        target_label = target_label.cpu().numpy().tolist()
        with CAM(model, target_layers, True) as cam:
            cam.weight_layer = "fc"
            gray_cam = cam(input_tensor, target_label)
            if idx == 0:
                cam_sets = gray_cam.copy()
                cam_labels = np.array(target_label)
            else:
                cam_sets = np.concatenate((cam_sets, gray_cam.copy()), axis=0)
                cam_labels = np.concatenate((cam_labels, np.array(target_label)), axis=0)
        print("idx: {} complete!".format(idx))

    data_dict['camset'], data_dict['label'] = cam_sets, cam_labels

    # The cam_score.npy is used for informative regions extraction
    np.save(r'../cam_dataset/cam_score', data_dict)
