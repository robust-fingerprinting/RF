import torch
import numpy as np
import cv2

from countermeasure.get_cam.cam_utils import *

device = torch.device("cuda:0")


class BaseCAM:
    def __init__(self, model, target_layers, use_gradient=False, scale=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.use_gradient = use_gradient
        self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers)
        self.scale = scale

    def get_cam_weights(self, input_tensor, target_layers, target_label, activations, grads):
        pass

    def get_loss(self, output, target_label):
        loss = 0
        for i in range(len(target_label)):
            loss = loss + output[i, target_label[i]]

        return loss

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_label,
                      activations,
                      grads):
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       target_label, activations, grads)
        if activations.ndim == 3:
            weighted_activations = weights[:, :, None] * activations
            weighted_activations = weighted_activations[:, :, None, :]
        else:
            weighted_activations = weights[:, :, None, None] * activations

        cam = weighted_activations.sum(axis=1)
        return cam

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            if target_size is not None:
                new_img = np.empty((0, target_size[0]))
            else:
                new_img = np.empty((0, img.shape[-1]))
            for channel in img:
                channel = channel[None, :]
                if self.scale:
                    channel = channel - np.min(channel)
                    channel = channel / (1e-7 + np.max(channel))
                if target_size is not None:
                    channel = cv2.resize(channel, (target_size[0], 1))
                new_img = np.append(new_img, channel, axis=0)
            result.append(new_img)

        result = np.float32(result)

        return result

    def compute_cam_per_layer(self, input_tensor, target_label):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]

        target_size = self.get_target_width_height(input_tensor)
        cam_per_target_layer = []

        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     target_label,
                                     layer_activations,
                                     layer_grads)
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def forward(self, input_tensor, target_label=None):

        output = self.activations_and_grads(input_tensor)

        if isinstance(target_label, int):
            target_label = [target_label] * input_tensor.size(0)

        if target_label is None:
            target_label = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert (len(target_label) == input_tensor.size(0))

        if self.use_gradient:
            self.model.zero_grad()
            loss = self.get_loss(output, target_label)
            loss.backward()

        cam_per_layer = self.compute_cam_per_layer(input_tensor, target_label)

        return self.aggregate_multi_layers(cam_per_layer)

    def __call__(self, input_tensor, target_label=None):
        return self.forward(input_tensor, target_label)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
