import torch.nn as nn
import math


class RF_CAM(nn.Module):

    def __init__(self, features, num_classes=95, init_weights=True):
        super(RF_CAM, self).__init__()
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        self.first_layer = make_first_layers()
        self.features = features
        self.class_num = num_classes
        self.classifier = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_classes, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.first_layer(x)
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, in_channels=64):
    layers = []

    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.3)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), stride=1, padding=(1, 1))
            layers += [conv2d, nn.BatchNorm2d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)


cfg = {
    'N': [128, 128, 'M', 256, 256, 'M', 512]
}


def getRF_CAM(num):
    model = RF_CAM(make_layers(cfg['N'] + [num]), num_classes=num)
    return model


if __name__ == '__main__':
    net = RF_CAM(95)
