
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BasicModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        num_classes = cfg.MODEL.NUM_CLASSES
        self.model = ResNet(num_classes, Bottleneck, [3, 8, 21, 3])

    def forward(self, x):
        out = self.model(x)
        # print("feature_maps:", [list(x.shape[2:]) for x in out])
        # print("out channels:", [x.shape[1] for x in out])
        return out


# Adapted Retina Net from https://github.com/yhenon/pytorch-retinanet

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 128
        
        super().__init__()
        
        # self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        resnet = models.resnet50(pretrained=True)
        resnet_layers = list(resnet.children())

        self.layer1 = nn.Sequential(*resnet_layers[:6])
        self.layer2 = resnet_layers[6]
        self.layer3 = resnet_layers[7]


        #fpn_sizes = [
        #    self.layer2[layers[1] - 1].conv3.out_channels,
        #    self.layer3[layers[2] - 1].conv3.out_channels,
        #    self.layer4[layers[3] - 1].conv3.out_channels
        #]

        # Intrinsic Resnet values, don't change
        fpn_sizes = [512, 1024, 2048]

        self.fpn = PyramidFeatures(*fpn_sizes, feature_size=1024)

        # self.init_weights()

        self.freeze_bn()

    def init_weights(self):

        from math import sqrt

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, sqrt(2 / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)

        #for feature in x1, x2, x3:
        #    print(feature.shape)
        #exit()

        # features = self.fpn([x2, x3, x4])
        features = self.fpn((x1, x2, x3))

        return features

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super().__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        
        P4_x = self.P4_1(C4)
        P3_x = self.P3_1(C3)

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, P4_x.shape[2:])
        P5_x = self.P5_2(P5_x)

        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, P3_x.shape[2:])
        P4_x = self.P4_2(P4_x)

        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]