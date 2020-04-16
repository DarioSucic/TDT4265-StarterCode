
import torch
import torch.nn as nn
from torchvision.models import resnet50

class BasicModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_size = cfg.INPUT.IMAGE_SIZE
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        resnet = resnet50(pretrained=True)

        self.resnet = nn.Sequential(*list(resnet.children())[:7])
        self.resnet[-1][0].conv1.stride = (1, 1)
        self.resnet[-1][0].conv2.stride = (1, 1)
        self.resnet[-1][0].downsample[0].stride = (1, 1)

        image_channels = 1024

        self.feature_extractor_0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[0], kernel_size=3, stride=1, padding=1))

        self.feature_extractor_1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            torch.nn.Conv2d(in_channels=self.output_channels[0], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[1], kernel_size=3, stride=1, padding=1))
        
        self.feature_extractor_2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            torch.nn.Conv2d(in_channels=self.output_channels[1], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[2], kernel_size=3, stride=1, padding=1))

        self.feature_extractor_3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            torch.nn.Conv2d(in_channels=self.output_channels[2], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[3], kernel_size=3, stride=1, padding=1))

        self.feature_extractor_4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            torch.nn.Conv2d(in_channels=self.output_channels[3], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[4], kernel_size=3, stride=1, padding=1))

        self.feature_extractor_5 = torch.nn.Sequential(    
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.output_channels[4], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[5], kernel_size=3, stride=1, padding=0))

        self.features = torch.nn.ModuleList([
            self.feature_extractor_0,
            self.feature_extractor_1,
            self.feature_extractor_2,
            self.feature_extractor_3,
            self.feature_extractor_4,
            self.feature_extractor_5
        ])


    def forward(self, x):
        x = self.resnet(x)

        #print("x dim after resnet :DDDD", x.shape)

        out_features = []
        for i, feature in enumerate(self.features):
            x = feature(x)
            #print(f"shape after iteration {i:2}: {x.shape}")
            out_features.append(x)
        return out_features


"""
class BasicModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
    
    def forward(self, x):
        out_features = []
        for idx, feature in enumerate(out_features):
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
"""