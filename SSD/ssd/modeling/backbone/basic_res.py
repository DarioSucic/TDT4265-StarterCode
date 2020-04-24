
import torch
import torch.nn as nn
from torchvision.models import resnet34

class BasicModel_(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_size = cfg.INPUT.IMAGE_SIZE
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        resnet = resnet34(pretrained=True)

        self.resnet = resnet
        resnet_layers = list(resnet.children())
        
        extra_layers = [
            # TODO
        ]

        self.features = torch.nn.ModuleList([
            nn.Sequential(*resnet_layers[0:6]),
            *resnet_layers[6:9],
            *extra_layers
        ])


    def forward(self, x):

        # print(0, x.shape)

        out_features = []
        for i, feature in enumerate(self.features, 1):
            x = feature(x)
            # print(i, x.shape)
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