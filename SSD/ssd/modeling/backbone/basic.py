
import torch
import torch.nn as nn
import torchvision.models as models

from itertools import repeat

class BasicModel(nn.Module):

    def __init__(self, cfg):

        super().__init__()
        
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_size = cfg.INPUT.IMAGE_SIZE
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        backbone = models.resnet50(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

        self._build_additional_features(output_channels)
        
        self._init_weights()

    def _build_additional_features(self, out_channels):
        self.additional_blocks = []

        in_size = out_channels[0]
        for i, out_size in enumerate(out_channels[1:]):
            self.additional_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_size, out_size, kernel_size=3, bias=False, **({"padding": 1, "stride": 2} if True else {})),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                )
            )
            in_size = out_size

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        for layer in self.additional_blocks:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.feature_extractor(x)

        # print(x.shape)
        
        features = [x]
        for block in self.additional_blocks:
            x = block(x)
            features.append(x)
            # print(x.shape)

        return features
