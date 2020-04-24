
import torch
import torch.nn as nn
import torchvision.models as models

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
        self.feature_extractor.out_channels = [1024, 2048, 1024, 1024, 512, 512]

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

        self._build_additional_features(self.feature_extractor.out_channels)
        
        self._init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [2048, 1024, 1024, 512, 512])):
            self.additional_blocks.append(nn.Sequential(
                nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, output_size, kernel_size=3, bias=False, **({"padding": 1, "stride": 2} if i < 3 else {})),
                nn.BatchNorm2d(output_size),
                nn.ReLU(inplace=True),
            ))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        for layer in self.additional_blocks:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x.shape)

        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            # print(x.shape)
            detection_feed.append(x)

        return detection_feed
