import torch


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

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
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[0], kernel_size=3, stride=2, padding=1))

        self.feature_extractor_1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[0]),
            torch.nn.Conv2d(in_channels=self.output_channels[0], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[1], kernel_size=3, stride=2, padding=1))
        
        self.feature_extractor_2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[1]),
            torch.nn.Conv2d(in_channels=self.output_channels[1], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[2], kernel_size=3, stride=2, padding=1))

        self.feature_extractor_3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[2]),
            torch.nn.Conv2d(in_channels=self.output_channels[2], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[3], kernel_size=3, stride=2, padding=1))

        self.feature_extractor_4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(self.output_channels[3]),
            torch.nn.Conv2d(in_channels=self.output_channels[3], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[4], kernel_size=3, stride=2, padding=1))

        self.feature_extractor_5 = torch.nn.Sequential(    
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.output_channels[4], out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1024),
            torch.nn.Conv2d(in_channels=1024, out_channels=self.output_channels[5], kernel_size=3, stride=1, padding=0))

        self.features = torch.nn.ModuleList([
            self.feature_extractor_0, self.feature_extractor_1, self.feature_extractor_2, self.feature_extractor_3, self.feature_extractor_4, self.feature_extractor_5  
        ])

    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for feature in self.features:
            x = feature(x)
            out_features.append(x)
        
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], self.output_feature_size[idx], self.output_feature_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        

        return tuple(out_features)

