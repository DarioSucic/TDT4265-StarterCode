import torch
from math import sqrt
from itertools import product


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, (fx, fy) in enumerate(self.feature_maps):

            # TODO: Make strides 2d as well

            x_scale = self.image_size[0] / self.strides[k]
            y_scale = self.image_size[1] / self.strides[k]

            #scale = self.image_size / self.strides[k]
            
            # for i, j in product(range(f), repeat=2):
            for i, j in product(range(fx), range(fy)):
                # unit center x,y
                cx = (j + 0.5) / x_scale
                cy = (i + 0.5) / y_scale

                # small sized square box
                size = self.min_sizes[k]
                
                # h = w = size / self.image_size
                w = size / self.image_size[0]
                h = size / self.image_size[1]
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                # h = w = size / self.image_size
                w = size / self.image_size[0]
                h = size / self.image_size[1]
                
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                # h = w = size / self.image_size
                w = size / self.image_size[0]
                h = size / self.image_size[1]
                
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
