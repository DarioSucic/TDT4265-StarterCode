import torch
from torch import nn
import torch.nn.functional as F
from ssd.modeling.box_head.prior_box import PriorBox
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss


class SSDBoxHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = BoxPredictor(cfg)
        self.loss_evaluator = MultiBoxLoss(neg_pos_ratio=cfg.MODEL.NEG_POS_RATIO)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

    def forward(self, features, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets["boxes"], targets["labels"]
        reg_loss, cls_loss = self.loss_evaluator(
            cls_logits, bbox_pred, gt_labels, gt_boxes
        )
        loss_dict = dict(reg_loss=reg_loss, cls_loss=cls_loss,)
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred,
            self.priors,
            self.cfg.MODEL.CENTER_VARIANCE,
            self.cfg.MODEL.SIZE_VARIANCE,
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections)
        return detections, {}


class BoxPredictor(nn.Module):
    """
    The class responsible for generating predictions for each prior
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.boxes_per_loc = cfg.MODEL.PRIORS.BOXES_PER_LOCATION
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.cls_models = nn.ModuleList()
        self.reg_models = nn.ModuleList()

        out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        for bpl, n_out in zip(self.boxes_per_loc, out_channels):
            self.cls_models.append(self.cls_block(n_out, bpl))
            self.reg_models.append(self.reg_block(n_out, bpl))

        self.init_parameters()

    def cls_block(self, out_channels, boxes_per_location):
        """
        return nn.Conv2d(
            out_channels,
            boxes_per_location * self.cfg.MODEL.NUM_CLASSES,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        """
        return ClassificationModel(
            out_channels,
            num_classes=self.num_classes,
            num_anchors=boxes_per_location
        )

    def reg_block(self, out_channels, boxes_per_location):
        #return nn.Conv2d(
        #    out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1
        #)
        return RegressionModel(
            out_channels,
            num_anchors=boxes_per_location
        )

    def init_parameters(self):

        from math import log, sqrt

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, std=sqrt(2/n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.eval()

        prior = 0.01
        p_bias = -log((1.0 - prior) / prior)

        for m in self.cls_models:
            nn.init.zeros_(m.output.weight)
            nn.init.constant_(m.output.bias, p_bias)

        for m in self.reg_models:
            nn.init.zeros_(m.output.weight)
            nn.init.zeros_(m.output.bias)


    def forward(self, features):

        cls_logits, bbox_preds = [], []
        for feature, cls_model, reg_model in zip(features, self.cls_models, self.reg_models):
            cls_logits.append(cls_model(feature))
            bbox_preds.append(reg_model(feature))

        batch_size = features[0].shape[0]
        cls_logits = torch.cat(
            [c.view(c.shape[0], -1) for c in cls_logits],
            dim=1).view(batch_size, -1, self.num_classes)
        bbox_preds = torch.cat(
            [l.view(l.shape[0], -1) for l in bbox_preds],
            dim=1).view(batch_size, -1, 4)

        return cls_logits, bbox_preds

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, feature_size=64):
        super().__init__()

        #self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        #self.act1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act2 = nn.ReLU(inplace=True)

        # self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act3 = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act4 = nn.ReLU(inplace=True)

        feature_size = num_features_in
        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):

        out = x

        #out = self.conv1(out)
        #out = self.act1(out)

        # out = self.conv2(out)
        # out = self.act2(out)

        # out = self.conv3(out)
        # out = self.act3(out)

        # out = self.conv4(out)
        # out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous()


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, num_classes, prior=0.01, feature_size=256):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        # self.act1 = nn.ReLU(inplace=True)

        # self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.act2 = nn.ReLU(inplace=True)

        #self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.act3 = nn.ReLU(inplace=True)

        #self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        #self.act4 = nn.ReLU(inplace=True)

        feature_size = num_features_in
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        #self.output_act = nn.Sigmoid()

    def forward(self, x):
        
        out = x
        
        #out = self.conv1(out)
        #out = self.act1(out)

        # out = self.conv2(out)
        # out = self.act2(out)

        #out = self.conv3(out)
        #out = self.act3(out)

        #out = self.conv4(out)
        #out = self.act4(out)
        
        out = self.output(out)
        #out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous()