import torch
import torch.nn as nn
from model.anchors_utils import generate_anchors
from model.configs import *

class FPN(nn.Module):
    def __init__(self, feature_map, gt_boxes):
        self.feature_map = feature_map
        self.gt_boxes = gt_boxes
        #

        self._predictions = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._losses = {}
        self._num_anchor_types = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self._num_classes = len(CLASSES)

        self.rpn = nn.Conv3d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1).cuda()
        self.relu = nn.ReLU().cuda()
        self.scorChal_conv = nn.Conv3d(in_channels=512, out_channels=self._num_anchor_types * 2, kernel_size=1, stride=1, padding=0).cuda()
        self.bboxChal_conv = nn.Conv3d(in_channels=512, out_channels=self._num_anchor_types * 4, kernel_size=1, stride=1, padding=0).cuda()
    def get_anchors(self):
        # TODO:这个16是原图到feature map缩小的倍数,提出来？ # 如何断定原图到feature map的倍数为16？ 中间的VGG层确定了？
        height_of_feature_map = torch.to_int32(torch.ceil(self.image_info[0] / torch.to_float(16)))
        width_of_feature_map = torch.to_int32(torch.ceil(self.image_info[1] / torch.to_float(16)))
        self.anchors, self.num_of_anchors = generate_anchors(height_of_feature_map, width_of_feature_map, 16,
                                                             ANCHOR_SCALES,
                                                             ANCHOR_RATIOS)
    def region_proposal(self, is_training):
        def _reshape_layer(bottom, num_dim, name):
            input_shape = torch.shape(bottom)  # 1*H*W*18
            with torch.variable_scope(name) as scope:
                to_caffe = torch.transpose(bottom, [0, 3, 1, 2])  # 1*18*H*W
                reshaped = torch.reshape(to_caffe,
                                      torch.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))  # 1*2*9H*W
                to_tf = torch.transpose(reshaped, [0, 2, 3, 1])  # 1*9H*W*2
                return to_tf

        def _softmax_layer(bottom, name):
            if name.startswith('rpn_cls_prob_reshape'):
                input_shape = torch.shape(bottom)  # 1*9H*W*2     # probability of A and the probability of B
                bottom_reshaped = torch.reshape(bottom, [-1, input_shape[-1]])  # 9HW*2
                reshaped_score = torch.nn.softmax(bottom_reshaped, name=name)
                return torch.reshape(reshaped_score, input_shape)  # 1*9H*W*2
            return torch.nn.softmax(bottom, name=name)


    def build(self, is_training):
        self.get_anchors()
        self.region_proposal(is_training)
        self.roi_pooling()
        self.head_to_tail(is_training)
        self.region_classification(is_training)
        if is_training:
            self.add_losses()

    def forward(self, input):
        output = self.rpn(input)
        output = self.relu(output)
        output = self.scorChal_conv(output)
        output = torch.argmax(torch.reshape(output, [-1, 2]), axis=1, name="rpn_cls_pred")

