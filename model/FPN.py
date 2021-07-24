import torch

from model.anchors_utils import generate_anchors


class FPN():
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

        def get_anchors(self):
            # TODO:这个16是原图到feature map缩小的倍数,提出来？ # 如何断定原图到feature map的倍数为16？ 中间的VGG层确定了？
            height_of_feature_map = torch.to_int32(torch.ceil(self.image_info[0] / torch.to_float(16)))
            width_of_feature_map = torch.to_int32(torch.ceil(self.image_info[1] / torch.to_float(16)))
            self.anchors, self.num_of_anchors = generate_anchors(height_of_feature_map, width_of_feature_map, 16,
                                                                 ANCHOR_SCALES,
                                                                 ANCHOR_RATIOS)

        def build(self, is_training):
            self.get_anchors()
            self.region_proposal(is_training)
            self.roi_pooling()
            self.head_to_tail(is_training)
            self.region_classification(is_training)
            if is_training:
                self.add_losses()