import torch
import torch.nn as nn
from model.anchors_utils import generate_anchors
from  model.target_layer import *
from model.configs import *
from torch.nn import functional as Fun

class FPN(nn.Module):
    def __init__(self, feature_map, gt_boxes, image_info = [224, 224, 224, 1]):
        super(FPN, self).__init__()
        self.feature_map = feature_map
        self.gt_boxes = gt_boxes
        self.image_info = image_info
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
        self.bboxChal_conv = nn.Conv3d(in_channels=512, out_channels=self._num_anchor_types * 6, kernel_size=1, stride=1, padding=0).cuda()  # 1, _num_anchor_types * 6, 14, 14, 56
        self.Pooling = nn.MaxPool3d(kernel_size=2, stride=2).cuda()
        self.fc6 = nn.Linear(4096, 4096).cuda()
        self.fc6 = nn.Linear(4096, 4096).cuda()

    def get_anchors(self):
        # TODO:这个16是原图到feature map缩小的倍数,提出来？ # 如何断定原图到feature map的倍数为16？ 中间的VGG层确定了？
        height_of_feature_map = torch.ceil(torch.tensor(self.image_info[0] / (16.0)))
        width_of_feature_map = torch.ceil(torch.tensor(self.image_info[1] / (16.0)))
        length_of_feature_map = torch.ceil(torch.tensor(self.image_info[2] / (4.0)))
        self.anchors, self.num_of_anchors = generate_anchors([height_of_feature_map, width_of_feature_map, length_of_feature_map], [16,16,4],
                                                             ANCHOR_SCALES,
                                                             ANCHOR_RATIOS)

    def get_rois(self, rpn_cls_prob, rpn_bbox_pred, output_size):

        def bbox_transform_inv_tf(boxes, deltas):  # possible anchor be push at the right top first, then redefine their positions
            # boxes:生成的anchor
            # deltas:包围框偏移量 [1*height*width*anchor_num,4]
            # boxes = tf.cast(boxes, deltas.dtype)
            # widths = tf.subtract(boxes[:, 3], boxes[:, 0]) + 1.0   # left top and right bottom
            # heights = tf.subtract(boxes[:, 4], boxes[:, 1]) + 1.0
            # lengths = tf.subtract(boxes[:, 5], boxes[:, 2]) + 1.0
            widths = boxes[:, 3] - boxes[:, 0] + 1.0   # left top and right bottom
            heights = boxes[:, 4] - boxes[:, 1] + 1.0
            lengths = boxes[:, 5] - boxes[:, 2] + 1.0
            ctr_x = torch.add(boxes[:, 0], widths * 0.5)
            ctr_y = torch.add(boxes[:, 1], heights * 0.5)
            ctr_z = torch.add(boxes[:, 2], lengths * 0.5)

            # 获取包围框预测结果[dx,dy,dw,dh](精修？)
            dx = deltas[:, 0]
            dy = deltas[:, 1]
            dz = deltas[:, 2]
            dw = deltas[:, 3]
            dh = deltas[:, 4]
            dl = deltas[:, 5]

            # tf.multiply对应元素相乘
            # pre_x = dx * w + ctr_x
            # pre_y = dy * h + ctr_y
            # pre_w = e**dw * w
            # pre_h = e**dh * h
            pred_ctr_x = torch.add(torch.mul(dx, widths), ctr_x)   # reflect they back?
            pred_ctr_y = torch.add(torch.mul(dy, heights), ctr_y)
            pred_ctr_z = torch.add(torch.mul(dz, lengths), ctr_z)
            pred_w = torch.mul(torch.exp(dw), widths)
            pred_h = torch.mul(torch.exp(dh), heights)
            pred_l = torch.mul(torch.exp(dl), lengths)

            # 将坐标转换为（xmin,ymin,xmax,ymax）格式
            # pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
            # pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
            # pred_boxes2 = tf.subtract(pred_ctr_z, pred_l * 0.5)
            pred_boxes0 = pred_ctr_x - pred_w * 0.5
            pred_boxes1 = pred_ctr_y - pred_h * 0.5
            pred_boxes2 = pred_ctr_z - pred_l * 0.5
            pred_boxes3 = torch.add(pred_ctr_x, pred_w * 0.5)
            pred_boxes4 = torch.add(pred_ctr_y, pred_h * 0.5)
            pred_boxes5 = torch.add(pred_ctr_z, pred_l * 0.5)

            # 叠加结果输出
            return torch.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4, pred_boxes5], axis=1)

        def clip_boxes_tf(boxes, im_info):
            # 按照图像大小裁剪boxes
            # b0 = torch.maximum(torch.minimum(boxes[:, 0], im_info[2] - 1), 0)   # ????????
            # b1 = torch.maximum(torch.minimum(boxes[:, 1], im_info[1] - 1), 0)
            # b2 = torch.maximum(torch.minimum(boxes[:, 2], im_info[0] - 1), 0)
            # b3 = torch.maximum(torch.minimum(boxes[:, 3], im_info[2] - 1), 0)
            # b4 = torch.maximum(torch.minimum(boxes[:, 4], im_info[1] - 1), 0)
            # b5 = torch.maximum(torch.minimum(boxes[:, 5], im_info[0] - 1), 0)
            b0 = torch.max(torch.cat((torch.min(
                torch.cat((boxes[:, 0], (im_info[2] - 1) * torch.ones([len(boxes), 1])), 1), 1),
                                      torch.zeros([len(boxes), 1])), 1), 1)
            b1 = torch.max(torch.cat((torch.min(
                torch.cat((boxes[:, 1], (im_info[1] - 1) * torch.ones([len(boxes), 1])), 1), 1),
                                      torch.zeros([len(boxes), 1])), 1), 1)
            b2 = torch.max(torch.cat((torch.min(
                torch.cat((boxes[:, 2], (im_info[0] - 1) * torch.ones([len(boxes), 1])), 1), 1),
                                      torch.zeros([len(boxes), 1])), 1), 1)
            b3 = torch.max(torch.cat((torch.min(
                torch.cat((boxes[:, 3], (im_info[2] - 1) * torch.ones([len(boxes), 1])), 1), 1),
                                      torch.zeros([len(boxes), 1])), 1), 1)
            b4 = torch.max(torch.cat((torch.min(
                torch.cat((boxes[:, 4], (im_info[1] - 1) * torch.ones([len(boxes), 1])), 1), 1),
                                      torch.zeros([len(boxes), 1])), 1), 1)
            b5 = torch.max(torch.cat((torch.min(
                torch.cat((boxes[:, 5], (im_info[0] - 1) * torch.ones([len(boxes), 1])), 1), 1),
                                      torch.zeros([len(boxes), 1])), 1), 1)
            return torch.stack([b0, b1, b2, b3, b4, b5], axis=1)

        # scores = rpn_cls_prob[:, :, :, self._num_anchor_types:]     # why cut the front
        scores = rpn_cls_prob[:, self._num_anchor_types:, :, :, :]
        scores = torch.reshape(scores, shape=(-1,))
        rpn_bbox_pred = torch.reshape(rpn_bbox_pred, shape=(-1, 6))    # reshape for X,Y,H,W

        proposals = bbox_transform_inv_tf(self.anchors, rpn_bbox_pred)
        proposals = clip_boxes_tf(proposals, self.image_info[:2])
        indices = tf.image.non_max_suppression(proposals, scores, max_output_size=output_size, iou_threshold=0.7)
        boxes = torch.gather(proposals, indices)
        # boxes = tf.to_float(boxes)
        scores = torch.gather(scores, indices)
        scores = torch.reshape(scores, shape=(-1, 1))
        # TODO:在每个indices前加入batch内索引，由于目前仅支持每个batch一张图像作为输入所以均为0
        batch_inds = torch.zeros((torch.shape(indices)[0], 1), dtype=torch.float32)
        rois = torch.cat([batch_inds, boxes], 1)
        rois.set_shape([None, 7])
        scores.set_shape([None, 1])
        return rois, scores

    def _proposal_target_layer(self, rois, roi_scores, name):
        with torch.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = proposal_target_layer(rois, roi_scores, self.gt_boxes, self._num_classes)
                # [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                # name="proposal_target")
            rois.set_shape([128, 7])
            roi_scores.set_shape([128])
            labels.set_shape([128, 1])
            bbox_targets.set_shape([128, self._num_classes * 6])
            bbox_inside_weights.set_shape([128, self._num_classes * 6])
            bbox_outside_weights.set_shape([128, self._num_classes * 6])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = torch.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            return rois, roi_scores

    def region_proposal(self, is_training, rpn_class_score, rpn_bbox_pred):
        def _reshape_layer(bottom, num_dim, name):
            # input_shape = torch.shape(bottom)  # 1*H*W*18
            input_shape = bottom.size()
            # with torch.variable_scope(name) as scope:
            # to_caffe = torch.transpose(bottom, [0, 3, 1, 2])  # 1*18*H*W
            reshaped = torch.reshape(bottom,
                                     torch.cat(axis=0, values=[[1, num_dim, -1], [input_shape[3], input_shape[4]]]))  # 1, 2, 9*h, w, l  # 1*2*9H*W
            # to_tf = torch.transpose(reshaped, [0, 2, 3, 1])  # 1*9H*W*2
            to_tf = reshaped.permute(0, 2, 3, 4, 1)  # 1*9H*W*2
            return to_tf
        def _reshape_layer_back(bottom, num_dim, name):
            input_shape = bottom.size() # 1, 9*h, w, l, 2
            bottom = bottom.permute(0, 4, 1, 2, 3) # 1, 2, 9*h, w, l
            reshaped = torch.reshape(bottom,
                                     torch.cat(axis=0, values=[[1, num_dim, -1], [input_shape[3], input_shape[4]]]))  # 1, 18, h, w, l
            return reshaped

        def _softmax_layer(bottom, name):
            # if name.startswith('rpn_cls_prob_reshape'):
            # input_shape = torch.shape(bottom)  # 1*9H*W*2     # probability of A and the probability of B
            input_shape = bottom.size()
            bottom_reshaped = torch.reshape(bottom, [-1, input_shape[-1]])  # 9HW*2
            # reshaped_score = torch.nn.softmax(bottom_reshaped, name=name)
            reshaped_score = Fun.softmax(bottom_reshaped, dim=1)
            return torch.reshape(reshaped_score, input_shape)  # 1*9H*W*2
            # return torch.nn.softmax(bottom, name=name)

        def _anchor_target_layer(self, rpn_class_score, name):
            # with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(rpn_class_score, self.gt_boxes, self.image_info, 16, self.anchors, self._num_anchor_types0)
            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchor_types * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchor_types * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchor_types * 4])
            # rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
            return rpn_labels

        # convolution and relu first
        # rpn = tf.layers.conv2d(inputs=self.feature_map, filters=512, kernel_size=[3, 3], padding='SAME',
        #                        trainable=is_training)
        # rpn_class_score = tf.layers.conv2d(rpn, self._num_anchor_types * 2, [1, 1], trainable=is_training)  # 1*H*W*18
        # reshape, then detect the class score in each pixel [coarse detection]
        rpn_cls_score_reshape = _reshape_layer(rpn_class_score, 2, 'rpn_cls_score_reshape') # 1, 9*h, w, l, 2
        rpn_cls_prob_reshape = _softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_pred = torch.argmax(torch.reshape(rpn_cls_score_reshape, [-1, 2]), axis=0)  # 9hw*2
        rpn_cls_prob = _reshape_layer_back(rpn_cls_prob_reshape, self._num_anchor_types * 2, "rpn_cls_prob")  # 1*h*w*18

        # convolution in bounding box channel
        # rpn_bbox_pred = tf.layers.conv2d(rpn, self._num_anchor_types * 4, [1, 1], trainable=is_training)  # 1*h*w*36 -> 4*9
        if is_training:  # TODO:test300,train2000。提取
            # refine the region and detect the target
            rois, scores = self.get_rois(rpn_cls_prob, rpn_bbox_pred,
                                         2000)  # draw coarse bounding box, filter the front 2000 boxes
            rpn_labels = self._anchor_target_layer(rpn_class_score,
                                                   "anchor")  # filter some proposal through some predefined thresholds
            # with torch.control_dependencies([rpn_labels]):
            #     # trace back the target
            rois, _ = self._proposal_target_layer(rois, scores,
                                                  "rpn_rois")  # according to these bounding box, generate the data will be trained
        else:
            rois, _ = self.get_rois(rpn_cls_prob, rpn_bbox_pred, 300)

        self._predictions["rpn_cls_score"] = rpn_class_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois
        self._rois = rois

    def roi_pooling(self):
        batch_ids = torch.squeeze(torch.slice(self._rois, [0, 0], [-1, 1], name="batch_id"), [1])
        # 获取包围框归一化后的坐标系（坐标与原图比例）
        bottom_shape = tf.shape(self.feature_map)
        height = (tf.to_float(bottom_shape[1]) - 1.) * tf.to_float(16)  # TODO:
        width = (tf.to_float(bottom_shape[2]) - 1.) * tf.to_float(16)
        x1 = tf.slice(self._rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(self._rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(self._rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(self._rois, [0, 4], [-1, 1], name="y2") / height

        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        pre_pool_size = 7 * 2
        crops = tf.image.crop_and_resize(self.feature_map, bboxes, tf.to_int32(batch_ids),
                                         [pre_pool_size, pre_pool_size],
                                         name="crops")

        self._pool = tf.layers.max_pooling2d(crops, [2, 2], 2, 'same')

    def head_to_tail(self, is_training):    # only 2 full connected? where is relu?
        pool_flatten = tf.layers.flatten(self._pool)
        fc6 = tf.layers.dense(pool_flatten, 4096)
        if is_training:
            fc6 = tf.layers.dropout(fc6, rate=0.5, training=True)
        self._fc7 = tf.layers.dense(fc6, 4096)
        if is_training:
            self._fc7 = tf.layers.dropout(self._fc7, rate=0.5, training=True)

    def region_classification(self, is_training):

        # use softmax to get each classes probability
        cls_score = tf.layers.dense(self._fc7, self._num_classes, trainable=is_training)
        cls_prob = tf.nn.softmax(cls_score)

        # use full convolutional layer to generate precise bounding box
        cls_pred = tf.argmax(cls_score, axis=1)
        bbox_pred = tf.layers.dense(self._fc7, self._num_classes * 4, trainable=is_training)

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

    def build(self, is_training):
        self.get_anchors()
        # self.region_proposal(is_training)
        self.roi_pooling()
        self.head_to_tail(is_training)
        self.region_classification(is_training)
        if is_training:
            self.add_losses()

    def forward(self, input, is_training):
        rpn = self.rpn(input)
        rpn = self.relu(rpn)
        rpn_class_score = self.scorChal_conv(rpn) # 1, 2*anchor_type, 14, 14, 56
        rpn_bbox_pred = self.bboxChal_conv(rpn)   # 1, 6*anchor_type, 14, 14, 56
        self.region_proposal(self, is_training, rpn_class_score, rpn_bbox_pred)

#         here we get the result cropped from feature map
        pooling_res = self.Pooling(crops)
        pooling_res = torch.flatten(pooling_res)


