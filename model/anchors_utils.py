import numpy as np
import torch

def _get_w_h_ctrs(anchor):
    """
    返回窗口的宽，高，长，中心点x，中心点y，中心点z
    """
    w = anchor[3] - anchor[0] + 1
    h = anchor[4] - anchor[1] + 1
    l = anchor[5] - anchor[2] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    z_ctr = anchor[2] + 0.5 * (h - 1)
    return w, h, l, x_ctr, y_ctr, z_ctr

def _make_anchors(ws, hs, ls, x_ctr, y_ctr, z_ctr):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    ls = ls[:, np.newaxis]
    # 生成的anchor为np数组
    # anchor坐标为左上xy，右下xy
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         z_ctr - 0.5 * (ls - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1),
                         z_ctr + 0.5 * (ls - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    w, h, l, x_ctr, y_ctr, z_ctr = _get_w_h_ctrs(anchor)
    ws = np.round(w * ratios[0])
    hs = np.round(h * ratios[1])
    ls = np.round(l * ratios[2])
    anchors = _make_anchors(ws, hs, ls, x_ctr, y_ctr, z_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """
    # 缩放枚举
    w, h, l, x_ctr, y_ctr, z_ctr = _get_w_h_ctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
    return anchors

def get_basic_anchors(anchor_ratios, anchor_scales, base_size=16):
    base_anchor = np.array([1, 1, 1, base_size, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, anchor_ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales) for i in range(ratio_anchors.shape[0])])
    return anchors

def generate_anchors(featureMapDim, stride_of_feature_map, anchor_scales, anchor_ratios):
    shift_x = range(featureMapDim[0]) * stride_of_feature_map[0]
    shift_y = range(featureMapDim[1]) * stride_of_feature_map[1]
    shift_z = range(featureMapDim[2]) * stride_of_feature_map[2]
    shift_x, shift_y, shift_z = torch.meshgrid(shift_x, shift_y, shift_z)
    sx = torch.reshape(shift_x, shape=(-1,))
    sy = torch.reshape(shift_y, shape=(-1,))
    sz = torch.reshape(shift_z, shape=(-1,))
    #
    shift_allDim = torch.stack(sx, sy, sz, sx, sy, sz, sx, sy, sz)
    num_of_feature_map_pixels = featureMapDim[0]*featureMapDim[1]*featureMapDim[2]
    #
    basic_anchors = get_basic_anchors(anchor_ratios=np.array(anchor_ratios), anchor_scales=np.array(anchor_scales))
    num_of_basic_anchors = basic_anchors.shape[0]
    total_num_of_anchors = num_of_basic_anchors * num_of_feature_map_pixels
    anchors = torch.add(basic_anchors, shift_allDim)
    return anchors, total_num_of_anchors

