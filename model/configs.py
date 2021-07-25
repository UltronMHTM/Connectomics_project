# anchor的放大倍数，原始尺寸为16*16
ANCHOR_SCALES = [8, 16, 32]

# anchor的长宽比 x, y, z
ANCHOR_RATIOS = [[1.0,1.0,0.5], [1.0,0.5,1.0], [1.0,0.5,0.5]]

# 分类阶段输入的rois数目
BATCH_SIZE = 2

# 非背景比例
FG_FRACTION = 0.25

# 训练时，若当前ROI对应的概率大于0.5，认为它是前景
FG_THRESH = 0.5