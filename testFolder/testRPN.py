import unittest
from model.FPN import FPN
import torch
import numpy as np

class Test_segmentation(unittest.TestCase):
    def test_get_anchors_region_proposal(self):
        FPNnet = FPN(torch.rand(1, 1, 14, 14, 56), np.zeros((3, 7), dtype=np.float32), [224, 224, 224, 1])
        FPNnet.get_anchors()
        FPNnet.region_proposal(True, torch.rand(1, 18, 14, 14, 56), torch.rand(1, 36, 14, 14, 56))
        # print(FPNnet.anchors.size(), FPNnet.num_of_anchors)
        # self.assertTrue(set(searchNeighbor_res) == set(pred_res))
if __name__ == '__main__':
    unittest.main()