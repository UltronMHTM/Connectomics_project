import unittest
from ..utils import segmentation as seg
import numpy as np

class Test_segmentation(unittest.TestCase):

    def test_searchNeighbor(self):
        testPara = np.array([[[1,1,1],[2,2,2],[3,3,3]],[[1,1,1],[2,2,2],[3,3,3]],[[1,1,1],[2,2,2],[3,3,3]]])
        segment = seg.segmentation(testPara)
        searchNeighbor_res = segment.searchNeighbor(1,1,1)
        pred_res = {1: [[0, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2]],
                    3: [[0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]}
        self.assertTrue(searchNeighbor_res == pred_res)

    def test_findNeigSeg(self):
        testPara = np.array([[[1,1,1],[2,2,2],[3,3,3],[2,2,2]],[[1,1,1],[2,2,2],[3,3,3],[2,2,2]],[[1,1,1],[2,2,2],[3,3,3],[2,2,2]]])
        segment = seg.segmentation(testPara)
        searchNeighbor_res = segment.findNeigSeg(1)
        pred_res = {2: [[0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2]]}
        self.assertTrue(set(searchNeighbor_res) == set(pred_res))

if __name__ == '__main__':
    unittest.main()