import numpy as np

class segmentation():
    def __init__(self, volume):
        self.volume = volume

    def searchNeighbor(self, z,y,x):
        neiList = []
        for z_bias in range(-1, 2):
            for y_bias in range(-1, 2):
                for x_bias in range(-1, 2):
                    if self.volume[z+z_bias, y+y_bias, x+x_bias] != self.volume[z_bias,y_bias,x_bias]:
                        val = self.volume[z + z_bias, y + y_bias, x + x_bias]
                        neiList.append(val)
        neiList = np.unique(np.array(neiList))
        return neiList


    def findNeigSeg(self, selectedLabel):
        allNeighbors = []
        for z in range(len(self.volume)):
            for y in range(len(self.volume[0])):
                for x in range(len(self.volume[0, 0])):
                    if self.volume[z,y,x] == selectedLabel:
                        neiList = self.searchNeighbor(z,y,x)
                        if len(neiList):
                            allNeighbors = np.append(allNeighbors, neiList)
                            allNeighbors = np.unique(allNeighbors)
        return allNeighbors
