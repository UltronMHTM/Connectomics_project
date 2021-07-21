import numpy as np

class segmentation():
    def __init__(self, volume):
        self.volume = volume

    def searchNeighbor(self, z,y,x):
        neiDict = {}
        for z_bias in range(-1, 2):
            for y_bias in range(-1, 2):
                for x_bias in range(-1, 2):
                    if (z+z_bias)<0 or (z+z_bias)>=len(self.volume) or (y+y_bias)<0 or (y+y_bias)>=len(self.volume[0]) or (x+x_bias)<0 or (x+x_bias)>=len(self.volume[0,0]):
                        continue
                    if self.volume[z+z_bias, y+y_bias, x+x_bias] != self.volume[z,y,x]:
                        val = self.volume[z + z_bias, y + y_bias, x + x_bias]
                        if val not in neiDict.keys():
                            neiDict[val] = []
                        neiDict[val].append([z+z_bias, y+y_bias, x+x_bias])
        return neiDict


    def findNeigSeg(self, selectedLabel):
        allNeighbors = {}
        for z in range(len(self.volume)):
            for y in range(len(self.volume[0])):
                for x in range(len(self.volume[0, 0])):
                    if self.volume[z,y,x] == selectedLabel:
                        neiDict = self.searchNeighbor(z,y,x)
                        if len(neiDict):
                            for label in neiDict.keys():
                                if label in allNeighbors.keys():
                                    allNeighbors[label] = np.vstack((allNeighbors[label], neiDict[label]))
                                    allNeighbors[label] = list(set(tuple(t) for t in allNeighbors[label]))
                                    allNeighbors[label] = list(list(i) for i in allNeighbors[label])
                                else:
                                    allNeighbors[label] = neiDict[label]
        return allNeighbors
