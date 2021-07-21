import numpy as np
import random
import kimimaro
import matplotlib.pyplot as plt
from utils import segmentation as seg

featureSize = 224
sampleSize = [100,100,100]

class dataGenerator():
    def __init__(self, rawData, groundTruth):
        self.rawData = np.array(rawData)
        self.groundTruth = np.array(groundTruth)
        self.groundTruth_cut = self.groundTruth[int(featureSize/2)+1:520-int(featureSize/2)+1,
                                                int(featureSize/2)+1:520-int(featureSize/2)+1,
                                                int(featureSize/2)+1:520-int(featureSize/2)+1] # 193:327
        self.groundTruth_cut = list(self.groundTruth_cut)
        # self.labelList = np.unique(np.array(self.groundTruth_cut).reshape((1, -1))[0])
        self.skels = kimimaro.skeletonize(
            self.groundTruth_cut,
            teasar_params={
                'scale': 4,
                'const': 500,  # physical units
                'pdrf_exponent': 4,
                'pdrf_scale': 100000,
                'soma_detection_threshold': 1100,  # physical units
                'soma_acceptance_threshold': 3500,  # physical units
                'soma_invalidation_scale': 1.0,
                'soma_invalidation_const': 300,  # physical units
                'max_paths': None,  # default None
            },
            # object_ids=[ ... ], # process only the specified labels
            # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
            # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
            dust_threshold=1000,  # skip connected components with fewer than this many voxels
            anisotropy=(1, 1, 1),  # default True
            fix_branching=True,  # default True
            fix_borders=True,  # default True
            fill_holes=False,  # default False
            fix_avocados=False,  # default False
            progress=True,  # default False, show progress bar
            parallel=1,  # <= 0 all cpu, 1 single process, 2+ multiprocess
            parallel_chunk_size=100,  # how many skeletons to process before updating progress bar
        )
        self.labelList = list(self.skels.keys())

    def generatePositiveDataset(self):
        sample_num = sampleSize[0]
        selectedLabel = random.sample(set(self.labelList), sample_num)
        positiveSamples = {}

        for label in selectedLabel:
            selectedskel = self.skels[label]
            selectedskel_randCenter = selectedskel.vertices[random.randint(0, len(selectedskel.vertices)-1)]
            selectedskel_randCenter = [selectedskel_randCenter[0]+int(featureSize/2)+1, selectedskel_randCenter[1]+int(featureSize/2)+1, selectedskel_randCenter[2]+int(featureSize/2)+1]
            selectedSegmentation = self.groundTruth[int(selectedskel_randCenter[0]-int(featureSize/2)):int(selectedskel_randCenter[0]+int(featureSize/2)),
                                                    int(selectedskel_randCenter[1]-int(featureSize/2)):int(selectedskel_randCenter[1]+int(featureSize/2)),
                                                    int(selectedskel_randCenter[2]-int(featureSize/2)):int(selectedskel_randCenter[2]+int(featureSize/2))]
            correspondingBlock = self.rawData[
                                   int(selectedskel_randCenter[0] - int(featureSize/2)):int(selectedskel_randCenter[0] + int(featureSize/2)),
                                   int(selectedskel_randCenter[1] - int(featureSize/2)):int(selectedskel_randCenter[1] + int(featureSize/2)),
                                   int(selectedskel_randCenter[2] - int(featureSize/2)):int(selectedskel_randCenter[2] + int(featureSize/2))]
            positiveSamples[label] = []
            positiveSamples[label].append(selectedskel_randCenter)
            positiveSamples[label].append(selectedSegmentation)
            positiveSamples[label].append(correspondingBlock)
        return positiveSamples # { label: [[center], [segmentation] ,[correspondingBlock]}

    def generateMergeErrorDataset(self):
        sample_num = sampleSize[1]
        selectedLabel = random.sample(set(self.labelList), sample_num)
        mergeErrSamples = {}
        for label in selectedLabel:
            segment = seg.segmentation(np.array(self.groundTruth_cut))
            neighbors = segment.findNeigSeg(label)
            neighbors_label = list(neighbors.keys())
            if len(neighbors_label) != 0:
                selected_neighbors_label = neighbors_label[random.randint(0, len(neighbors_label)-1)]
                selected_Center = [int(np.average(np.array(neighbors[selected_neighbors_label])[:, 0])),
                                   int(np.average(np.array(neighbors[selected_neighbors_label])[:, 1])),
                                   int(np.average(np.array(neighbors[selected_neighbors_label])[:, 2]))]
                selected_Center = [selected_Center[0]+int(featureSize/2), selected_Center[1]+int(featureSize/2), selected_Center[2]+int(featureSize/2)]
                groundTruth_merged = self.groundTruth
                # groundTruth_merged[groundTruth_merged==selected_neighbors_label] = label
                for z in range(len(groundTruth_merged)):
                    for y in range(len(groundTruth_merged[0])):
                        for x in range(len(groundTruth_merged[0, 0])):
                            if (groundTruth_merged[z,y,x] == selected_neighbors_label):
                                groundTruth_merged[z, y, x] = label
                selectedSegmentation = groundTruth_merged[selected_Center[0]-int(featureSize/2):selected_Center[0]+int(featureSize/2),
                                                          selected_Center[0]-int(featureSize/2):selected_Center[0]+int(featureSize/2),
                                                          selected_Center[0]-int(featureSize/2):selected_Center[0]+int(featureSize/2)]
                correspondingBlock = self.rawData[
                                       int(selected_Center[0] - int(featureSize/2)):int(selected_Center[0] + int(featureSize/2)),
                                       int(selected_Center[1] - int(featureSize/2)):int(selected_Center[1] + int(featureSize/2)),
                                       int(selected_Center[2] - int(featureSize/2)):int(selected_Center[2] + int(featureSize/2))]
                mergeErrSamples[label] = []
                mergedLabels = [label, selected_neighbors_label]
                mergeErrSamples[label].append(mergedLabels)
                mergeErrSamples[label].append(selected_Center)
                mergeErrSamples[label].append(selectedSegmentation)
                mergeErrSamples[label].append(correspondingBlock)
        return mergeErrSamples  # { label: [ [label, selected_neighbors_label], [center], [segmentation], [correspondingBlock]}
    def generateSplitErrorDataset(self):
        sample_num = sampleSize[2]
        selectedLabel = random.sample(set(self.labelList), sample_num)
        splitErrSamples = {}
        for label in selectedLabel:
            # select a random center in skeleton
            selectedskel = self.skels[label]
            selectedskel_randCenter = selectedskel.vertices[random.randint(0, len(selectedskel.vertices)-1)]
            selectedskel_randCenter = [selectedskel_randCenter[0]+int(featureSize/2), selectedskel_randCenter[1]+int(featureSize/2), selectedskel_randCenter[2]+int(featureSize/2)]
            # select a random surface and split a seg by this surface
            randomParaX = random.random()
            randomParaY = random.random()
            randomParaZ = random.random()
            groundTruth_split = self.groundTruth
            for z in range(len(groundTruth_split)):
                for y in range(len(groundTruth_split[0])):
                    for x in range(len(groundTruth_split[0, 0])):
                        if( (groundTruth_split[z,y,x] == label) and
                            ((x-selectedskel_randCenter[2]) * randomParaX + (y-selectedskel_randCenter[1]) * randomParaY + (z-selectedskel_randCenter[0]) * randomParaZ <= 0)):
                            groundTruth_split[z, y, x] = groundTruth_split[z, y, x] + 1
            selectedSegmentation = groundTruth_split[int(selectedskel_randCenter[0]-int(featureSize/2)):int(selectedskel_randCenter[0]+int(featureSize/2)),
                                                    int(selectedskel_randCenter[1]-int(featureSize/2)):int(selectedskel_randCenter[1]+int(featureSize/2)),
                                                    int(selectedskel_randCenter[2]-int(featureSize/2)):int(selectedskel_randCenter[2]+int(featureSize/2))]
            correspondingBlock = self.rawData[int(selectedskel_randCenter[0]-int(featureSize/2)):int(selectedskel_randCenter[0]+int(featureSize/2)),
                                              int(selectedskel_randCenter[1]-int(featureSize/2)):int(selectedskel_randCenter[1]+int(featureSize/2)),
                                              int(selectedskel_randCenter[2]-int(featureSize/2)):int(selectedskel_randCenter[2]+int(featureSize/2))]
            splitErrSamples[label] = []
            splitSurface = [randomParaZ, randomParaX, randomParaY]
            splitErrSamples[label].append(splitSurface)
            splitErrSamples[label].append(selectedskel_randCenter)
            splitErrSamples[label].append(selectedSegmentation)
            splitErrSamples[label].append(correspondingBlock)
        return splitErrSamples  # { label: [ splitSurface, [center], [segmentation], [correspondingBlock]}