import h5py
from dataGenerator import dataGenerator
import numpy as np
rawData = h5py.File('Dataset1_validation/grayscale_maps.h5','r')["raw"]
groundTruth = h5py.File('Dataset1_validation/groundtruth.h5','r')["stack"]

generator = dataGenerator(list(rawData), list(groundTruth))

positiveDataset = generator.generatePositiveDataset()
mergeErrorDataset = generator.generateMergeErrorDataset()
splitErrorDataset = generator.generateSplitErrorDataset()

positiveLabel = []
positiveCenters = []
positiveSegmentation = []
positiveRawData = []
for label in positiveDataset.keys():
    positiveLabel.append(label)
    positiveCenters.append(positiveDataset[label][0])
    positiveSegmentation.append(positiveDataset[label][1])
    positiveRawData.append(positiveDataset[label][2])
print(len(positiveCenters))
print(len(positiveSegmentation))
print(np.array(positiveSegmentation).shape)

mergedLabel = []
mergedCenters = []
mergedSegmentation = []
mergedRawData = []
for label in mergeErrorDataset.keys():
    mergedLabel.append(mergeErrorDataset[label][0])
    mergedCenters.append(mergeErrorDataset[label][1])
    mergedSegmentation.append(mergeErrorDataset[label][2])
    mergedRawData.append(mergeErrorDataset[label][3])
print(len(mergedLabel))
print(len(mergedCenters))
print(len(mergedSegmentation))
print(np.array(mergedSegmentation).shape)

splitLabel = []
splitPlane = []
splitCenters = []
splitSegmentation = []
splitRawData= []
for label in splitErrorDataset.keys():
    splitLabel.append(label)
    splitPlane.append(splitErrorDataset[label][0])
    splitCenters.append(splitErrorDataset[label][1])
    splitSegmentation.append(splitErrorDataset[label][2])
    splitRawData.append(splitErrorDataset[label][3])
print(len(splitPlane))
print(len(splitCenters))
print(len(splitSegmentation))
print(np.array(splitSegmentation).shape)

with h5py.File("generatedDataset224_30.h5", "w") as f:
    f.create_dataset('positiveLabel', data=positiveLabel)
    f.create_dataset('positiveSegmentation',data=positiveSegmentation)
    f.create_dataset('positive_location', data=positiveCenters)
    f.create_dataset('positiveRawData', data=positiveRawData)

    f.create_dataset('mergedLabel', data=mergedLabel)
    f.create_dataset('mergedCenters', data=mergedCenters)
    f.create_dataset('mergedSegmentation', data=mergedSegmentation)
    f.create_dataset('mergedRawData', data=mergedRawData)

    f.create_dataset('splitLabel', data=splitLabel)
    f.create_dataset('splitPlane', data=splitPlane)
    f.create_dataset('splitCenters', data=splitCenters)
    f.create_dataset('splitSegmentation', data=splitSegmentation)
    f.create_dataset('splitRawData', data=splitRawData)
# demanded result:  384*384*384 for each block,
#                   positive:merge error:split error = 2:1:1

