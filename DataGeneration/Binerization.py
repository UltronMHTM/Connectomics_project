import h5py
import numpy as np

def binerization(segVol, label):
    for z in range(len(segVol)):
        for y in range(len(segVol[0])):
            for x in range(len(segVol[0, 0])):
                if segVol[z,y,x] == label:
                    segVol[z, y, x] = 1
                else:
                    segVol[z, y, x] = 0
    return segVol
f = h5py.File('generatedDataset.h5','r+')
positiveLabel = np.array(f["positiveLabel"])
positiveSegmentation = np.array(f["positiveSegmentation"])

mergedLabel = np.array(f["mergedLabel"])
mergedSegmentation = np.array(f["mergedSegmentation"])

splitLabel = np.array(f["splitLabel"])
splitSegmentation = np.array(f["splitSegmentation"])

res_positiveSegmentation = []
for label, segmentation in zip(positiveLabel,positiveSegmentation):
    res_positiveSegmentation.append(binerization(segmentation,label))
res_positiveSegmentation = np.array(res_positiveSegmentation)

res_mergedSegmentation = []
for label, segmentation in zip(mergedLabel,mergedSegmentation):
    res_mergedSegmentation.append(binerization(segmentation,label[0]))
res_mergedSegmentation = np.array(res_mergedSegmentation)

res_splitSegmentation = []
for label, segmentation in zip(splitLabel,splitSegmentation):
    res_splitSegmentation.append(binerization(segmentation,label))
res_splitSegmentation = np.array(res_splitSegmentation)


f.create_dataset('positiveSegmentation_bin', data=res_positiveSegmentation)
f.create_dataset('mergedSegmentation_bin', data=res_mergedSegmentation)
f.create_dataset('splitSegmentation_bin', data=res_splitSegmentation)