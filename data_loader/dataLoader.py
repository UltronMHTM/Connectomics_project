import h5py
import numpy as np

class dataLoader():
    def __init__(self, H5path):
        self.h5file = h5py.File(H5path,'r')
        self.pos_sample_label = np.array(self.h5file["positiveLabel"])
        self.pos_sample_seg = np.array(self.h5file["positiveSegmentation"])
        self.pos_sample_seg_b = np.array(self.h5file["positiveSegmentation_bin"])
        self.pos_sample_location = np.array(self.h5file["positive_location"])
        self.pos_sample_raw = np.array(self.h5file["positiveRawData"])

        self.neg_m_sample_labels = np.array(self.h5file["mergedLabel"])
        self.neg_m_sample_seg = np.array(self.h5file["mergedSegmentation"])
        self.neg_m_sample_seg_b = np.array(self.h5file["mergedSegmentation_bin"])
        self.neg_m_sample_location = np.array(self.h5file["mergedCenters"])
        self.neg_m_sample_raw = np.array(self.h5file["mergedRawData"])

        self.neg_s_sample_label = np.array(self.h5file["mergedLabel"])
        self.neg_s_sample_cutPlane = np.array(self.h5file["splitPlane"])
        self.neg_s_sample_seg = np.array(self.h5file["splitSegmentation"])
        self.neg_s_sample_seg_b = np.array(self.h5file["splitSegmentation_bin"])
        self.neg_s_sample_location = np.array(self.h5file["splitCenters"])
        self.neg_s_sample_raw = np.array(self.h5file["splitRawData"])