import h5py
import numpy as np
# f1 = h5py.File('Dataset1_validation/grayscale_maps.h5','r')
# f2 = h5py.File('Dataset1_validation/groundtruth.h5','r')

# print(np.array(f1["raw"]).shape)
# print(np.array(f2["stack"]).shape)
f = h5py.File('generatedDataset.h5','r')

for key in f.keys():
    print("key=",key)
    print("shape=",np.array(f[key]).shape)

# value_cnt = {}
# for value in np.array(volunme_cut).reshape((1,-1))[0]:
#     value_cnt[value] = value_cnt.get(value, 0) + 1
# print(value_cnt)
# print(len(value_cnt))
# print([key for key in value_cnt.keys()])

# for group in f1.keys():
#     print(group)






# for i in range(len(transforms[:,0])):
#     if transforms[i,0] != transforms[i,1]:
#         print("not")
#         break

#
#
# value_cnt = {}
# for value in np.array(f2["stack"]).reshape((1,-1))[0]:
#     value_cnt[value] = value_cnt.get(value, 0) + 1
# print(value_cnt)
# print(len(value_cnt))
# print([key for key in value_cnt.keys()])

# value_cnt = {}
# for value in np.array(f2["stack"]).reshape((1,-1))[0]:
#     if len(transforms[np.argwhere(transforms[:, 0] == value)[0], 1]) != 0:
#         val = transforms[np.argwhere(transforms[:, 0] == value)[0], 1][0]
#         value_cnt[val] = value_cnt.get(val, 0) + 1
#
# print(value_cnt)
# print(len(value_cnt))
# print([key for key in value_cnt.keys()])
#
# print("-------------------")
#

# value_cnt = {}
# for value in np.array(f1["raw"]).reshape((1,-1))[0]:
# 	value_cnt[value] = value_cnt.get(value, 0) + 1
#
# print(value_cnt)
# print(len(value_cnt))
# print([key for key in value_cnt.keys()])
# print([value for value in value_cnt.values()])

# for group in f1.keys():
#     print(group)

# print("--")
#
# with h5py.File("mytestfile.h5", "w") as f:
#     dset = f.create_dataset('dset2',data=arr)