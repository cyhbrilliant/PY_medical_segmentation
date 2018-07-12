import scipy.io
import numpy as np
import utility as util
mat=scipy.io.loadmat('brain_1125.mat')
print(mat)
files=mat['img']
# file=mat['V']
y=np.zeros([files.shape[2],files.shape[1],files.shape[0]])
for i in range(files.shape[2]):
    x=files[:,:,i]
    for j in range(files.shape[1]):
        y[i,j]=x[:,j]
print(y.shape)
util.save_nuarray_as_mha('ktest.mha',y)