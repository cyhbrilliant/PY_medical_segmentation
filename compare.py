import numpy as np
import os
import utility as util
seg="E://zhangjinjing/brain2D/test210.mha"
label="E://zhangjinjing/brain2D/BrainTrain/210/0005/0005.mha"
def compare(path1,path2):
    data1=util.read_mha_image_as_nuarray(path1)
    data2=util.read_mha_image_as_nuarray(path2)
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            for k in range (data1.shape[2]):
                if data1[i,j,k]==data2[i,j,k]:
                    data1[i,j,k]=0
    return data1

if __name__=="__main__":
    data=compare(seg,label)
    util.save_nuarray_as_mha("E://zhangjinjing/brain2D/error.mha",data)