import numpy as np
import h5py
import utility as util
import os

def image_getindex():
    index1=[]
    index2=[]
    index3=[]
    index4=[]
    patient=[]
    patch=65
    path="E:/zhangjinjing/brain2D/Image_Data"
    for i in os.listdir(path):
        patient.append(i)
    patient=np.array(patient)
    for j in range (patient.shape[0]):
        label=util.read_mha_image_as_nuarray(path+"/"+patient[j]+'/005/005.mha')
        for x in range(label.shape[0]):
            for y in range(label.shape[1]-patch):
                y=y+int(patch/2)
                for z in range(label.shape[2]-patch):
                    z=z+int(patch/2)
                    pixel=label[x,y,z]
                    if pixel==1:
                       index1.append([j,x,y,z])
                       print('j,x,y,z:',j,x,y,z)
                    if pixel==2:
                        index2.append([j,x,y,z])
                        print('j,x,y,z:', j, x, y, z)
                    if pixel==3:
                        index3.append([j,x,y,z])
                        print('j,x,y,z:', j, x, y, z)
                    if pixel==4:
                        index4.append([j,x,y,z])
                        print('j,x,y,z:', j, x, y, z)
    return np.array(index1),np.array(index2),np.array(index3),np.array(index4)
index1,index2,index3,index4=image_getindex()
print(index1.shape,index2.shape,index3.shape,index4.shape)
file=h5py.File('image_index.h5','w')
file.create_dataset('train1',data=index1)
file.create_dataset('train2',data=index2)
file.create_dataset('train3',data=index3)
file.create_dataset('train4',data=index4)
print('index ok')