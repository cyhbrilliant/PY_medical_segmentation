import numpy as np
import h5py
import utility as util
#import tensorflow as tf

patch_size=65

def getpix():
    index1=[]
    index2=[]
    index3=[]
    index4=[]

    for zip_num in range(274):
        print(zip_num)
        CHLABEL=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0005/0005.mha')
        for pos_x in range(155):
            pos_x=int(pos_x)
            for pos_y in range(240-patch_size):
                pos_y=int(pos_y+patch_size/2)
                for pos_z in range(240-patch_size):
                    pos_z=int(pos_z+patch_size/2)
                    voxel_label=int(CHLABEL[pos_x,pos_y,pos_z])
                    if voxel_label==1:
                        index_temp=[zip_num,pos_x,pos_y,pos_z]
                        index1.append(index_temp)
                    if voxel_label==2:
                        index_temp=[zip_num,pos_x,pos_y,pos_z]
                        index2.append(index_temp)
                    if voxel_label==3:
                        index_temp=[zip_num,pos_x,pos_y,pos_z]
                        index3.append(index_temp)
                    if voxel_label==4:
                        index_temp=[zip_num,pos_x,pos_y,pos_z]
                        index4.append(index_temp)
    return np.array(index1),np.array(index2),np.array(index3),np.array(index4)                          

index1,index2,index3,index4=getpix()
print(index1.shape,index2.shape,index3.shape,index4.shape)

file = h5py.File('index_Server_all.h5','w')
file.create_dataset('index1', data = index1)  
print('index1 OK')
file.create_dataset('index2', data = index2)  
print('index2 OK')
file.create_dataset('index3', data = index3)
print('index3 OK')  
file.create_dataset('index4', data = index4)
print('index4 OK')  