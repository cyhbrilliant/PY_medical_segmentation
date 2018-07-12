import numpy as np
import utility as util
import h5py

def readDataLocal():
    All_data=[]
    All_label=[]

    for zip_num in range(274):
        print(zip_num)
        CH1=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0001/0001.mha')
        CH2=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0002/0002.mha')
        CH3=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0003/0003.mha')
        CH4=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0004/0004.mha')
        CHLABEL=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0005/0005.mha')

        All_data.append(np.concatenate([CH1[:,:,:,np.newaxis],CH2[:,:,:,np.newaxis],CH3[:,:,:,np.newaxis],CH4[:,:,:,np.newaxis]],3))
        All_label.append(CHLABEL)
    

    return All_data,All_label

All_data,All_label=readDataLocal()

file = h5py.File('Data_Server_all.h5','w')
file.create_dataset('data', data = All_data)
file.create_dataset('label', data = All_label)