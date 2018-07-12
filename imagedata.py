import numpy as np
import h5py
import utility as util
import  os
patientdata=[]
patientlabel=[]
def readdata():
    file = h5py.File('File_Patientdata.h5', 'w')
    patient = []
    path="E:/zhangjinjing/brain2D/Image_Data"
    for i in os.listdir(path):
        patient.append(i)
    patient=np.array(patient)
    for j in range(patient.shape[0]):
        CH1 = util.read_mha_image_as_nuarray(path +  "/"+patient[j]  + "/001/001.mha")
        CH2 = util.read_mha_image_as_nuarray(path + "/" + patient[j] + "/002/002.mha")
        CH3 = util.read_mha_image_as_nuarray(path + "/" + patient[j] + "/003/003.mha")
        CH4 = util.read_mha_image_as_nuarray(path + "/" + patient[j] + "/004/004.mha")
        LABEL = util.read_mha_image_as_nuarray(path + "/" + patient[j] + "/005/005.mha")
        file.create_dataset('data'+str(j),data=np.concatenate([CH1[:,:,:,np.newaxis],CH2[:,:,:,np.newaxis],CH3[:,:,:,np.newaxis],CH4[:,:,:,np.newaxis]],3))
        file.create_dataset('label'+str(j),data=LABEL)
    print('set data ok!')
if __name__=='__main__':
    readdata()