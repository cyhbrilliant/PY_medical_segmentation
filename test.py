import numpy as np
import utility as util
import os
def readdata():
    path='E:/zhangjinjing/brain2D/Image_Data'
    patient=[]
    for i in os.listdir(path):
        print('i',i)
        patient.append(i)
    patient=np.array(patient)
    for i in range(patient.shape[0]):
        data=util.read_mha_image_as_nuarray(path+'/'+str(patient[i])+'/001/001.mha')
        print(data.shape)
#shapenum=np.loadtxt("E:/zhangjinjing/brain2D/index.txt" ,dtype=int)
#num=np.array(num[np.newaxis,:])
#key=[]
#dict={}
#d=("name","sex","age")
#for i in range(110):
 #   key.append(i+1)
#for j in range(110):
 #   dict[j+1]=num[0,j]
if __name__=='__main__':
    readdata()
    print('ok')