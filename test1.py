import h5py
import numpy as np
patchsize=65
import tensorflow as tf
import utility as util
# data=np.array([[222,333,444],[777,888,999]])
# label=np.array([0,1,0])
# img_num=np.array([0,1,2])
# file=h5py.File('TrainSet_rotate.h5','w')
# file.create_dataset('train_set_1',data=data)
# file.create_dataset('train_set_y',data=label)
# file.create_dataset('train_set_num',data=img_num)
# file1=h5py.File('File_Patientdata','r')
# x=file1['data1'][:]
# print(x.shape)
# print(np.array(list(file1.keys())))
# for i in range(30):
#     print('label'+str(i))
# a=np.array([[1,2,3],[4,5,6]])
# x=tf.reduce_sum(a)
# print(x)
def readdata():
    data=h5py.File('File_Patientdata.h5','r')
    all_data=[]
    all_label=[]
    for i in range(int(np.array(list(data.keys())).shape[0]/2)):
        #print(data['data'+str(i)][:])
        all_data.append(data['data'+str(i)][:])
        all_label.append(data['label'+str(i)][:])
    return np.array(all_data),np.array(all_label)
def readindex():
    index=h5py.File('image_index.h5','r')
    index1=index['train1'][:]
    index2=index['train2'][:]
    index3=index['train3'][:]
    index4=index['train4'][:]
    return index1,index2,index3,index4
def GetClassIndex():
    Patient_Index1,Patient_Index2,Patient_Index3,Patient_Index4 = readindex()
    Patient_Data, Patient_Label = readdata()
    class0 = 150
    class1 = 3
    class2 = 14
    class3 = 2
    class4 = 6
    index=[]
    num=0
    while num<class0:
        Patient_Num = np.random.randint(Patient_Label.shape[0])
        x = np.random.randint(np.array(Patient_Label[Patient_Num]).shape[0])
        y = np.random.randint(np.array(Patient_Label[Patient_Num]).shape[1] - patchsize)
        z = np.random.randint(np.array(Patient_Label[Patient_Num]).shape[2] - patchsize)
        if np.array(Patient_Label[Patient_Num])[ x, y + int(patchsize/2)+1, z + int(patchsize/2)+1] == 0:
            index.append([Patient_Num,x,y+int(patchsize/2)+1,z+int(patchsize/2)+1])
            num=num+1

        print('index0:',Patient_Label[n][x,y+int(patchsize/2)+1,z+int(patchsize/2)+1])
    for i in range(class1):
        Patient_Num=np.random.randint(Patient_Index1.shape[0])
        index.append(Patient_Index1[Patient_Num])
    for i in range(class2):
        Patient_Num=np.random.randint(Patient_Index2.shape[0])
        index.append(Patient_Index2[Patient_Num])
    for i in range(class3):
        Patient_Num=np.random.randint(Patient_Index3.shape[0])
        index.append(Patient_Index3[Patient_Num])
    for i in range(class4):
        Patient_Num=np.random.randint(Patient_Index4.shape[0])
        index.append(Patient_Index4[Patient_Num])
    index=np.array(index)
    return np.array(index)
def Getbatch():
    patchsize=65
    data_batch=[]
    label_batch=[]
    All_index=GetClassIndex()
    n=np.arange(All_index.shape[0])
    np.random.shuffle(n)
    Patient_Data, Patient_Label=readdata()
    for i in range(All_index.shape[0]):
        Patient_Index=All_index[n[i]]
        data_batch.append(np.array(Patient_Data[Patient_Index[0]])[Patient_Index[1],Patient_Index[2]-int(patchsize/2):Patient_Index[2]+int(patchsize/2)+1,Patient_Index[3]-int(patchsize/2):Patient_Index[3]+int(patchsize/2)+1][:])
        label = np.zeros([5])
        label[np.array(Patient_Label[Patient_Index[0]])[Patient_Index[1], Patient_Index[2], Patient_Index[3]]] = 1
        label_batch.append(label)
    return np.array(data_batch), np.array(label_batch)
if __name__=='__main__':
    data,label =readdata()
    # Index1,Index2,Index3,Index4=readindex()
    # for i in range(Index4.shape[0]):
    #     num=Index4[i,0]
    #     x=Index4[i,1]
    #     y=Index4[i,2]
    #     z=Index4[i,3]
    #     a=label[num][x,y,z]
    #     print('index1:',a)
    #print(np.array(list(data.keys())))
    #data,label=Getbatch()
    #a=np.array(data[0])
    #print(a.shape)
    #filter=tf.Variable(initial_value=tf.truncated_normal(shape=[17,17,4,5],stddev=0.1))
    #outimage=tf.nn.atrous_conv2d(a,filters=filter,rate=2,padding='VALID')
    #print(outimage.shape)

