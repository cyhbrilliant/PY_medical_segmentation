import numpy as np
import os
import h5py
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
largepatchsize=65
smallpatchsize=33
def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1),name='weight')
def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(value=0.1,shape=shape),name='bias')
def conv_2d(x,w):
    return tf.nn.conv2d(input=x,filter=w,strides=[1,1,1,1],padding="VALID")
def maxpool_2(x):
    return  tf.nn.max_pool(value=x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
def readindex():
    index=h5py.File('image_index.h5','r')
    index1=index['train1'][:]
    index2=index['train2'][:]
    index3=index['train3'][:]
    index4=index['train4'][:]
    return index1,index2,index3,index4
def readdata():
    data=h5py.File('File_Patientdata.h5','r')
    all_data=[]
    all_label=[]
    for i in range(int(np.array(list(data.keys())).shape[0]/2)):
        all_data.append(data['data'+str(i)][:])
        all_label.append(data['label'+str(i)][:])
    return np.array(all_data),np.array(all_label)
def GetClassIndex(Patient_Label, Patient_Index1,Patient_Index2,Patient_Index3,Patient_Index4):
    class0 = 150
    class1 = 3
    class2 = 14
    class3 = 2
    class4 = 6
    index=[]
    num0=0
    while num0<class0:
        Patient_Num0 = np.random.randint(Patient_Label.shape[0])
        x = np.random.randint(Patient_Label[Patient_Num0].shape[0])
        y = np.random.randint(Patient_Label[Patient_Num0].shape[1]-largepatchsize)
        z = np.random.randint(Patient_Label[Patient_Num0].shape[2]- largepatchsize)
        if Patient_Label[Patient_Num0][x,y + int(largepatchsize/2)+1,z + int(largepatchsize/2)+1] == 0:
            index.append([Patient_Num0,x,y+int(largepatchsize/2)+1,z+int(largepatchsize/2)+1])
            num0=num0+1

    for num1 in range(class1):
        Patient_Num1=np.random.randint(Patient_Index1.shape[0])
        index.append(Patient_Index1[Patient_Num1])
    for num2 in range(class2):
        Patient_Num2=np.random.randint(Patient_Index2.shape[0])
        index.append(Patient_Index2[Patient_Num2])
    for num3 in range(class3):
        Patient_Num3=np.random.randint(Patient_Index3.shape[0])
        index.append(Patient_Index3[Patient_Num3])
    for num4 in range(class4):
        Patient_Num4=np.random.randint(Patient_Index4.shape[0])
        index.append(Patient_Index4[Patient_Num4])
    index=np.array(index)
    print('indexshape:',index.shape)
    return index
def Getbatch(Patient_Data, Patient_Label,Patient_Index1, Patient_Index2, Patient_Index3, Patient_Index4):
    All_index = GetClassIndex(Patient_Label, Patient_Index1, Patient_Index2, Patient_Index3, Patient_Index4)
    small_batch=[]
    large_batch=[]
    label_batch=[]
    n=np.arange(All_index.shape[0])
    halfsmallp=int(smallpatchsize/2)
    halflargep=int(largepatchsize/2)
    np.random.shuffle(n)
    for i in range(All_index.shape[0]):
        Patient_Index=All_index[n[i],0]
        x=All_index[n[i],1]
        y=All_index[n[i],2]
        z=All_index[n[i],3]
        small_batch.append(Patient_Data[Patient_Index][x,y-halfsmallp:y+halfsmallp+1,z-halfsmallp:z+halfsmallp+1])
        large_batch.append(Patient_Data[Patient_Index][x,y-halflargep:y+halflargep+1,z-halflargep:z+halflargep+1])
        label = np.zeros(5)
        label[Patient_Label[Patient_Index][x,y,z]] = 1
        label_batch.append(label)
    return np.array(large_batch), np.array(small_batch),np.array(label_batch)

def Network(X1,X2):
    downsampling=tf.image.resize_nearest_neighbor(X1,size=[33,33])
    INPUT=tf.concat([downsampling,X2],3)
    conv1_1=tf.nn.relu(tf.nn.conv2d(INPUT,weight_variable(shape=[7,7,8,64]),strides=[1,1,1,1],padding='VALID',name='localconv1')+bias_variable(shape=[64]))
    maxpooling_1=tf.nn.max_pool(conv1_1,ksize=[1,4,4,1],strides=[1,1,1,1],padding='VALID')
    conv1_2=tf.nn.relu(tf.nn.conv2d(maxpooling_1,weight_variable(shape=[3,3,64,64]),strides=[1,1,1,1],padding='VALID',name='localconv2')+bias_variable(shape=[64]))
    maxpooling_2=tf.nn.max_pool(conv1_2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')
    conv2_1=tf.nn.relu(tf.nn.conv2d(INPUT,weight_variable(shape=[13,13,8,160]),strides=[1,1,1,1],padding='VALID',name='globalconv')+bias_variable(shape=[160]))
    Combine=tf.nn.conv2d(tf.concat([conv2_1,maxpooling_2],3),weight_variable(shape=[21,21,224,5]),strides=[1,1,1,1],padding='VALID')+bias_variable(shape=[5])
    OUT=tf.nn.softmax(tf.reshape(Combine,[175,5]))
    return OUT
def Segment(iter=500000):
    X1 = tf.placeholder(dtype=tf.float32, shape=[None, 65, 65, 4], name='big_patch')
    X2 = tf.placeholder(dtype=tf.float32, shape=[None, 33, 33, 4], name='small_patch')
    Y = tf.placeholder(dtype=tf.float32, shape=[175, 5], name='labels')
    OUT=Network(X1,X2)
    Index1, Index2, Index3, Index4 = readindex()
    Data, Label = readdata()
    loss=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
    optimizer=tf.train.AdamOptimizer(0.000001).minimize(loss)
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess=sess, save_path='E:/zhangjinjing/brain2D/brats2013/brain_session_2013_8.ckpt')
    for i in range (iter):
        batch1, batch2, label = Getbatch(Data, Label,Index1, Index2, Index3, Index4)
        error, result = sess.run([loss, optimizer], feed_dict={X1: batch1, X2: batch2, Y: label})
        print('error:',error)
        if (i + 1) % 1000 == 0:
            path = saver.save(sess=sess, save_path='E:/zhangjinjing/brain2D/brats2013/brain_session_2013_8.ckpt')
            print(path)

if __name__=='__main__':
  Segment()

