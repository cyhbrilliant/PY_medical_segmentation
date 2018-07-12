import numpy as np
import h5py
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
patchsize=65
def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1),name='weight')
def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(value=0.1,shape=shape),name='bias')
def atrous_conv2d(x,filter):
    return tf.nn.atrous_conv2d(x,filters=filter,rate=2,padding='VALID')
def conv_2d(x,w):
    return tf.nn.conv2d(input=x,filter=w,strides=[1,1,1,1],padding="VALID")
def maxpool(x,ksize):
    return  tf.nn.max_pool(value=x,ksize=ksize,strides=[1,1,1,1],padding='VALID')
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
    all_data=np.array(all_data)
    all_label=np.array(all_label)
    return all_data,all_label
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
    return index
def Getbatch():
    patchsize=65
    data_batch=[]
    label_batch=[]
    All_index=GetClassIndex()
    n=np.arange(All_index.shape[0])
    np.random.shuffle(n)
    Patient_Data, Patient_Label=readdata()
    for i in range(All_index.shape[0]):
        Patient_Index=All_index[n[i],0]
        x=All_index[n[i],1]
        y=All_index[n[i],2]
        z=All_index[n[i],3]
        data_batch.append(Patient_Data[Patient_Index][x,y-int(patchsize/2):y+int(patchsize/2)+1,z-int(patchsize/2):z+int(patchsize/2)+1])
        label = np.zeros([5])
        label[Patient_Label[Patient_Index][x, y, z]] = 1
        label_batch.append(label)
    return np.array(data_batch), np.array(label_batch)
def Network(x):
    weights = {'w_conv1':weight_variable(shape=[17,17,4,10]),'w_conv2': weight_variable(shape=[7, 7, 10, 64]), 'w_conv3': weight_variable(shape=[3, 3, 64, 64]),
               'w_conv4': weight_variable(shape=[13, 13, 10, 160]), 'w_conv5': weight_variable(shape=[21, 21, 224, 5])}
    bias = {'b_conv1':bias_variable(shape=[10]),'b_conv2': bias_variable(shape=[64]), 'b_conv3': bias_variable(shape=[64]),
            'b_conv4': bias_variable(shape=[160]), 'b_conv5': bias_variable(shape=[5])}
    preconv=atrous_conv2d(x,weights['w_conv1'])+bias['b_conv1']
    conv1_1=tf.nn.relu(conv_2d(preconv,weights['w_conv2']))
    maxpool1_1=tf.nn.relu(maxpool(conv1_1,[1,4,4,1]))
    conv2_1=tf.nn.relu(conv_2d(preconv,weights['w_conv4'])+bias['b_conv4'])
    conv1_2=tf.nn.relu(conv_2d(maxpool1_1,weights['w_conv3'])+bias['b_conv3'])
    maxpool1_2=maxpool(conv1_2,[1,2,2,1])
    Combine=conv_2d(tf.concat([maxpool1_2,conv2_1],3),weights['w_conv5'])+bias['b_conv5']
    OUT = tf.nn.softmax(tf.reshape(Combine, [175, 5]))
    return OUT
def Segwork(iter = 500000):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 65, 65, 4], name='data')
    Y = tf.placeholder(dtype=tf.float32, shape=[175, 5], name='label')
    OUT=Network(X)
    loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(OUT, 1e-10, 1.0)), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(0.00001).minimize(loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess=sess, save_path='E:/zhangjinjing/brain2D/brats2013/brain_session_2013_kongdong.ckpt')
    for i in range(iter):
        batch, label = Getbatch()
        print('batch,label:',batch.shape,label.shape)
        error = sess.run([loss, optimizer], feed_dict={X: batch, Y: label})
        print('error:',error)
        if (i + 1) % 1000 == 0:
            path = saver.save(sess=sess, save_path='E:/zhangjinjing/brain2D/brats2013/brain_session_2013_kongdong.ckpt')
            # train_accuracy=accuracy.eval(feed_dict={X1:batch1,X2:batch2,Y:label})
            # print('train_accuracy:'% train_accuracy)
            print(path)


if __name__ == '__main__':
    Segwork()
