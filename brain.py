import numpy as np
import utility as util
import h5py
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
patch_size=65
patch_size2=33
MHANUM=274

def Readindex():
    file = h5py.File('index_Server_all.h5','r')
    index1 = file['index1'][:]
    index2 = file['index2'][:]
    index3 = file['index3'][:]
    index4 = file['index4'][:]
    return index1,index2,index3,index4

index1,index2,index3,index4=Readindex()
print('read index OK')
#print(index1.shape,index2.shape,index3.shape,index4.shape)

def Readdata():
    file = h5py.File('Data_Server_all.h5','r')
    data=file['data'][:]
    label=file['label'][:]
    return data,label

All_data,All_label=Readdata()
print('read data OK')
#print(All_data.shape,All_label.shape)

def getShuffle():
    MAX0=150
    MAX1=3
    MAX2=14
    MAX3=2
    MAX4=6
    num0=0
    num1=0
    num2=0
    num3=0
    num4=0
    index0_input=[]
    index1_input=[]
    index2_input=[]
    index3_input=[]
    index4_input=[]
    while True:
        zip_num=np.random.randint(MHANUM)
        x_num=np.random.randint(155)
        y_num=np.random.randint(240-patch_size)
        z_num=np.random.randint(240-patch_size)
        pos_x=int(x_num)
        pos_y=int(patch_size/2+y_num)
        pos_z=int(patch_size/2+z_num)
        #CHLABEL=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0005/0005.mha')
        CHLABEL=All_label[zip_num,pos_x,pos_y,pos_z]
        if num0==MAX0:
            break
        if int(CHLABEL)==0:
            num0+=1
            index0_input.append([zip_num,pos_x,pos_y,pos_z])
    

    
    for num1 in range(MAX1):
        choose1=np.random.randint(index1.shape[0])
        index1_input.append(index1[choose1,:])
    for num2 in range(MAX2):
        choose2=np.random.randint(index2.shape[0])
        index2_input.append(index2[choose2,:])
    for num3 in range(MAX3):
        choose3=np.random.randint(index3.shape[0])
        index3_input.append(index3[choose3,:])
    for num4 in range(MAX4):
        choose4=np.random.randint(index4.shape[0])
        index4_input.append(index4[choose4,:])
    index0_input=np.array(index0_input)
    #print(index0_input.shape)
    index1_input=np.array(index1_input)
    #print(index1_input.shape)
    index2_input=np.array(index2_input)
    #print(index2_input.shape)
    index3_input=np.array(index3_input)
    #print(index3_input.shape)
    index4_input=np.array(index4_input)
    #print(index4_input.shape)


    indexALL=np.concatenate([index0_input,index1_input,index2_input,index3_input,index4_input],0)
    #print(indexALL.shape[0])
    indexlist=np.arange(indexALL.shape[0])
    np.random.shuffle(indexlist)
    #print(indexlist)
    index_shuffle=[]
    for i in range(indexALL.shape[0]):
        index_shuffle.append(indexALL[indexlist[i],:])

    return np.array(index_shuffle)

def getBatch2D():

    index_shuffle=getShuffle()
    #print(index_shuffle.shape)
    Input_batch1=[]
    Input_batch2=[]
    Label_batch=[]
    for i in range(index_shuffle.shape[0]):
        zip_num=index_shuffle[i,0]    
        pos_x=index_shuffle[i,1]
        pos_y=index_shuffle[i,2]
        pos_z=index_shuffle[i,3]
        #CH1=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0001/0001.mha')
        #CH2=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0002/0002.mha')
        #CH3=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0003/0003.mha')
        #CH4=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0004/0004.mha')
        #CHLABEL=util.read_mha_image_as_nuarray('BrainTrain/'+str(zip_num)+'/0005/0005.mha')

        #halfpz=int(patch_size/2)
        #halfpz2=int(patch_size2/2)
        #Input_batch1.append(np.concatenate([CH1[pos_x-halfpz:pos_x+halfpz,pos_y-halfpz:pos_y+halfpz,pos_z-halfpz:pos_z+halfpz][:,:,:,np.newaxis],
        #                                    CH2[pos_x-halfpz:pos_x+halfpz,pos_y-halfpz:pos_y+halfpz,pos_z-halfpz:pos_z+halfpz][:,:,:,np.newaxis],
        #                                    CH3[pos_x-halfpz:pos_x+halfpz,pos_y-halfpz:pos_y+halfpz,pos_z-halfpz:pos_z+halfpz][:,:,:,np.newaxis],
        #                                    CH4[pos_x-halfpz:pos_x+halfpz,pos_y-halfpz:pos_y+halfpz,pos_z-halfpz:pos_z+halfpz][:,:,:,np.newaxis]],3))
        #Input_batch2.append(np.concatenate([CH1[pos_x-halfpz2:pos_x+halfpz2,pos_y-halfpz2:pos_y+halfpz2,pos_z-halfpz2:pos_z+halfpz2][:,:,:,np.newaxis],
        #                                    CH2[pos_x-halfpz2:pos_x+halfpz2,pos_y-halfpz2:pos_y+halfpz2,pos_z-halfpz2:pos_z+halfpz2][:,:,:,np.newaxis],
        #                                    CH3[pos_x-halfpz2:pos_x+halfpz2,pos_y-halfpz2:pos_y+halfpz2,pos_z-halfpz2:pos_z+halfpz2][:,:,:,np.newaxis],
        #                                    CH4[pos_x-halfpz2:pos_x+halfpz2,pos_y-halfpz2:pos_y+halfpz2,pos_z-halfpz2:pos_z+halfpz2][:,:,:,np.newaxis]],3))


        #templabel=np.zeros(5)
        #templabel[int(CHLABEL[pos_x,pos_y,pos_z])]=1
        #Label_batch.append(templabel)


        halfpz=int(patch_size/2)
        halfpz2=int(patch_size2/2)
        Input_batch1.append(All_data[zip_num,pos_x,pos_y-halfpz:pos_y+halfpz+1,pos_z-halfpz:pos_z+halfpz+1,:])
        Input_batch2.append(All_data[zip_num,pos_x,pos_y-halfpz2:pos_y+halfpz2+1,pos_z-halfpz2:pos_z+halfpz2+1,:])
        templabel=np.zeros(5)
        templabel[int(All_label[zip_num,pos_x,pos_y,pos_z])]=1
        Label_batch.append(templabel)


    return np.array(Input_batch1),np.array(Input_batch2),np.array(Label_batch)




#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



Xp1 = tf.placeholder(tf.float32,shape=[None,patch_size,patch_size,4])
Xp2 = tf.placeholder(tf.float32,shape=[None,patch_size2,patch_size2,4])
Yp = tf.placeholder(tf.float32,shape=[None,5])


Wconv_pre=weight_variable([33,33,4,5])
Bconv_pre=bias_variable([5])
conv_pre=tf.nn.conv2d(Xp1,Wconv_pre,strides=[1,1,1,1],padding='VALID')+Bconv_pre

Input=tf.concat([conv_pre,Xp2],3)

#Path1
WP1conv1=weight_variable([7,7,9,64])
BP1conv1=bias_variable([64])
P1conv1=tf.nn.relu(tf.nn.conv2d(Input,WP1conv1,strides=[1,1,1,1],padding='VALID')+BP1conv1)
P1pool1=tf.nn.max_pool(P1conv1,ksize=[1,4,4,1],strides=[1,1,1,1],padding='VALID')

WP1conv2=weight_variable([3,3,64,64])
BP1conv2=bias_variable([64])
P1conv2=tf.nn.relu(tf.nn.conv2d(P1pool1,WP1conv2,strides=[1,1,1,1],padding='VALID')+BP1conv2)
P1pool2=tf.nn.max_pool(P1conv2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')

#Path2
WP2conv1=weight_variable([13,13,9,160])
BP2conv1=bias_variable([160])
P2conv1=tf.nn.relu(tf.nn.conv2d(Input,WP2conv1,strides=[1,1,1,1],padding='VALID')+BP2conv1)

Combine=tf.concat([P1pool2,P2conv1],3)

WCombine_conv=weight_variable([21,21,224,5])
BCombine_conv=bias_variable([5])
Combine_conv=tf.nn.conv2d(Combine,WCombine_conv,strides=[1,1,1,1],padding='VALID')+BCombine_conv
ReCombine_conv=tf.reshape(Combine_conv,[175,5])

OUT=tf.nn.softmax(ReCombine_conv)

loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
TrainStep=tf.train.AdamOptimizer(0.00001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'E:/zhangjinjing/brain2D/brain_session_HLGG/brain_session.ckpt')
for iter in range(500000):
    print('\n',iter)
    Input_batch1,Input_batch2,Label_batch=getBatch2D()
    error,result=sess.run([loss,TrainStep],feed_dict={Xp1:Input_batch1,Xp2:Input_batch2,Yp:Label_batch})
    print(error)

    if (iter+1)%1000==0:
        path = saver.save(sess=sess,save_path='E:/zhangjinjing/brain2D/brain_session_HLGG/brain_session.ckpt')
        print(path)







