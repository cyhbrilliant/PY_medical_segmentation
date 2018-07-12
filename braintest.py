import numpy as np
import utility as util
import h5py
import gc
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'



patch_size=65
patch_size2=33
halfpz=int(patch_size/2)
halfpz2=int(patch_size2/2)
def readindex():
    f=open("E://zhangjinjing/brain2D/index.txt")
    data=[]
    a=f.readline()
    while a:
        data.append(a.rstrip('\n'))
        a=f.readline()
    return data



def getBatch2D(pos_x,pos_y):

    #print(index_shuffle.shape)
    Input_batch1=[]
    Input_batch2=[]
    Label_batch=[]
    pos_y=pos_y+halfpz
    for i in range(175):
        pos_z=halfpz+i
        print('x,y,z:',pos_x,pos_y,pos_z)
        Input_batch1.append(ALLtest[pos_x,pos_y-halfpz:pos_y+halfpz+1,pos_z-halfpz:pos_z+halfpz+1,:])
        Input_batch2.append(ALLtest[pos_x,pos_y-halfpz2:pos_y+halfpz2+1,pos_z-halfpz2:pos_z+halfpz2+1,:])
        #templabel=np.zeros(5)
        #templabel[int(CHLABEL[pos_x,pos_y,pos_z])]=1
        #Label_batch.append(templabel)

    #return np.array(Input_batch1),np.array(Input_batch2),np.array(Label_batch)
    return np.array(Input_batch1), np.array(Input_batch2)

#Input_batch1,Input_batch2,Label_batch=getBatch2D(0,0,0)
#print(Input_batch1.shape,Input_batch2.shape,Label_batch.shape)

Xp1 = tf.placeholder(tf.float32,shape=[None,patch_size,patch_size,4])
Xp2 = tf.placeholder(tf.float32,shape=[None,patch_size2,patch_size2,4])
#Yp = tf.placeholder(tf.float32,shape=[None,5])

#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


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

#loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(OUT),reduction_indices=[1]))
#correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

index=readindex()
index=np.array(index)
for zip_num in range(index.shape[0]):
    TestPath = 'Testing/'+ str(zip_num+1)
    CH1 = util.read_mha_image_as_nuarray(TestPath + '/0001/0001.mha')
    CH2 = util.read_mha_image_as_nuarray(TestPath + '/0002/0002.mha')
    CH3 = util.read_mha_image_as_nuarray(TestPath + '/0003/0003.mha')
    CH4 = util.read_mha_image_as_nuarray(TestPath + '/0004/0004.mha')
    # CHLABEL=util.read_mha_image_as_nuarray(TestPath+'/0005/0005.mha')
    ALLtest = np.concatenate([CH1[:, :, :, np.newaxis],
                              CH2[:, :, :, np.newaxis],
                              CH3[:, :, :, np.newaxis],
                              CH4[:, :, :, np.newaxis]], 3)

    del CH1, CH2, CH3, CH4
    gc.collect()
    print('ALLTEST:',ALLtest.shape)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, 'E:/zhangjinjing/brain2D/brain_session_HLGG/brain_session.ckpt')
    print('read sessionOK')
    iter=0
    OUTMHA=np.zeros([ALLtest.shape[0],ALLtest.shape[1],ALLtest.shape[2]])
    for pos_x in range(ALLtest.shape[0]):
        for pos_y in range(ALLtest.shape[1]-patch_size):
            iter+=1
            print('\n',iter)
            print('finish in: ',iter*700./189875.,'%')
            #Input_batch1,Input_batch2,Label_batch=getBatch 2D(pos_x,pos_y)
            Input_batch1, Input_batch2 = getBatch2D(pos_x, pos_y)
            #OUTLINE=sess.run([OUT],feed_dict={Xp1:Input_batch1,Xp2:Input_batch2,Yp:Label_batch})

            OUTLINE = sess.run([OUT], feed_dict={Xp1: Input_batch1, Xp2: Input_batch2})
            OUTLINE=np.array(OUTLINE)
            # print(tf.Session().run(tf.arg_max(OUTLINE,1),feed_dict={Xp1:Input_batch1,Xp2:Input_batch2,Yp:Label_batch}))
            # print(tf.Session().run(tf.arg_max(Label_batch,1),feed_dict={Xp1:Input_batch1,Xp2:Input_batch2,Yp:Label_batch}))
            OUTMHA[pos_x,pos_y+halfpz,halfpz:175+halfpz]=np.argmax(OUTLINE[0,:,:],1)

    print(OUTLINE.shape)
    util.save_nuarray_as_mha("VSD.Seg_HG_001."+index[zip_num]+".mha", OUTMHA)