import numpy as np
import utility as util
import h5py
import gc
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

patch_size=65
patch_size2=33
half_patch=int(patch_size/2)
half_patch2=int(patch_size2/2)
testing_num ='105'
test_path='Testing/'+testing_num
CH1=util.read_mha_image_as_nuarray(test_path+'/0001/0001.mha')
CH2=util.read_mha_image_as_nuarray(test_path+'/0002/0002.mha')
CH3=util.read_mha_image_as_nuarray(test_path+'/0003/0003.mha')
CH4=util.read_mha_image_as_nuarray(test_path+'/0004/0004.mha')
ALLTest=np.concatenate([CH1[:,:,:,np.newaxis],CH2[:,:,:,np.newaxis],CH3[:,:,:,np.newaxis],CH4[:,:,:,np.newaxis]],3)
del CH1,CH2,CH3,CH4
gc.collect()

def get_2Dbatch(pos_x,pos_y):
    batch_1=[]
    batch_2=[]
    pos_y = pos_y + half_patch
    for i in range(175):
        pos_z=i+half_patch
        batch_1.append(ALLTest[pos_x,pos_y-half_patch:pos_y+half_patch+1,pos_z-half_patch:pos_z+half_patch+1,:])
        batch_2.append(ALLTest[pos_x,pos_y-half_patch2:pos_y+half_patch2+1,pos_z-half_patch2:pos_z+half_patch2+1,:])
    return np.array(batch_1),np.array(batch_2)


XP1=tf.placeholder(tf.float32,shape=[None,patch_size,patch_size,4])
XP2=tf.placeholder(tf.float32,shape=[None,patch_size2,patch_size2,4])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

resizepatch=tf.image.resize_nearest_neighbor(XP1,[patch_size2,patch_size2])
convprepatch=tf.concat([XP2,resizepatch],3)
conv1=tf.nn.conv2d(convprepatch,weight_variable([7,7,8,320]),strides=[1,1,1,1],padding='VALID')+bias_variable([320])
maxout1=tf.reduce_max(tf.reshape(conv1,[175,27,27,64,5]),4)
pool1=tf.nn.max_pool(maxout1,ksize=[1,4,4,1],strides=[1,1,1,1],padding='VALID')
conv2=tf.nn.conv2d(pool1,weight_variable([3,3,64,320]),strides=[1,1,1,1],padding='VALID')+bias_variable([320])
maxout2=tf.reduce_max(tf.reshape(conv2,[175,22,22,64,5]),4)
pool2=tf.nn.max_pool(maxout2,ksize=[1,2,2,1],strides=[1,1,1,1],padding='VALID')

conv1_1=tf.nn.conv2d(convprepatch,weight_variable([13,13,8,800]),strides=[1,1,1,1],padding='VALID')+bias_variable([800])
maxout1_1=tf.reduce_max(tf.reshape(conv1_1,[175,21,21,160,5]),4)
output=tf.concat([pool2,maxout1_1],3)
convout=tf.nn.conv2d(output,weight_variable([21,21,224,5]),strides=[1,1,1,1],padding='VALID')+bias_variable([5])
convout=tf.reshape(convout,[175,5])
OUT=tf.nn.softmax(convout)

OUTMHA=np.zeros([ALLTest.shape[0],ALLTest.shape[1],ALLTest.shape[2]])
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
saver = tf.train.Saver()
saver.restore(sess,'E:/zhangjinjing/brain2D/brain_session_HLGG/brain_session_resize2.ckpt')
print('read sessionOK')
iter=1;
for pos_x in range(ALLTest.shape[0]):
    for pos_y in range(ALLTest.shape[1]-patch_size):
        input_batch1,input_batch2=get_2Dbatch(pos_x,pos_y)
        out=sess.run([OUT],feed_dict={XP1:input_batch1,XP2:input_batch2})
        out=np.array(out)
        Endpatch=half_patch+175
        OUTMHA[pos_x,pos_y+half_patch,half_patch:Endpatch]=np.argmax(out[0,:,:],1)
        print("inter=",iter/26250)
        iter=iter+1

print(out.shape)
util.save_nuarray_as_mha('VSD.Seg_HG_001.41163.mha', OUTMHA)
