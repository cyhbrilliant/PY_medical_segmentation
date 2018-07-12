import utility as util
import tensorflow as tf
import os
import numpy as np
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    f=open("E://zhangjinjing/brain2D/BRATS_Leaderboard/number.txt")
    data=[]
    a=f.readline()
    while a:
        data.append(a.rstrip('\n'))
        a=f.readline()
    return data
def DataProcess(zip_num):
        TestPath = 'E://zhangjinjing/Brain2D/BRATS_Leaderboard/' + str(zip_num + 1)
        CH1 = util.read_mha_image_as_nuarray(TestPath + '/1/1.mha')
        CH2 = util.read_mha_image_as_nuarray(TestPath + '/2/2.mha')
        CH3 = util.read_mha_image_as_nuarray(TestPath + '/3/3.mha')
        CH4 = util.read_mha_image_as_nuarray(TestPath + '/4/4.mha')
        ALLtest = np.concatenate([CH1[:, :, :, np.newaxis],
                                  CH2[:, :, :, np.newaxis],
                                  CH3[:, :, :, np.newaxis],
                                  CH4[:, :, :, np.newaxis]], 3)

        del CH1, CH2, CH3, CH4
        gc.collect()
        return ALLtest
def Getbatch(ALLtest,x,y):
    largepatch=65
    smallpatch=33
    halfpatch1=int(largepatch/2)
    halfpatch2=int(smallpatch/2)
    Input_batch1 = []
    Input_batch2 = []
    for i in range(ALLtest.shape[2]-largepatch+1):
        z=i+halfpatch1
        Input_batch1.append(ALLtest[x, y - halfpatch1:y + halfpatch1 + 1, z - halfpatch1:z + halfpatch1 + 1, :])
        Input_batch2.append(ALLtest[x, y - halfpatch2:y + halfpatch2 + 1, z - halfpatch2:z + halfpatch2 + 1, :])
    return np.array(Input_batch1),np.array(Input_batch2)


def Network(x,number):
    weights = {'w_conv1': weight_variable(shape=[17, 17, 4, 10]), 'w_conv2': weight_variable(shape=[7, 7, 10, 64]),
               'w_conv3': weight_variable(shape=[3, 3, 64, 64]),
               'w_conv4': weight_variable(shape=[13, 13, 10, 160]), 'w_conv5': weight_variable(shape=[21, 21, 224, 5])}
    bias = {'b_conv1': bias_variable(shape=[10]), 'b_conv2': bias_variable(shape=[64]),
            'b_conv3': bias_variable(shape=[64]),
            'b_conv4': bias_variable(shape=[160]), 'b_conv5': bias_variable(shape=[5])}
    preconv = atrous_conv2d(x, weights['w_conv1']) + bias['b_conv1']
    conv1_1 = tf.nn.relu(conv_2d(preconv, weights['w_conv2']))
    maxpool1_1 = tf.nn.relu(maxpool(conv1_1, [1, 4, 4, 1]))
    conv2_1 = tf.nn.relu(conv_2d(preconv, weights['w_conv4']) + bias['b_conv4'])
    conv1_2 = tf.nn.relu(conv_2d(maxpool1_1, weights['w_conv3']) + bias['b_conv3'])
    maxpool1_2 = maxpool(conv1_2, [1, 2, 2, 1])
    Combine = conv_2d(tf.concat([maxpool1_2, conv2_1], 3), weights['w_conv5']) + bias['b_conv5']
    OUT = tf.nn.softmax(tf.reshape(Combine, [175, 5]))
    return OUT
def Segment():
    patient = readindex()
    patient = np.array(patient)
    for i in range(25):
        data = DataProcess(i+1)
        print('data shape:'+str(data.shape))
        OUTMHA = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
        for x in range(data.shape[0]):
            for y in range(data.shape[1]-64):
                    x1, x2 = Getbatch(data,x,y+32)
                    print('x1,x2:'+str(x1.shape)+str(x2.shape))
                    X1 = tf.placeholder(dtype=tf.float32, shape=[None, 65, 65, 4], name='big_patch')
                    X2 = tf.placeholder(dtype=tf.float32, shape=[None, 33, 33, 4], name='small_patch')
                    OUT=Network(X1,X2,data.shape[2]-64)
                    config=tf.ConfigProto()
                    config.gpu_options.allow_growth = True
                    sess = tf.Session(config=config)
                    saver=tf.train.Saver()
                    saver.restore(sess=sess, save_path='E:/zhangjinjing/brain2D/brats2013/brain_session_2013_kongdong.ckpt')
                    OUT=sess.run(OUT,feed_dict={X1:x1,X2:x2})
                    OUTMHA[x,y,32:data.shape[2]-32]=np.argmax(OUT,1)
        print(OUTMHA.shape)
        util.save_nuarray_as_mha('E://zhangjinjing/brain2D/2013test/VSD.Seg_HG_001.'+str(patient[i])+'.mha',OUTMHA)



if __name__=='__main__':
  Segment()