import utility as util
import tensorflow as tf
import os
import numpy as np
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def weight_variable(shape):
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape,stddev=0.1),name='weight')
def bias_variable(shape):
    return tf.Variable(initial_value=tf.constant(value=0.1,shape=shape),name='bias')
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
def Getcoord(data):
    index=[]
    for i in range(data.shape[0]):
            for j in range(data.shape[1]-64):
                for k in range(data.shape[2]-64):
                    index.append([i,j+32,k+32])
    return np.array(index)

def Getbatch(data,coord):
    halfpatch1=32
    halfpatch2=16
    Input_batch1=[]
    Input_batch2=[]
    for i in range(175):
        x=coord[i,0]
        y=coord[i,1]
        z=coord[i,2]
        print('x,y,z:',x,y,z)
        Input_batch1.append(data[x, y - halfpatch1:y + halfpatch1 + 1, z - halfpatch1:z + halfpatch1 + 1, :])
        Input_batch2.append(data[x, y - halfpatch2:y + halfpatch2 + 1, z - halfpatch2:z + halfpatch2 + 1, :])
    return np.array(Input_batch1),np.array(Input_batch2)


def Network(X1,X2):
    weights = {'w_conv1': weight_variable(shape=[7, 7, 8, 64]), 'w_conv2': weight_variable(shape=[3, 3, 64, 64]),'w_conv3': weight_variable(shape=[13, 13, 8, 160]), 'w_conv4': weight_variable(shape=[21, 21, 224, 5])}
    bias = {'b_conv1': bias_variable(shape=[64]), 'b_conv2': bias_variable(shape=[64]),'b_conv3': bias_variable(shape=[160]), 'b_conv4': bias_variable(shape=[5])}
    downsampling = tf.image.resize_nearest_neighbor(X1, size=[33, 33])
    INPUT = tf.concat([downsampling, X2], 3)
    conv1_1 = tf.nn.relu(tf.nn.conv2d(INPUT, weights['w_conv1'], strides=[1, 1, 1, 1], padding='VALID', name='localconv1') + bias['b_conv1'])
    maxpooling_1 = tf.nn.max_pool(conv1_1, ksize=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
    conv1_2 = tf.nn.relu(tf.nn.conv2d(maxpooling_1, weights['w_conv2'], strides=[1, 1, 1, 1], padding='VALID', name='localconv2') + bias['b_conv2'])
    maxpooling_2 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
    conv2_1 = tf.nn.relu(tf.nn.conv2d(INPUT, weights['w_conv3'], strides=[1, 1, 1, 1], padding='VALID', name='globalconv') + bias['b_conv3'])
    Combine = tf.nn.conv2d(tf.concat([conv2_1, maxpooling_2], 3), weights['w_conv4'], strides=[1, 1, 1, 1],padding='VALID') + bias['b_conv4']
    OUT = tf.nn.softmax(tf.reshape(Combine, [175, 5]))
    return OUT
def Segment():
    X1 = tf.placeholder(dtype=tf.float32, shape=[None, 65, 65, 4], name='big_patch')
    X2 = tf.placeholder(dtype=tf.float32, shape=[None, 33, 33, 4], name='small_patch')
    OUT = Network(X1, X2)
    patient = readindex()
    patient = np.array(patient)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, 'E:/zhangjinjing/brain2D/brats2013/brain_session_2013_8.ckpt')
    for i in range(25):
        data = DataProcess(i)
        print('data shape:'+str(data.shape))
        OUTMHA = np.zeros([data.shape[0], data.shape[1], data.shape[2]])
        print('MHA:',OUTMHA.shape)
        index=Getcoord(data)
        print('index:',index.shape)
        n=0
        while (n<index.shape[0]-175):
            subindex=np.array(index[n:n+175])
            print('sunindex:',subindex.shape)
            print('n：',n)
            x1, x2 = Getbatch(data,subindex)
            n=n+175
            OUTLINE=sess.run(OUT,feed_dict={X1:x1,X2:x2})
            OUTLINE=np.array(OUTLINE)
            #print('OUTLINE:',OUTLINE.shape)
            for s in range(175):
                OUTMHA[subindex[s,0],subindex[s,1],subindex[s,2]]=np.argmax(OUTLINE,1)[s]
                print('subindex:',subindex[s,0],subindex[s,1],subindex[s,2])
                print('max:',np.argmax(OUTLINE,1)[s])
            print('finish!')
        util.save_nuarray_as_mha('E://zhangjinjing/brain2D/2013test/VSD.Seg_HG_001.'+str(patient[i])+'.mha',OUTMHA)



if __name__=='__main__':
  Segment()