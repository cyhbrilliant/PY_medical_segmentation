import numpy as np
import utility as util
import PIL.Image as Image
#patient="picture/resize/VSD.Seg_HG_001.36563.mha"
patient="E://zhangjinjing/brain2D/error.mha"
#test="test210_resize_9.mha"
test="error.mha"
trainpatient="BrainTrain/210/0001/0001.mha"
def colorize():
    file=util.read_mha_image_as_nuarray(test)
    newfile=np.zeros([file.shape[0],file.shape[1],file.shape[2],3])
    for i in range(file.shape[0]):
        for j in range(file.shape[1]):
            for k in range(file.shape[2]):
                if file[i, j, k] == 1:
                    newfile[i, j, k] = [192, 57, 43]
                elif file[i, j, k] == 2:
                    newfile[i, j, k] = [39, 174, 96]
                elif file[i, j, k] == 3:
                    newfile[i, j, k] = [155, 89, 182]
                elif file[i, j, k] == 4:
                    newfile[i, j, k] = [243, 156, 18]
                else:
                    newfile[i, j, k] = [0, 0, 0]
    for i in range(file.shape[2]):
        img=Image.fromarray(np.uint8(newfile[0:newfile.shape[0],0:newfile.shape[1],i]))
        img.save("picture/test9/green/test9-g"+str(i)+".jpg",'jpeg')
def adjust():
    file = util.read_mha_image_as_nuarray(trainpatient)
    for i in range(file.shape[1]):
        img=Image.fromarray(np.uint8(file[0:file.shape[0],0:file.shape[2],i]))
        img.save("picture/flair/green/flair"+str(i)+".jpg",'jpeg')
if __name__=="__main__":
    colorize()