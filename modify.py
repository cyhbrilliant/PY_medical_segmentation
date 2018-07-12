import numpy as np
import utility as util

modi210=util.read_mha_image_as_nuarray('test210.mha')
temp=np.zeros([155,240,240])
temp[:,32:207,32:207]=modi210
util.save_nuarray_as_mha('test210_2.mha', temp)


