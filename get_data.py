import numpy as np
import pickle
import random
from PIL import Image
import cv2
from keras.utils import np_utils
import config

server = config.server()

def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def getTrainData(keys,batch_size,classes,mode,train,opt_size): 
    """
    mode 1: RGB Stream
    mode 2: Optical Stream
    mode 3: RGB + Optical Stream
    """
    if server:
        data_folder_rgb = r'/home/oanhnt/thainh/data/rgb/{}/'.format(train)
        data_folder_opt = r'/home/oanhnt/thainh/data/opt{}/{}/'.format(opt_size, train)
    else:
        data_folder_rgb = r'/mnt/smalldata/rgb/{}/'.format(train)
        data_folder_opt = r'/mnt/smalldata/opt/{}/'.format(train)

    while 1:
        for i in range(0, len(keys), batch_size):
            if mode == 1:
                X_train,Y_train=stackRGB(keys[i:i+batch_size],data_folder_rgb)

            elif mode == 2:
                X_train,Y_train=stackOpticalFlow(keys[i:i+batch_size],data_folder_opt,opt_size)

            else:
                X_train,Y_train=stackOpticalFlowRGB(keys[i:i+batch_size],data_folder_opt,data_folder_rgb,opt_size)

            Y_train=np_utils.to_categorical(Y_train,classes)
            if train == 'test':
                print 'Test batch {}'.format(i/batch_size+1)
            yield X_train, np.array(Y_train)

def stackRGB(chunk,data_folder_rgb):
    labels = []
    stack_rgb = []
    for opt in chunk:
        folder_opt = opt[0]
        start_opt = opt[1]
        labels.append(opt[2])

        if (start_opt%20>0):
            start_rgb = (int(np.floor(start_opt/20)) + 1 ) * 10
        else:
            start_rgb = int(start_opt/2)
        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_rgb) + '.jpg')
        # resize_rgb = cv2.resize(rgb, (224, 224))
        resize_rgb = resize_rgb.astype('float16',copy=False)
        resize_rgb/=255

        stack_rgb.append(resize_rgb)

    return (np.array(stack_rgb), labels)

def stackOpticalFlow(chunk,data_folder,opt_size):
    labels = []
    stack_opt = []
    for opt in chunk:
        folder_opt = opt[0] + '/'
        start_opt = opt[1]
        labels.append(opt[2])
        arrays = []

        for i in range(start_opt, start_opt + 20):
            img = cv2.imread(data_folder + folder_opt + str(i) + '.jpg', 0)

            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (224, 224))
            arrays.append(resize_img)

        nstack = np.dstack(arrays)
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        stack_opt.append(nstack)

    return (np.array(stack_opt), labels)

def stackOpticalFlowRGB(chunk,data_folder_opt,data_folder_rgb,opt_size):
    labels = []
    stack_opt = []
    stack_rgb = []
    for opt in chunk:
        folder_opt = opt[0]
        start_opt = opt[1]
        labels.append(opt[2])
        arrays = []

        # RGB Frame
        if (start_opt%20>0):
            start_rgb = (int(np.floor(start_opt/20)) + 1 ) * 10 *opt_size
        else:
            start_rgb = int(start_opt * opt_size / 2)

        # Stack RGB
        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_rgb) + '.jpg')
        resize_rgb = cv2.resize(rgb, (224, 224))
        resize_rgb = resize_rgb.astype('float16',copy=False)
        resize_rgb/=255

        # Stack optical flow
        for i in range(start_opt, start_opt + 20):
            img = cv2.imread(data_folder_opt + folder_opt  + '/' +  str(i) + '.jpg', 0)
#             print img.shape
            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (224, 224))
            arrays.append(resize_img)

        nstack = np.dstack(arrays)
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        
        # Stack chunk
        stack_rgb.append(resize_rgb)
        stack_opt.append(nstack)

    return [np.array(stack_rgb), np.array(stack_opt)], labels

def convert_weights(weights, depth):
    mat = weights[0]
    mat2 = np.empty([3,3,depth,32])
    for i in range(32):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]
