import numpy as np
import pickle
import random
from PIL import Image
import cv2
from keras.utils import np_utils
import config
from sklearn.metrics import classification_report
from keras import backend as K

server = config.server()

def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def getTrainData(keys,batch_size,classes,mode,train,opt_size,opt_size1=0): 
    """
    mode 1: RGB Stream
    mode 2: Optical Stream
    mode 3: RGB + Optical Stream
    """
    if server:
        data_folder_rgb = r'/home/oanhnt/thainh/data/rgb/'
        data_folder_opt = r'/home/oanhnt/thainh/data/opt{}/'.format(opt_size)
    else:
        data_folder_rgb = r'/mnt/smalldata/rgb/{}/'.format(train)
        data_folder_opt = r'/mnt/smalldata/opt/{}/'.format(train)

    while 1:
        for i in range(0, len(keys), batch_size):
            if mode == 1:
                X_train,Y_train=stackRGB(keys[i:i+batch_size],data_folder_rgb)

            elif mode == 2:
                X_train,Y_train=stackOpticalFlow(keys[i:i+batch_size],data_folder_opt,opt_size)

            elif mode == 3:
                X_train,Y_train=stackOpticalFlowRGB(keys[i:i+batch_size],data_folder_opt,data_folder_rgb,opt_size)

            elif mode == 4:
                X_train,Y_train=stackMultiple(keys[i:i+batch_size],data_folder_rgb,opt_size,opt_size1)
            else:
                X_train,Y_train=stackMultipleInput(keys[i:i+batch_size],opt_size)

            Y_train=np_utils.to_categorical(Y_train,classes)
            if train == 'test':
                print 'Test batch {}'.format(i/batch_size+1)
            yield X_train, np.array(Y_train)

def getClassData(keys):
    labels = []
    for opt in keys:
        labels.append(opt[2])

    return labels

def getScorePerVideo(result, data):

    indVideo = []
    dataVideo = []
    length = len(data)
    for i in range(length):
        name = data[i][0].split('/')[1]
        if name not in indVideo:
            indVideo.append(name)
            dataVideo.append([name,data[i][2],result[i], 1])
        else:
            index = indVideo.index(name)
            dataVideo[index][2] = dataVideo[index][2] + result[i]
            dataVideo[index][3] += 1

    resultVideo = []
    classVideo = []
    len_data = len(dataVideo)
    for i in range(len_data):
        pred = dataVideo[i][2] / dataVideo[i][3]
        resultVideo.append(pred)
        classVideo.append(dataVideo[i][1])

    resultVideoArr = np.array(resultVideo)
    classVideoArr = np.array(classVideo)

    y_classes = resultVideoArr.argmax(axis=-1)
    return (classification_report(classVideoArr, y_classes, digits=6))

def stackRGB(chunk,data_folder_rgb):
    labels = []
    stack_rgb = []
    for rgb in chunk:
        folder_rgb = rgb[0]
        start_rgb = rgb[1]
        labels.append(rgb[2])

        rgb = cv2.imread(data_folder_rgb + folder_rgb + '-' + str(start_rgb) + '.jpg')
        rgb = rgb.astype('float16',copy=False)
        rgb/=255

        stack_rgb.append(rgb)

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
        if (start_opt % 20 > 0):
            start_rgb = (int(np.floor(start_opt * opt_size / 20)) + 1 ) * 10
        else:
            start_rgb = int(start_opt * opt_size / 2)

        # Stack RGB
        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_rgb) + '.jpg')
        if not server:
            rgb = cv2.resize(rgb, (224, 224))
        if rgb is None:
            print opt
            break
        rgb = rgb.astype('float16',copy=False)
        rgb/=255

        # Stack optical flow
        for i in range(start_opt, start_opt + 20):
            img = cv2.imread(data_folder_opt + folder_opt  + '/' +  str(i) + '.jpg', 0)
            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (224, 224))
            arrays.append(resize_img)

        nstack = np.dstack(arrays)
        nstack = nstack.astype('float16',copy=False)
        nstack/=255
        
        # Stack chunk
        stack_rgb.append(rgb)
        stack_opt.append(nstack)

    return [np.array(stack_rgb), np.array(stack_opt)], labels

def stackMultiple(chunk,data_folder_rgb,opt_size,opt_size1):
    labels = []
    stack_opt1 = []
    stack_opt2 = []
    stack_opt4 = []
    stack_rgb = []
    data_folder_opt1 = '/home/oanhnt/thainh/data/opt1/'
    data_folder_opt2 = '/home/oanhnt/thainh/data/opt2/'
    data_folder_opt4 = '/home/oanhnt/thainh/data/opt4/'
    opts = [opt_size, opt_size1]
    returns = []

    for opt in chunk:
        folder_opt = opt[0]
        start_opt1 = opt[4]
        labels.append(opt[2])
        start_opt2 = opt[3]
        start_opt4 = opt[1]
        arrays1 = []
        arrays2 = []
        arrays4 = []

        rgb = cv2.imread(data_folder_rgb + folder_opt + '-' + str(start_opt1/2) + '.jpg')
        if not server:
            rgb = cv2.resize(rgb, (224, 224))
        if rgb is None:
            print opt
            break
        rgb = rgb.astype('float16',copy=False)
        rgb/=255
        stack_rgb.append(rgb)

        # Stack optical flow 1
        if 1 in opts:
            for i in range(start_opt1, start_opt1 + 20):
                img = cv2.imread(data_folder_opt1 + folder_opt  + '/' +  str(i) + '.jpg', 0)
                height, width = img.shape
                crop_pos = int((width-height)/2)
                img = img[:,crop_pos:crop_pos+height]
                resize_img = cv2.resize(img, (224, 224))
                arrays1.append(resize_img)

            nstack1 = np.dstack(arrays1)
            nstack1 = nstack1.astype('float16',copy=False)
            nstack1/=255
            
            stack_opt1.append(nstack1)

        # Stack optical flow 2
        if 2 in opts:
            for i in range(start_opt2, start_opt2 + 20):
                img = cv2.imread(data_folder_opt2 + folder_opt  + '/' +  str(i) + '.jpg', 0)
                height, width = img.shape
                crop_pos = int((width-height)/2)
                img = img[:,crop_pos:crop_pos+height]
                resize_img = cv2.resize(img, (224, 224))
                arrays2.append(resize_img)

            nstack2 = np.dstack(arrays2)
            nstack2 = nstack2.astype('float16',copy=False)
            nstack2/=255
            
            stack_opt2.append(nstack2)

        # Stack optical flow 1
        if 4 in opts:
            for i in range(start_opt4, start_opt4 + 20):
                img = cv2.imread(data_folder_opt4 + folder_opt  + '/' +  str(i) + '.jpg', 0)
                height, width = img.shape
                crop_pos = int((width-height)/2)
                img = img[:,crop_pos:crop_pos+height]
                resize_img = cv2.resize(img, (224, 224))
                arrays4.append(resize_img)

            nstack4 = np.dstack(arrays4)
            nstack4 = nstack4.astype('float16',copy=False)
            nstack4/=255
            
            stack_opt4.append(nstack4)

    returns.append(np.array(stack_rgb))
    if 1 in opts:
        returns.append(np.array(stack_opt1))
    if 2 in opts:
        returns.append(np.array(stack_opt2))
    if 4 in opts:
        returns.append(np.array(stack_opt4))

    return returns, labels

def stackMultipleInput(chunk,opt_size):
    labels = []
    stack_opt = []
    data_folder_opt1 = '/home/oanhnt/thainh/data/opt1/'
    data_folder_opt2 = '/home/oanhnt/thainh/data/opt2/'
    data_folder_opt4 = '/home/oanhnt/thainh/data/opt4/'

    for opt in chunk:
        folder_opt = opt[0]
        start_opt1 = opt[4]
        labels.append(opt[2])
        start_opt2 = opt[3]
        start_opt4 = opt[1]
        arrays = []

        # Stack optical flow 1
        for i in range(start_opt1, start_opt1 + 20):
            img = cv2.imread(data_folder_opt1 + folder_opt  + '/' +  str(i) + '.jpg', 0)
            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (224, 224))
            arrays.append(resize_img)

        # Stack optical flow 2
        for i in range(start_opt2, start_opt2 + 20):
            img = cv2.imread(data_folder_opt2 + folder_opt  + '/' +  str(i) + '.jpg', 0)
            height, width = img.shape
            crop_pos = int((width-height)/2)
            img = img[:,crop_pos:crop_pos+height]
            resize_img = cv2.resize(img, (224, 224))
            arrays.append(resize_img)

        # Stack optical flow 4
        for i in range(start_opt4, start_opt4 + 20):
            img = cv2.imread(data_folder_opt4 + folder_opt  + '/' +  str(i) + '.jpg', 0)
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

def convert_weights(weights, depth, size=3, ins=32):
    mat = weights[0]
    mat2 = np.empty([size,size,depth,ins])
    for i in range(ins):
        x=(mat[:,:,0,i] + mat[:,:,1,i] + mat[:,:,2,i])/3
        for j in range(depth):
            mat2[:,:,j,i] = x
    return [mat2]

def concat_weights(weights, depth, length):

    mat = []
    for i in range(depth):
        mat.extend(weights[0])

    mat2 = []
    for i in range(length):
        mat2.append(mat)
    
    return np.array(mat2)


def get_model_memory_usage(model, batch_size):

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes



