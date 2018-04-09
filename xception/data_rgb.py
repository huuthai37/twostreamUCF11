import cv2
import os
import sys
import random
import numpy as np
import config
import pickle

# data_rgb.py train 10 run 

def crop_image(frame, name_video, i, y, x):
    crop = frame[y:y+224, x:x+224].copy()
    cv2.imwrite(r'{}-{}-{}{}.jpg'.format(name_video, i, y, x), crop)
    crop_flip = cv2.flip(crop, 1)
    cv2.imwrite(r'{}-{}-{}{}-flh.jpg'.format(name_video, i, y, x), crop_flip)

# Lay tuy chon 
train = sys.argv[1]
sample_rate = int(sys.argv[2])
if sys.argv[3] == 'run':
    debug = False
else:
    debug = True

# Cau hinh folder du lieu
server = config.server()
if server:
    data_output_folder = r'/home/oanhnt/thainh/data/rgb/{}/'.format(train)
    data_input_folder = '/home/oanhnt/thainh/UCF-11/'
    out_file = r'/home/oanhnt/thainh/data/database/{}-rgb.pickle'.format(train)
else:
    data_output_folder = r'/mnt/data-11new/rgb/{}/'.format(train)
    data_input_folder = '/mnt/UCF-11/'
    out_file = r'/mnt/data-11new/database/{}-rgb.pickle'.format(train)

text_file = r'data/{}list.txt'.format(train)
class_file = r'data/classInd.txt'
count = 0
v = 0

data=[]
classInd=[]

# Tao class index tu file classInd.txt
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)


with open(text_file) as f:
    for line in f:
        # Tao duong dan va ten file anh
        if train != 'test':
            arr_line = line.split(' ')[0] # return folder/subfolder/name.mpg
        else:
            arr_line = line.rstrip()
        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0] # return folder
        path = data_output_folder + folder_video + '/' #return data-output/folder/
        video_class = classInd.index(folder_video) # index cua video

        # tao folder moi neu chua ton tai
        if not os.path.isdir(path):
            os.makedirs(path)
            print 'Created folder: {}'.format(folder_video)

        cap = cv2.VideoCapture(data_input_folder + arr_line)
        v += 1
        i = -1
        os.chdir(path)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
                cap.release()
                sys.exit()
            i = i + 1
            if (i%sample_rate != 0):
                continue
            
            if not debug:
                height, width, channel = frame.shape

                #random crop max size (heightxheight)
                x = random.randint(0, (width - height))
                crop = frame[:, x:x+height].copy()

                resize_img = cv2.resize(crop, (224, 224))

                cv2.imwrite(r'{}-{}.jpg'.format(name_video, i),resize_img)

            data.append([folder_video + '/' + name_video, i, video_class])

            count += 1

        
        print name_video
        # Giai phong capture
        cap.release()
print 'Generate RGB: {} samples for {} dataset with {} videos'.format(count,train,v)

# Ghi du lieu dia chi ra file
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)
