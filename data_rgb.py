import cv2
import os
import sys
import random
import numpy as np
import config

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
else:
    data_output_folder = r'/mnt/data-11new/rgb/{}/'.format(train)
    data_input_folder = '/mnt/UCF-11/'

text_file = r'data/{}list.txt'.format(train)
count = 0
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

        # tao folder moi neu chua ton tai
        if not os.path.isdir(path):
            os.makedirs(path)
            print 'Created folder: {}'.format(folder_video)

        cap = cv2.VideoCapture(data_input_folder + arr_line)
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
            
            height, width, channel = frame.shape

            #random crop max size (heightxheight)
            x = random.randint(0, (width - height))
            crop = frame[:, x:x+height].copy()

            if debug & (i == 0):
                print crop.shape

            resize_img = cv2.resize(crop, (224, 224))

            if not debug:
                cv2.imwrite(r'{}-{}.jpg'.format(name_video, i),resize_img)
            
            crop_flip = resize_img.copy()
            crop_flip = cv2.flip(crop_flip, 1)

            if not debug:
                cv2.imwrite(r'{}-{}-flip.jpg'.format(name_video, i),crop_flip)
            count += 2

        
        print name_video
        # Giai phong capture
        cap.release()
print 'Generate RGB: {} samples for {} dataset'.format(count,train)
