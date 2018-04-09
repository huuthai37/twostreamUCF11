import cv2
import os
import sys
import random
import numpy as np
import pickle
import config

# data_opt.py train 1 run

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

train = sys.argv[1]
sample_rate = int(sys.argv[2])
if sample_rate == 1:
    opt_rate = 10
else:
    opt_rate = 5

if sys.argv[3] == 'run':
    debug = False
else: 
    debug = True

server = config.server()

if server:
    data_output_folder = r'/home/oanhnt/thainh/data/opt{}/'.format(sample_rate)
    text_file = r'data/{}list.txt'.format(train)
    class_file = r'data/classInd.txt'
    data_input_folder = '/home/oanhnt/thainh/UCF-11/'
    out_file = r'/home/oanhnt/thainh/data/database/{}-opt{}.pickle'.format(train,sample_rate)
else:
    data_output_folder = r'/mnt/data-11new/opt{}/{}/'.format(sample_rate,train)
    out_file = r'/mnt/data-11new/database/{}-opt{}.pickle'.format(train,sample_rate)
    text_file = r'data/{}list.txt'.format(train)
    class_file = r'data/classInd.txt'
    data_input_folder = '/mnt/UCF-11/'

data=[]
classInd=[]

# Tao class index tu file classInd.txt
with open(class_file) as f0:
    for line in f0:
        class_name = line.rstrip()
        if class_name:
            classInd.append(class_name)

# Khoi tao tinh optical flow bang Dual TVL1
c = 0
# optical_flow = cv2.DualTVL1OpticalFlow_create()
inst = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst.setUseSpatialPropagation(True)

with open(text_file) as f1:
    for line in f1:
        # tao ten va folder anh
        if train == 'test':
            arr_line = line.rstrip() # return folder/subfolder/name.mpg
        else:
            arr_line = line.split(' ')[0]
        path_video = arr_line.split('/') # return array (folder,subfolder,name.mpg)
        num_name = len(path_video) # return 3
        name_video = path_video[num_name - 1].split('.')[0] # return name ko co .mpg
        folder_video = path_video[0] # return folder
        path = data_output_folder + folder_video + '/' # return data-output/folder/
        video_class = classInd.index(folder_video) # index cua video

        if not os.path.isdir(path):
            os.makedirs(path) # tao data-ouput/folder/
            print 'make dir ' + path

        if not os.path.isdir(path + name_video):
            os.makedirs(path + name_video) #tao data-output/foldet/name/
            print 'make dir ' + path + name_video

        cap = cv2.VideoCapture(data_input_folder + arr_line)
        ret, frame1 = cap.read()
        if not ret:
            continue
            print 'out'
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        k = 0
        m = 0
        os.chdir(path + name_video)
        while(True):  
            # Capture frame-by-frame
            ret, frame2 = cap.read()
            if not ret:
                break;

            if m%sample_rate == 0:

                if (k%opt_rate == 0) & (k > 9):
                    data.append([folder_video + '/' + name_video, 2*(k-10), video_class])
                    c+=1
                if not debug:
                    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                    
                    # flow = optical_flow.calc(prvs, next, None)
                    # flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow = inst.calc(prvs, next, None)
            
                    prvs = next

                    # Chuan hoa gia tri diem anh ve tu 0 den 255
                    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)

                    # Chuyen kieu ve int8
                    horz = horz.astype('uint8')
                    vert = vert.astype('uint8')

                    # Ghi anh
                    cv2.imwrite(str(2*k)+'.jpg',horz,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    cv2.imwrite(str(2*k+1)+'.jpg',vert,[int(cv2.IMWRITE_JPEG_QUALITY), 90])

                k+=1

            m+=1

        if ((k%opt_rate > int(opt_rate/2)) | (k%opt_rate == 0)) & (k > 9):
            data.append([folder_video + '/' + name_video, 2*(k-10), video_class])
            c+=1

        print name_video

        # Giai phong capture
        cap.release()

print 'Generate opt: {} samples for {} dataset'.format(c,train)

# Ghi du lieu dia chi ra file
with open(out_file,'wb') as f2:
    pickle.dump(data,f2)
