import numpy as np 
import pickle
import sys
import cv2
import config

# check_data.py train opt2

train = sys.argv[1]
file = sys.argv[2]
opt_size = int(sys.argv[3])
server = config.server()

if server:
	dirr = '/home/oanhnt/thainh/data/database/'
	dir_data = '/home/oanhnt/thainh/data/rgb/{}/'.format(train)
else:
	dirr = '/mnt/data11/database/'
	dir_data = '/mnt/data11/{}/{}/'.format(file, train)

with open(dirr + train + '-' + file + '.pickle','rb') as f1:
    data = pickle.load(f1)
length = len(data)
print length
for i in range(length):
	fileimg = data[i][0]
	start_opt = data[i][1]

	start_rgb = (int(np.floor(start_opt * opt_size / 20)) + 1 ) * 10
	rgb = cv2.imread(dir_data + fileimg + '-' + str(start_rgb) + '.jpg')

	if (rgb is None):
		print data[i]
		print('Start', start_rgb)

	if i%1000 == 0:
		print 'Checked {}/{}'.format(i, length)

