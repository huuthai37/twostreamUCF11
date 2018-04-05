import numpy as np 
import pickle
import sys
import cv2
import config

# check_data.py train opt1 opt

train = sys.argv[1]
file = sys.argv[2]
type_data = sys.argv[3]
server = config.server()

if server:
	dirr = '/home/oanhnt/thainh/data/database/'
	dir_data = '/home/oanhnt/thainh/data/{}/{}/'.format(file, train)
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
	if (i>0) & (data[i] == data[i-1]):
		print ('Duplicate', i)
	for j in range(start_opt, start_opt + 20):
		if type_data != 'rgb':
			img = cv2.imread(dir_data + fileimg + '/' + str(j) + '.jpg')
		else:
			img = cv2.imread(dir_data + fileimg + '-' + str(j) + '.jpg')
		
		height, width, channels = img.shape
		if type_data == 'rgb':
			if (img is None) | ((height != 224) & (width != 224)):
				print data[i]
				print img.shape
				break
		else:
			if (img is None):
				print data[i]
				break
	if i%1000 == 0:
		print 'Checked {}/{}'.format(i, length)

