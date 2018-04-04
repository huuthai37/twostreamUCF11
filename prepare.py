import os
import config

# Cau hinh folder du lieu
server = config.server()
if server:
    data_output_folder = '/home/oanhnt/thainh/data/'
else:
    data_output_folder = '/mnt/data-11test/'

folders = ['rgb', 'opt1', 'opt2', 'opt3']
sub_folders = ['train', 'test', 'valid']

for i in range(4):
    path = data_output_folder + folders[i]
    if not os.path.isdir(path):
        os.makedirs(path)
        print 'make dir ' + path

    for j in range(3):
        sub_path = path + '/' + sub_folders[j]
        if not os.path.isdir(sub_path):
            os.makedirs(sub_path)
            print 'make dir ' + sub_path

print 'Done!'
