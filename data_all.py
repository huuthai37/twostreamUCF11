import numpy as np 
import pickle
import sys
import config

train = sys.argv[1]
debug = sys.argv[2]
server = config.server()

if server:
    opt_file = r'/home/oanhnt/thainh/data/database/{}-opt4.pickle'.format(train)
    out_file = r'/home/oanhnt/thainh/data/database/{}-all.pickle'.format(train)

with open(opt_file,'rb') as f1:
    opt = pickle.load(f1)

l = len(opt)
opt_all = []
for i in range(l):
		x = int(np.floor(opt[i][1]*1.0/5))*5
		opt_all.append([opt[i][0], opt[i][1], opt[i][2], (x*2), (x*4)])
		opt_all.append([opt[i][0], opt[i][1], opt[i][2], (x*2 + 10), ((x * 2 + 10) * 2)])

print len(opt_all)
if debug == 'run':
	with open(out_file,'wb') as f3:
	    pickle.dump(opt_all,f3)