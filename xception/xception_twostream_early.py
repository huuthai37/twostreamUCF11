import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Dropout, Flatten, Input, ZeroPadding2D, Average, Multiply, Maximum, GlobalAveragePooling2D, Concatenate
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report
import xception

# train: python xception_twostream_early.py train 1 avg 32 1 101 0 0
# test: python xception_twostream_early.py test 1 avg 32 1 101
# retrain: python xception_twostream_early.py retrain 1 avg 32 1 101 1

if sys.argv[1] == 'train':
    train = True
    retrain = False
    old_epochs = 0
    spa_epochs = int(sys.argv[7])
    tem_epochs = int(sys.argv[8])
elif sys.argv[1] == 'retrain':
    train = True
    retrain = True
    old_epochs = int(sys.argv[7])
else:
    train = False
    retrain = False

opt_size = int(sys.argv[2])
fusion = sys.argv[3]
batch_size = int(sys.argv[4])
epochs = int(sys.argv[5])
classes = int(sys.argv[6])

drop_rate = 0.5
server = config.server()
if server:
    if train:
        out_file = r'/home/oanhnt/thainh/data/database/train-opt{}.pickle'.format(opt_size)
    else:
        out_file = r'/home/oanhnt/thainh/data/database/test-opt{}.pickle'.format(opt_size)
    valid_file = r'/home/oanhnt/thainh/data/database/test-opt{}.pickle'.format(opt_size)
else:
    if train:
        out_file = '/mnt/smalldata/database/train-opt2.pickle'
    else:
        out_file = '/mnt/smalldata/database/test-opt2.pickle'

# Spatial
temporal_model = xception.XceptionFix(
    input_shape=(299,299,20),
    classes=classes
)

spatial_model = xception.XceptionFix(
    input_shape=(299,299,3),
    classes=classes
)

if train & (not retrain) & server:
    spatial_model.load_weights('weights/xception_spatial_{}e.h5'.format(spa_epochs))
    temporal_model.load_weights('weights/xception_temporal{}_{}e.h5'.format(opt_size,tem_epochs))
    

# Fusion
if fusion == 'avg':
    x = Average()([spatial_model.layers[-4].output, temporal_model.layers[-4].output])
elif fusion == 'max':
    x = Maximum()([spatial_model.layers[-4].output, temporal_model.layers[-4].output])
elif fusion == 'concat':
    x = Concatenate()([spatial_model.layers[-4].output, temporal_model.layers[-4].output])
else:
    x = Multiply()([spatial_model.layers[-4].output, temporal_model.layers[-4].output])

x = GlobalAveragePooling2D()(x)
x = Dropout(drop_rate)(x)
x = Dense(classes, activation='softmax')(x)

# Final touch
result_model = Model(inputs=[spatial_model.input, temporal_model.input], outputs=x)

result_model.summary()

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/xception_twostream{}_early_{}_{}e.h5'.format(opt_size,fusion,old_epochs))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)
    if server:
        with open(valid_file,'rb') as f2:
            keys_valid = pickle.load(f2)
        len_valid = len(keys_valid)

    print('-'*40)
    print 'xception Two-Stream #{} {} mode: Training'.format(opt_size,fusion)
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)
    if server:
        print 'Number valid: {}'.format(len_valid)
    histories = []
    
    for e in range(epochs):
        print('-'*40)
        print('Epoch', e+1)
        print('-'*40)

        random.shuffle(keys)
        if server:
            history = result_model.fit_generator(
                gd.getTrainData(
                    keys,
                    batch_size,
                    classes,
                    3,
                    'train', 
                    opt_size), 
                verbose=1, 
                max_queue_size=2, 
                steps_per_epoch=len_samples/batch_size, 
                epochs=1,
                validation_data=gd.getTrainData(
                    keys_valid,
                    batch_size,
                    classes,
                    3,
                    'test', 
                    opt_size),
                validation_steps=int(np.ceil(len_valid*1.0/batch_size))
            )
            histories.append([
                history.history['acc'],
                history.history['val_acc'],
                history.history['loss'],
                history.history['val_loss']
            ])

        result_model.save_weights('weights/xception_twostream{}_early_{}_{}e.h5'.format(opt_size,fusion,old_epochs+1+e))

        if not retrain:
            with open('data/XceptionTwoStreamEarly{}_{}_{}e'.format(opt_size, fusion, epochs), 'wb') as file_pi:
                pickle.dump(histories, file_pi)

else:
    result_model.load_weights('weights/xception_twostream{}_early_{}_{}e.h5'.format(opt_size,fusion,epochs))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print('-'*40)
    print 'xception Two-Stream #{} {} mode: Testing'.format(opt_size,fusion)
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    Y_test = gd.getClassData(keys)
    y_pred = result_model.predict_generator(
        gd.getTrainData(
            keys,
            batch_size,
            classes,
            3,
            'test', 
            opt_size), 
        max_queue_size=3, 
        steps=int(np.ceil(len_samples*1.0/batch_size)))
    y_classes = y_pred.argmax(axis=-1)
    print 'Score per samples'
    print(classification_report(Y_test, y_classes, digits=6))

    print 'Score per video'
    print(gd.getScorePerVideo(y_pred, keys))