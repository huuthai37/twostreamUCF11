import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D
import get_data as gd
from keras import optimizers
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report

# train: python mobilenet_temporal.py train 1 32 1 101
# test: python mobilenet_temporal.py test 1 32 1 101
# retrain: python mobilenet_temporal.py retrain 1 32 1 101 1

if sys.argv[1] == 'train':
    train = True
    retrain = False
    old_epochs = 0
elif sys.argv[1] == 'retrain':
    train = True
    retrain = True
    old_epochs = int(sys.argv[6])
else:
    train = False
    retrain = False

opt_size = int(sys.argv[2])
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
classes = int(sys.argv[5])

depth = 60
input_shape = (224,224,depth)

server = config.server()
if server:
    if train:
        out_file = '/home/oanhnt/thainh/data/database/train-all.pickle'
    else:
        out_file = '/home/oanhnt/thainh/data/database/test-all.pickle'
    valid_file = '/home/oanhnt/thainh/data/database/test-all.pickle'
else:
    if train:
        out_file = r'/mnt/smalldata/database/train-opt{}.pickle'.format(opt_size)
    else:
        out_file = r'/mnt/smalldata/database/test-opt{}.pickle'.format(opt_size)

# MobileNet model
if train & (not retrain):
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        weights='imagenet',
        dropout=0.5
    )
else:
    model = keras.applications.mobilenet.MobileNet(
        include_top=True,  
        dropout=0.5
    )

# Disassemble layers
layers = [l for l in model.layers]

# Defining new convolutional layer.
input_new = Input(shape=input_shape)
x = ZeroPadding2D(padding=(1, 1), name='conv1_pad')(input_new)
x = Conv2D(filters=32, 
          kernel_size=(3, 3),
          padding='valid',
          use_bias=False,
          strides=(2,2),
          name='conv_new')(x)

for i in range(3, len(layers)-3):
    x = layers[i](x)

x = Flatten()(x)
x = Dense(classes, activation='softmax', name='predictions')(x)

# Final touch
result_model = Model(inputs=input_new, outputs=x)

# Run
result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

if train:
    if retrain:
        result_model.load_weights('weights/mobilenet_temporal_mulinput_{}e.h5'.format(old_epochs))
    else:
        result_model.get_layer('conv_new').set_weights(gd.convert_weights(layers[2].get_weights(), depth))

    with open(out_file,'rb') as f1:
        keys = pickle.load(f1)
    len_samples = len(keys)

    if server:
        with open(valid_file,'rb') as f2:
            keys_valid = pickle.load(f2)
        len_valid = len(keys_valid)

    print('-'*40)
    print 'MobileNet Optical #{} stream only: Training'.format(opt_size)
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
                    5,
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
                    5,
                    'test',
                    opt_size),
                validation_steps=len_valid/batch_size
            )
            histories.append([
                history.history['acc'],
                history.history['val_acc'],
                history.history['loss'],
                history.history['val_loss']
            ])
        else:
            history = result_model.fit_generator(
                gd.getTrainData(
                    keys,
                    batch_size,
                    classes,
                    5,
                    'train',
                    opt_size), 
                verbose=1, 
                max_queue_size=2, 
                steps_per_epoch=3, 
                epochs=1
            )

            histories.append([
                history.history['acc'],
                history.history['loss']
            ])
        result_model.save_weights('weights/mobilenet_temporal_mulinput_{}e.h5'.format(old_epochs+1+e))

        with open('data/trainHistoryTemporal{}_{}_{}e'.format(opt_size, old_epochs, epochs), 'wb') as file_pi:
            pickle.dump(histories, file_pi)

else:
    result_model.load_weights('weights/mobilenet_temporal_mulinput_{}e.h5'.format(epochs))

    with open(out_file,'rb') as f2:
        keys = pickle.load(f2)
    len_samples = len(keys)
    print('-'*40)
    print 'MobileNet Optical #{} stream only: Testing'.format(opt_size)
    print('-'*40)
    print 'Number samples: {}'.format(len_samples)

    # score = result_model.evaluate_generator(
    #     gd.getTrainData(
    #         keys,
    #         batch_size,
    #         classes,
    #         2,
    #         'test',
    #         opt_size), 
    #     max_queue_size=3, 
    #     steps=len_samples/batch_size
    # )
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    Y_test = gd.getClassData(keys)
    y_pred = result_model.predict_generator(
        gd.getTrainData(
            keys,
            batch_size,
            classes,
            5,
            'test', 
            opt_size), 
        max_queue_size=3, 
        steps=int(np.ceil(len_samples*1.0/batch_size)))
    y_classes = y_pred.argmax(axis=-1)
    print 'Score per samples'
    print(classification_report(Y_test, y_classes, digits=6))

    print 'Score per video'
    print(gd.getScorePerVideo(y_pred, keys))
