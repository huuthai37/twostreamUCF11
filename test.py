import sys
import pickle
import random
import numpy as np
import config
from sklearn.metrics import classification_report
import mobilenet
import keras
from keras import optimizers
import get_data as gd
from keras.models import Model
from keras.layers import Dense, Concatenate, GlobalAveragePooling2D, Dropout, Reshape, Flatten, Input

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

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

classes = 11
depth = 20
drop_rate = 0.5

input_x = Input(shape=(224,224,3))

x = mobilenet.mobilenet_by_me(
    name='spatial', 
    inputs=input_x, 
    input_shape=(224,224,3), 
    classes=classes)

# Temporal
input_y1 = Input(shape=(224,224,20))
input_y2 = Input(shape=(224,224,20))
input_y3 = Input(shape=(224,224,20))
inputs = [input_y1,input_y2,input_y3]

y = mobilenet.mobilenet_new(
    name='temporal', 
    inputs=inputs, 
    input_shape=(224,224,20), 
    classes=classes)

z = Concatenate()([x, y])
z = GlobalAveragePooling2D()(z)
z = Reshape((1,1,2048))(z)
z = Dropout(0.5)(z)
z = Flatten()(z)
z = Dense(classes, activation='softmax')(z)
# Final touch
result_model = Model(inputs=[input_x,input_y1,input_y2,input_y3], outputs=z)
result_model.summary()

result_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

opt_size = 2
batch_size = int(sys.argv[1])
out_file = r'/home/oanhnt/thainh/data/database/train-opt{}.pickle'.format(opt_size)
with open(out_file,'rb') as f1:
    keys = pickle.load(f1)
len_samples = len(keys)

result_model.fit_generator(
    gd.getTrainData(
        keys,
        batch_size,
        classes,
        4,
        'train', 
        opt_size), 
    verbose=1, 
    max_queue_size=2, 
    steps_per_epoch=len_samples/batch_size, 
    epochs=1,
)


