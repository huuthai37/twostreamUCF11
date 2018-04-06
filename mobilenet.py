import keras
import sys
from keras.models import Model
from keras.layers import Dense, Conv2D, Activation, Reshape, Flatten, Input, ZeroPadding2D, Average, GlobalAveragePooling2D

def mobilenet_by_me(name, inputs, input_shape, classes, weight = '', cut = 5):
	model = keras.applications.mobilenet.MobileNet(
	    include_top=True,
	    dropout=0.5
	)
	name = name + '_'

	# Disassemble layers
	layers = [l for l in model.layers]

	new_input = Input(shape=input_shape)
	x = ZeroPadding2D(padding=(1, 1))(new_input)
	x = Conv2D(filters=32, 
	          kernel_size=(3, 3),
	          padding='valid',
	          use_bias=False,
	          strides=(2,2))(x)

	for i in range(3, len(layers)-3):
	    layers[i].name = str(name) + layers[i].name
	    x = layers[i](x)

	x = Flatten()(x)
	x = Dense(classes, activation='softmax')(x)
	model2 = Model(inputs=new_input, outputs=x)
	if weight != '':
	    model2.load_weights(weight)

	new_layers = [l for l in model2.layers]

	y = inputs
	for i in range(0, len(new_layers)-cut):
	    y = new_layers[i](y)

	return y
