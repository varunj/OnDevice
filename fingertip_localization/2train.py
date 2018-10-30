import time
start_time = time.time()
import os 
import keras
import keras.backend as K
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, Callback
import pickle
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from pylab import figure, axes, pie, title, show
import scipy.misc 
import cv2


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

OUTPUT_PATH = 'trained_models/20181009-1100/'
INP_IMG_PATH = '../data/fingertip_localization/cropped_images/'
INP_LABELS_PATH = '../data/fingertip_localization/labels_train/'

img_rows, img_cols = 99,99



model = Sequential()
model.add(Convolution2D(24, (3,3), strides = (2,2), input_shape=(99,99,3), border_mode='same', data_format= 'channels_last', activation='relu', init='uniform'))
# print(model.layers[-1].output_shape)
model.add(Convolution2D(24, (3,3), border_mode='same', activation='relu', init='uniform'))
model.add(Convolution2D(24, (3,3), border_mode='same', activation='relu', init='uniform'))
model.add(MaxPooling2D( pool_size=(2,2) ))
#model.add(Dropout(0.25))
model.add(Convolution2D(48, (3,3), strides = (2,2), border_mode='same', activation='relu', init='uniform'))
model.add(Convolution2D(48, (3,3), border_mode='same', activation='relu', init='uniform'))
model.add(Convolution2D(48, (3,3), border_mode='same', activation='relu', init='uniform'))
model.add(MaxPooling2D(pool_size=(2,2) ))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu', init='uniform'))
model.add(Dense(128, activation='relu', init='uniform'))
#model.add(Dropout(0.25))
model.add(Dense(2, init='uniform'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc','mae'])



def mean_pred(y_true, y_pred):
	return K.mean(y_pred)



def shuffle_data(num_samples, labels):
	seq = np.arange(num_samples)
	np.random.shuffle(seq)
	temp = 0
	for k in seq:
		if (temp == 0):
			result = labels[k:k+1,:]
		else:
			result = np.vstack((result, labels[k:k+1,:]))
		temp = temp + 1

	return result



def make_chan_first(path):
	img = cv2.imread(path)
	a = np.array(img[:,:,0])
	b = np.array(img[:,:,1])
	c = np.array(img[:,:,2])
	a = np.reshape(a, (1,img_rows,img_cols))
	b = np.reshape(b, (1,img_rows,img_cols))
	c = np.reshape(c, (1,img_rows,img_cols))
	img1 = np.append(a,b, axis=0)
	chan_fst_img = np.append(img1, c, axis =0)
	return chan_fst_img		


def make_chan_last(path):
	img = cv2.imread(path)
	a = np.array(img[:,:,0])
	b = np.array(img[:,:,1])
	c = np.array(img[:,:,2])
	a = np.reshape(a, (img_rows,img_cols,1))
	b = np.reshape(b, (img_rows,img_cols,1))
	c = np.reshape(c, (img_rows,img_cols,1))
	img1 = np.append(a,b, axis=2)
	chan_lst_img = np.append(img1, c, axis =2)
	return chan_lst_img		



labels = []
count = 0
for filename in os.listdir(INP_LABELS_PATH):
	if filename.endswith('.csv'):
		gen_annotations = pd.read_csv(INP_LABELS_PATH+filename, header = None, sep=',')
		if (count == 0):
			labels = np.array(gen_annotations.ix[:,0:5])
		else:	
			labels = np.append(labels, gen_annotations.ix[:,0:5],axis=0)
		count = count + 1



count = 0
labels = shuffle_data(len(labels),labels)
print('time taken upto label shuffling: ', (time.time()-start_time))

labels2 = np.float16(labels[:,0:4])		# if output (?,4)
# labels2 = np.float16(labels[:,0:2])			# if output (?,2)

imgArr = []
for img in labels[:,4]:
	img = img.split('.png')[0]+'_crop.png'
	if (count == 0):
		im1 = make_chan_last(INP_IMG_PATH + img)
		im1 = np.float16(im1)
		imgstack = im1.reshape(1,im1.shape[0],im1.shape[1],im1.shape[2])
		imgArr.append(imgstack)
	else:
		im2 = make_chan_last(INP_IMG_PATH + img)
		im2 = np.float16(im2)
		imgstack = im2.reshape(1,im1.shape[0],im1.shape[1],im1.shape[2])
		imgArr.append(imgstack)
	count = count + 1
imgstack = np.vstack(imgArr)
imgs = np.float16(imgstack)
imgs = imgs/255
print('time taken upto imgs loading: ', (time.time()-start_time))



class LossHistory(keras.callbacks.Callback):
 	def on_train_begin(self, logs={}):
		self.losses = []

	def on_epoch_end(self, batch, logs={}):
		print('predicted values are: ', model.predict(imgs[100:105,:,:,:]))
		print('actual labels are: ', labels[100:105,:])



history = LossHistory()
filepath = OUTPUT_PATH + 'frozen_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(imgs, labels2, nb_epoch=150, batch_size=300, validation_split=0.25, callbacks=callbacks_list, verbose = 1)
print(history.history.keys())



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(OUTPUT_PATH + 'accuracy.png', dpi=100)
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(OUTPUT_PATH + 'loss.png', 	dpi=100)
plt.close()

print('time taken to complete training: ', (time.time()-start_time))
