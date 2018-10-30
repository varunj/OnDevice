import time
start_time = time.time()
import os 
import theano
import keras
import keras.backend as K
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

MODEL_PATH = 'trained_models/20181009-1100/frozen_model.hdf5'
img_rows, img_cols = 99,99
INP_IMG_PATH = '../data/fingertip_localization/cropped_images/'
INP_LABELS_PATH = '../data/fingertip_localization/labels_test/'


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
print('time taken upto label shuffling: ', (time.time()-start_time))

# labels2 = np.float16(labels[:,0:4])		# if output (?,4)
labels2 = np.float16(labels[:,0:2])			# if output (?,2)

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


model = load_model(MODEL_PATH)
scores = model.predict(imgs, verbose = 1)
diff = scores - labels2
diffabs = np.abs(diff)
total_error = np.mean(diffabs)
print ("Mean Absolute Error: %.2f%% pixels" % total_error)
np.savetxt('results.csv', scores, delimiter=',')


row_mean = diffabs.flatten()
count = np.zeros(100)
for k in range (0, 100):
	c = sum(v <= k for v in row_mean)
	count[k] = (c*100)/(28155*4)
print('count: ', count)

plt.title('Error/Pixel Curve')
plt.xlabel('Error')
plt.ylabel('Accuracy') 
plt.plot(range(0,100), count, marker='+')
plt.savefig('error.png')
