import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import h5py
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import cv2

CSV_PATH = 'trained_models/20181009-1100/results.csv'
img_rows, img_cols = 99,99
INP_IMG_PATH = '../data/fingertip_localization/cropped_images/'
INP_LABELS_PATH = '../data/fingertip_localization/labels_test/'
OUT_IMG_PATH = 'test_images_out/'

count = 0
for filename in os.listdir(INP_LABELS_PATH):
	if filename.endswith(".csv"):
		gen_annotations = pd.read_csv(INP_LABELS_PATH+filename, header = None, sep=",")
		if (count == 0):
			labels = np.array(gen_annotations.ix[:,4])
		else:
			labels = np.append(labels, gen_annotations.ix[:,4],axis=0)
		count = count + 1
print('read labels')

pred = np.loadtxt(CSV_PATH, delimiter = ',')
pred = pred.astype(int)

count = 0
for i in range(0,len(pred)):
	img = cv2.imread(INP_IMG_PATH + labels[i][:-4]+'_crop.png')
	cv2.circle(img, (pred[i,0],pred[i,1]), 6, 25500, thickness=-1, lineType=8, shift=0)
	cv2.imwrite(OUT_IMG_PATH + labels[i][:-4]+'_crop.png', img)
	count = count + 1
	print('done img#: ', count)