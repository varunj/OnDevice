'''
19157/19419 = 0.986508059% of the frames have hand detection

run /home/varunj/Github/OnDevice/classification_bi-lstm/test_new.py for results
'''
import numpy as np
np.set_printoptions(threshold=np.nan)
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import cv2
import os



IMGS_IN_PATH = 'testvideo_imgs'
IMGS_OUT_PATH = 'testvideo_imgs_cropped'
SEQ_OUT_PATH = 'testvideo_seq'
COOR_FILE_PATH = 'testvideo_imgs_coordinates.txt'



def cropAndSaveImage(imageName, boxes):
	'''
		imageName
		boxes [left, top, right, bottom, w, h]
	'''
	y = boxes[0]
	x = 480-boxes[1]
	h = boxes[4]
	w = boxes[5]
	img = cv2.imread(os.path.join(IMGS_IN_PATH, imageName))
	img = img[y:y+h, x:x+w]
	cv2.imwrite(os.path.join(IMGS_OUT_PATH, imageName),img)



def saveDetectionCrops():

	with open(COOR_FILE_PATH) as fp:  
		line = fp.readline()
		while line:
			line = line.replace('D: ', '').strip()
			imageName = line.split(':')[0]
			boxes = [int(x) for x in line.split(':')[1].split(',')]
			cropAndSaveImage(imageName, boxes)
			line = fp.readline()



def readDetectionSequencesToDic():
	'''
		dic
		{
			left : 	{ 	1: {1:(x,y), 2:(x,y) ...}
						2: {1:(x,y), 2:(x,y) ...}
					}
		}
	'''
	dic = {}
	with open(COOR_FILE_PATH) as fp:  
		line = fp.readline()
		while line:
			line = line.replace('D: ', '').strip()
			imageName = line.split(':')[0]
			boxes = [int(x) for x in line.split(':')[1].split(',')]

			gestureName = imageName.split('_')[1]
			gestureNum = int(imageName.split('_')[2])
			gestureSeq = int(imageName.replace('.jpg', '').split('_')[3])

			if (gestureName not in dic):
				dicTemp = {}
				dicTemp[gestureNum] = {gestureSeq: (480-boxes[1], boxes[0])}
				dic[gestureName] = dicTemp
			else:
				if (gestureNum in dic[gestureName]):
					dic[gestureName][gestureNum][gestureSeq] = (480-boxes[1], boxes[0])
				else:
					dic[gestureName][gestureNum] = {gestureSeq: (480-boxes[1], boxes[0])}

			line = fp.readline()

	return dic



def writeDicToFile(dic):
	for eachGesute in dic:
		for eachGesuteSeq in dic[eachGesute]:
			for x,y in dic[eachGesute][eachGesuteSeq].values():
				with open(os.path.join(SEQ_OUT_PATH, 'eval_' + eachGesute + '_' + str(eachGesuteSeq) +'.txt'), 'a') as f:
					f.write("%e" % x + " %e" % y + '\n')



if __name__ == '__main__':
	# saveDetectionCrops()
	dic = readDetectionSequencesToDic()
	writeDicToFile(dic)
