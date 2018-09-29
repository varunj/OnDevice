'''
	https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3
	https://stats.stackexchange.com/questions/276914/how-can-i-calculate-the-false-positive-rate-for-an-object-detection-algorithm-w
	https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

	input files
		next image
		(370.9997253417969, 126.99979400634766, 484.0, 311.0002136230469, [])
		(370.72564697265625, 123.91508483886719, 486.5634765625, 313.5149230957031, ['hand: 98%'])
		...
'''

import numpy as np
import matplotlib.pyplot as plt


# INP_FILE = '/home/arteam/on-device/data/trained_models/20180918-1345/testing/out_20180830-1400.txt'
INP_FILE = '/home/arteam/on-device/data/trained_models/20180918-1345/testing/out_20180918-1345.txt'
DEF_CONF, DEF_IOU = 0.85, 0.5



def readFile(filePath):
	'''
		output:
		[	[	[xmin,ymin,xmax,ymax,100],		groundtruth
				[xmin,ymin,xmax,ymax,%],		
				...								for different detections
			]
			...									for different images
		]
	'''
	arrMain = []
	arrAux = []

	with open(filePath) as f:
		for eachLine in f:
			eachLine = eachLine.rstrip('\n').translate(None, '()[]%\'\'').split(', ')
			
			if (eachLine[0] == 'next image'):
				arrMain.append(arrAux)
				arrAux = []

			else:
				# groundtruth lines
				if (eachLine[4] == ''):
					eachLine[4] = '100.0'
					arrAux.append([float(x) for x in eachLine]) 
				# detection lines
				else:
					eachLine[4] = float(eachLine[4].split(': ')[1])/100
					arrAux.append([float(x) for x in eachLine]) 

	return arrMain[1:]



def calcIOU(bb1, bb2):
	'''
		input:
		[xmin,ymin,xmax,ymax,%], [xmin,ymin,xmax,ymax,%]
	'''
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box. compute the area of both AABBs
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou



def calcAccuracy(data, thresh_confidence, thresh_iou):
	'''
		input:
		[	[	[xmin,ymin,xmax,ymax,100],		groundtruth
				[xmin,ymin,xmax,ymax,%],		
				...								for different detections
			]
			...									for different images
		]
		thresh_confidence
		thresh_iou
	'''
	count = 0
	for eachImage in data:
		groundtruth = eachImage[0]

		for eachDetection in eachImage[1:]:
			iou = calcIOU(groundtruth, eachDetection)
			predictionConfidence = eachDetection[4]
			
			if (iou >= thresh_iou and predictionConfidence >= thresh_confidence):
				count = count + 1

	return count*1.0/len(data)



def calcPrecRecall(data, thresh_confidence, thresh_iou):
	'''
		calculate precision, recall at a specified thresh_confidence and thresh_iou
		input:
		[	[	[xmin,ymin,xmax,ymax,100],		groundtruth
				[xmin,ymin,xmax,ymax,%],		
				...								for different detections
			]
			...									for different images
		]
		thresh_confidence
		thresh_iou
	'''
	tp, fp, tn, fn = 0, 0, 0 ,0
	for eachImage in data:
		groundtruth = eachImage[0]

		if (len(eachImage) == 1):
			fn = fn + 1

		for eachDetection in eachImage[1:]:
			iou = calcIOU(groundtruth, eachDetection)
			predictionConfidence = eachDetection[4]
			
			if (iou >= thresh_iou and predictionConfidence >= thresh_confidence):
				tp = tp + 1
			if (iou < thresh_iou and predictionConfidence >= thresh_confidence):
				fp = fp + 1

	return tp*1.0/(tp+fp), tp*1.0/(tp+fn)



def plotPR(data):
	'''
		plot PR curve at varying thresh_confidence and thresh_iou
		input:
		[	[	[xmin,ymin,xmax,ymax,100],		groundtruth
				[xmin,ymin,xmax,ymax,%],		
				...								for different detections
			]
			...									for different images
		]
	'''
	rangeOfIoU = range(50,100, 5)
	rangeOfConf = range(0,100, 1)
	print('rangeOfIoU, rangeOfConf:', rangeOfIoU, rangeOfConf)

	for thresh_iou in rangeOfIoU:
		thresh_iou = thresh_iou/100.0
		print('calculating at thresh_iou: ', thresh_iou)

		arr = []
		for thresh_confidence in rangeOfConf:
			thresh_confidence = thresh_confidence/100.0		
			prec, recall = calcPrecRecall(data, thresh_confidence, thresh_iou)
			arr.append([prec, recall])
		
		plt.scatter([x[1] for x in arr], [x[0] for x in arr], label=thresh_iou)
	
	plt.title('PR CURVE')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend()
	plt.show()


if __name__ == "__main__":
	data = readFile(INP_FILE)

	acc = calcAccuracy(data, DEF_CONF, DEF_IOU)	
	prec, recall = calcPrecRecall(data, DEF_CONF, DEF_IOU)
	print('acc, prec, recall: ', acc, prec, recall)

	plotPR(data)
