import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from imageio import imread
import base64
import cv2



MODEL_PATH = 'detect.tflite'
DATA_PATH = 'pascal_test.record'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300



def findNosImagesFromTFRecord(dataPath):
	c = 0
	for record in tf.python_io.tf_record_iterator(dataPath):
		c = c + 1
	print('number of samples: ', c)



def readTFRecordandPredict(dataPath, modelPath=MODEL_PATH):
	# Load TFLite model and allocate tensors. Get input and output tensors.
	interpreter = tf.contrib.lite.Interpreter(model_path=modelPath)
	interpreter.allocate_tensors()
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()


	# read image from TFRecord file
	record_iterator = tf.python_io.tf_record_iterator(path = dataPath)
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)
		
		height = int(example.features.feature['image/height'].int64_list.value[0])
		width = int(example.features.feature['image/width'].int64_list.value[0])		
		xmin = int(float(example.features.feature['image/object/bbox/xmin'].float_list.value[0]) * IMAGE_WIDTH)
		xmax = int(float(example.features.feature['image/object/bbox/xmax'].float_list.value[0]) * IMAGE_WIDTH)
		ymin = int(float(example.features.feature['image/object/bbox/ymin'].float_list.value[0]) * IMAGE_HEIGHT)
		ymax = int(float(example.features.feature['image/object/bbox/ymax'].float_list.value[0]) * IMAGE_HEIGHT)
		
		img = (example.features.feature['image/encoded'].bytes_list.value[0])
		img = imread(io.BytesIO(base64.b64decode(base64.b64encode(img).decode())))
		img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)).reshape([1,IMAGE_WIDTH,IMAGE_HEIGHT,3])
		img = img - 127.5
		img = img * 0.007843


		# show image
		fig, ax = plt.subplots(1)
		ax.imshow(img[0])
		rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,linewidth=3,edgecolor='r',facecolor='none')
		ax.add_patch(rect)


		# Test model on input data.
		interpreter.set_tensor(input_details[0]['index'], img.astype(np.float32))
		interpreter.invoke()
		predictions = interpreter.get_tensor(output_details[0]['index'])
		print(predictions)


		plt.show()
		break



if __name__ == '__main__':
	readTFRecordandPredict(DATA_PATH)


'''
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=//home/varunj/Desktop/tflite_graph.pb
Found 1 possible inputs: (name=normalized_input_image_tensor, type=float(1), shape=[1,300,300,3]) 
No variables spotted.
Found 1 possible outputs: (name=TFLite_Detection_PostProcess, op=TFLite_Detection_PostProcess) 
Found 3103591 (3.10M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 491 Identity, 420 Const, 76 FusedBatchNorm, 59 Relu6, 55 Conv2D, 33 DepthwiseConv2dNative, 12 BiasAdd, 12 Reshape, 10 Add, 2 ConcatV2, 1 Placeholder, 1 RealDiv, 1 Sigmoid, 1 Squeeze, 1 TFLite_Detection_PostProcess
To use with tensorflow/tools/benchmark:benchmark_model try these arguments:
bazel run tensorflow/tools/benchmark:benchmark_model -- --graph=//home/varunj/Desktop/tflite_graph.pb --show_flops --input_layer=normalized_input_image_tensor --input_layer_type=float --input_layer_shape=1,300,300,3 --output_layer=TFLite_Detection_PostProcess

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
[[[ 0.25783527  0.5790124   0.652686    0.7596711 ]
  [-0.05865747  0.10282001  1.1076174   0.835273  ]
  [ 0.02419469  0.04099467  0.47520384  0.5577259 ]
  [ 0.25934047  0.5048453   0.65081894  0.72945017]
  [-0.0138727  -0.04683641  0.4508599   0.4189244 ]
  [ 0.23364908  0.6981144   0.6937504   0.97195256]
  [ 0.3761379   0.6420527   0.4306547   0.6751202 ]
  [-0.00785279 -0.01352056  0.6812844   0.15017396]
  [-0.07630548 -0.07311815  0.41461653  0.6099345 ]
  [ 0.23126584  0.30569303  0.71523136  0.5940569 ]]]

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def readTFRecordOld(dataPath):
	reader = tf.TFRecordReader()
	filename_queue = tf.train.string_input_producer([dataPath], num_epochs=10)
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
		features={ 'image/height': tf.FixedLenFeature([], tf.int64),
					'image/width': tf.FixedLenFeature([], tf.int64),
					'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
					'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
					'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
					'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
					'image/encoded': tf.FixedLenFeature([], tf.string)
					})	

	height = tf.cast(features['image/height'], tf.int32)
	width = tf.cast(features['image/width'], tf.int32)
	xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
	xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
	ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
	ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)

	image = tf.decode_raw(features['image/encoded'], tf.uint8)
	image = tf.reshape(image, tf.stack([height, width, 3]))
	image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
	
	images, labels = tf.train.shuffle_batch([image, tf.stack([xmin,xmax,ymin,ymax])], batch_size=2,
													capacity=30, num_threads=1, min_after_dequeue=10)

	return images, labels

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

'''