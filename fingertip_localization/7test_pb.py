import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from pprint import pprint
import cv2

PB_PATH = 'trained_models/20181009-1100/frozen_model.pb'
INP_IMG_PATH = '../data/fingertip_localization/cropped_images/I_Avenue_0_crop.png'
img_rows, img_cols = 99,99

tf.reset_default_graph()


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


def runInference():

	with tf.Session() as persisted_sess:

		def run_inference_on_image():

			output_tensor = persisted_sess.graph.get_tensor_by_name('import/output_node0:0')
			def predict(img):
				predictions = persisted_sess.run(output_tensor, {'import/conv2d_1_input:0': img})
				predictions = np.squeeze(predictions)
				print(predictions)

				cv2.circle(img[0], (predictions[0],predictions[1]), 6, 25500, thickness=-1, lineType=8, shift=0)
				cv2.imshow('img', img[0])
				cv2.waitKey(0)

			inputImg = make_chan_last(INP_IMG_PATH)
			inputImg = np.divide(inputImg, np.float32(255.0))
			predict([inputImg])



		with gfile.FastGFile(PB_PATH, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			persisted_sess.graph.as_default()
			tf.import_graph_def(graph_def)
			run_inference_on_image()



if __name__ == "__main__":
	runInference()
