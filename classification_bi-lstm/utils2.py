import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from pprint import pprint



MAIN_PATH = 'trained_models/2018xxxx-0000/'
# MAIN_PATH = 'trained_models/20180930-1100/'
PB_OUT_FILE_PATH = MAIN_PATH + 'frozen_model.pb'


def PBtoTENSORBOARD():
	with tf.Session() as sess:
		model_filename = PB_OUT_FILE_PATH
		with gfile.FastGFile(model_filename, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			g_in = tf.import_graph_def(graph_def)

	train_writer = tf.summary.FileWriter(MAIN_PATH)
	train_writer.add_graph(sess.graph)


def validatePB():
	g = tf.GraphDef()
	g.ParseFromString(open(PB_OUT_FILE_PATH).read())

	# print ops used
	pprint(set([n.op for n in g.node]))

	# prints everything
	pprint([n for n in g.node])



if __name__ == "__main__":
	PBtoTENSORBOARD()
	validatePB()