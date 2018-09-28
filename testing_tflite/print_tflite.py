import tensorflow as tf
from pprint import pprint

g = tf.GraphDef()
g.ParseFromString(open('tflite_graph.pb', 'rb').read())

pprint([n for n in g.node if n.name.find('out') != -1])
pprint(set([n.op for n in g.node]))
