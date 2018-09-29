'''
    testing if trained w/ names
'''
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import pandas as pd
from tqdm import tqdm
import tensorflow.contrib.slim as slim
from sklearn.metrics import confusion_matrix
import time

# --------------- new added --------------
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# ----------------------------------------

learning_rate=0.0001
num_hidden = 30
batch_size=1
CLASSES = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES = len(CLASSES)
loss_graph=[]
Validation_loss=0
trueArr=[]
PATH = 'trained_models/20180903-1000/bilstm_3770.ckpt'


def shuffle_data(labels, seq):
    temp = 0
    labels2=[]
    for k in seq:
        labels2.append(labels[k]) 
    return labels2

dataSeq = []
targetSeq = []
length=[]
predArr=[]
for fileName in glob.glob("test_seq/*.txt"):#/home/gaurav/Documents/dsnt-master/test_seq
    file = pd.read_csv(fileName, delim_whitespace=True, header=None)
    fname = fileName.split('_')[-2]
    arr = np.array(file.ix[:, :])
    #arr1 = np.transpose(arr)
    targetarr = np.zeros(NOS_CLASSES)
    for i in range(0, NOS_CLASSES):
        if (CLASSES[i]==fname):
            targetarr[i] = 1
    #l=arr.shape[0]
    #length.append(l)
    #arrstack = arr.reshape(1,l,2)
    dataSeq.append(arr)
    #targetarr=targetarr.reshape(1,10)
    targetSeq.append(targetarr)
    #print(fname, fileName)

#print(len(dataSeq))
#data=dataSeq

seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(dataSeq, seq)
targets = shuffle_data(targetSeq, seq)

#data = np.array(dataSeq) 
#target = np.array(targetSeq)


data_count = len(data)
train_percent = 1.0
train_data_count = int(train_percent * data_count)
#test_data_count = data_count - train_data_count

train_data = data[:train_data_count]
#test_data = data[train_data_count:]
train_targets = targets[:train_data_count]
#test_targets = targets[train_data_count:]

def Train_batchwise_data_toarray(curr_data):
	length=[]
	pdd_data_list=[]
	for k in range(batch_size):
		length.append(len(curr_data[k]))
	maxlen=max(length)
	for k in range(batch_size):
		if(maxlen==length[k]):
			padd_data=curr_data[k].reshape(1,maxlen,2)
			#arrstack = arr.reshape(1,200,2)
			pdd_data_list.append(padd_data)
		else:
			app=np.zeros((maxlen-length[k],2))
			padd_data=np.append(curr_data[k],app,axis=0).reshape(1,maxlen,2)
			#arrstack = arr.reshape(1,200,2)
			pdd_data_list.append(padd_data)
	data=np.vstack(pdd_data_list)
	return data,length

data1 = tf.placeholder(tf.float32, [None, None,2],name='input_data')
target1= tf.placeholder(tf.float32, [None, NOS_CLASSES],name='input_target')
_index = tf.placeholder(tf.int32, [None, ],name='input_index')


bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True, name='lstm_bck')
fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True, name='lstm_fwd')
_,((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, data1,sequence_length=_index, dtype=tf.float32)
#val, state = tf.nn.dynamic_rnn(fw_cell, data1,sequence_length=_index, dtype=tf.float32)multiply
#output1 = tf.concat([output_fw, output_bw], axis=-1)
output1 = tf.multiply(output_fw, output_bw, name='multiply')
#print(state)
#something can be added
output = tf.layers.dense(inputs=output1, units=NOS_CLASSES, name='dense')
#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=target1))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(PATH + '.meta')
    
    acc=0
    saver.restore(sess, PATH)
    start_time = time.time()
    for i in tqdm(range(int(len(train_data)/batch_size))):
        j=i*batch_size
        curr_data=train_data[j:(i+1)*batch_size]
        data,length=Train_batchwise_data_toarray(curr_data)
        length=np.array(length)
        curr_target = train_targets[j:(i+1)*batch_size]
        result = sess.run(output, 
            {
                data1: data,
                _index: length
            }
        )
        print("prediction is {} while actual class is {} ".format(CLASSES[np.argmax(result)],CLASSES[np.argmax(curr_target)]))     
        predArr.append(np.argmax(result))
        trueArr.append(np.argmax(curr_target))
        if(np.argmax(result)==np.argmax(curr_target)):
            acc=acc+1
        print ("accuracy is:{}".format(acc))
        #total_loss_train+=loss_train
        #print("Training loss after {0} epoch is {1}".format(epoch+1, total_loss_train))
    end_time = time.time()
print (CLASSES)
print(confusion_matrix(trueArr, predArr))   	
