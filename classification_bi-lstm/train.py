from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob
import pandas as pd
from tqdm import tqdm
import tensorflow.contrib.slim as slim
import time


# --------------- new added --------------
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
# ----------------------------------------

LEARNING_RATE = 0.0001
NUM_HIDDEN_UNITS = 30
BATCH_SIZE = 64
EPOCHS = 5000
TRAIN_PERCENT = 0.8
CLASSES = ('up','down','left','right','star','del','square','carret','tick','circlecc')
NOS_CLASSES = len(CLASSES)
PATH = 'trained_models/20181003-1900/'

loss_graph_val = []
loss_graph_train = []
Validation_loss = 0
trueArr = []

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
for fileName in glob.glob("../data/train/*.txt"):
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
 #error
#data=dataSeq

seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(dataSeq, seq)
targets = shuffle_data(targetSeq, seq)

#data = np.array(dataSeq) 
#target = np.array(targetSeq)


data_count = len(data)
train_data_count = int(TRAIN_PERCENT * data_count)

train_data = data[:train_data_count]
test_data = data[train_data_count:]
train_targets = targets[:train_data_count]
test_targets = targets[train_data_count:]

def Train_batchwise_data_toarray(curr_data):
	length=[]
	pdd_data_list=[]
	for k in range(BATCH_SIZE):
                length.append(len(curr_data[k]))
	maxlen=max(length)
	for k in range(BATCH_SIZE):
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


bw_cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN_UNITS,state_is_tuple=True, name='lstm_bck')
fw_cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN_UNITS,state_is_tuple=True, name='lstm_fwd')
_,((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, data1,sequence_length=_index, dtype=tf.float32)
#val, state = tf.nn.dynamic_rnn(fw_cell, data1,sequence_length=_index, dtype=tf.float32)multiply
#output1 = tf.concat([output_fw, output_bw], axis=-1)
output1 = tf.multiply(output_fw, output_bw, name='multiply')
#print(state)
#something can be added
output = tf.layers.dense(inputs=output1, units=NOS_CLASSES, name='dense')
# output1 = tf.nn.softmax(output, name='softmax')
print (output)
#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=target1))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    

    total = 0
    for v in tf.trainable_variables():
        dims = v.get_shape().as_list()
        num  = int(np.prod(dims))
        total += num
        print('  %s \t\t Num: %d \t\t Shape %s ' % (v.name, num, dims))
    print('\nTotal number of params: %d' % total)

    # saver.restore(sess,'/home/gaurav/Documents/DSNT_final/Bilstm_final/Bilstm _Model_30/model_Bilstm1.ckpt')
    for epoch in range(EPOCHS):
        total_loss_train=0
        for i in tqdm(range(int(len(train_data)/BATCH_SIZE))):
            j=i*BATCH_SIZE
            curr_data=train_data[j:(i+1)*BATCH_SIZE]
            data,length=Train_batchwise_data_toarray(curr_data)
            length=np.array(length)
            curr_target = train_targets[j:(i+1)*BATCH_SIZE]

            _, loss_train = sess.run([opt, loss], 
                {
                    data1: data,
                    target1: curr_target,
                    _index: length
                }
            )
            total_loss_train+=loss_train
        print("Training loss after {0} epoch is {1}".format(epoch+1, total_loss_train))
        loss_graph_train.append(total_loss_train)

        total_loss_val=0
        for i in tqdm(range(int(len(test_data)/BATCH_SIZE))):
            j=i*BATCH_SIZE
            curr_data=test_data[j:(i+1)*BATCH_SIZE]
            data,length=Train_batchwise_data_toarray(curr_data)
            length=np.array(length)
            curr_target = test_targets[j:(i+1)*BATCH_SIZE]
            loss_val = sess.run(loss, 
                {
                    data1: data,
                    target1: curr_target,
                    _index: length
                }
            )
            total_loss_val+=loss_val
        print("Validation loss after {0} epoch is {1}".format(epoch+1, total_loss_val))
        loss_graph_val.append(total_loss_val)

        plt.plot(loss_graph_val)
        plt.ylabel('validation loss')
        plt.xlabel('epoch')
        plt.savefig('loss_val.png')
        plt.close()
        plt.plot(loss_graph_train)
        plt.ylabel('training loss')
        plt.xlabel('epoch')
        plt.savefig('loss_training.png')
        plt.close()

        if(epoch==0):
            Validation_loss=total_loss_val
            save_path = saver.save(sess, PATH + "/bilstm_"+str(epoch+1)+".ckpt")
            print("Model saved in path: %s" % save_path)
        else:
            if(Validation_loss>total_loss_val):
                Validation_loss=total_loss_val
                save_path = saver.save(sess, PATH + "/bilstm_"+str(epoch+1)+".ckpt")
                print("Model saved in path: %s" % save_path)

                