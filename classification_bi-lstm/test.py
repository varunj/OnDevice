'''
    testing if trained w/o names
'''
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import pandas as pd
from tqdm import tqdm
import tensorflow.contrib.slim as slim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import seaborn as sn
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
CLASSES_ALL = ('up','down','left','right','star','del','square','carret','tick','circlecc', 'unclassified')
NOS_CLASSES_ALL = len(CLASSES_ALL)
loss_graph=[]
Validation_loss=0
trueArr=[]
PATH = 'trained_models/2018xxxx-0000/bilstm_5000.ckpt'
SEQ_PATH = '../data/test/*.txt'
# SEQ_PATH = '/home/varunj/Github/OnDevice/testing_eval/testvideo_seq/*.txt'


def shuffle_data(labels, seq):
    temp = 0
    labels2=[]
    for k in seq:
        labels2.append(labels[k]) 
    return labels2

dataSeq = []
targetSeq = []
targetSeq_all = []
length=[]
predArr=[]
for fileName in glob.glob(SEQ_PATH):
    file = pd.read_csv(fileName, delim_whitespace=True, header=None)
    fname = fileName.split('_')[-2]
    arr = np.array(file.ix[:, :])
    #arr1 = np.transpose(arr)
    targetarr = np.zeros(NOS_CLASSES)
    for i in range(0, NOS_CLASSES):
        if (CLASSES[i]==fname):
            targetarr[i] = 1
    targetarr_all = np.zeros(NOS_CLASSES_ALL)
    for i in range(0, NOS_CLASSES_ALL):
        if (CLASSES_ALL[i]==fname):
            targetarr_all[i] = 1
    #l=arr.shape[0]
    #length.append(l)
    #arrstack = arr.reshape(1,l,2)
    dataSeq.append(arr)
    #targetarr=targetarr.reshape(1,10)
    targetSeq.append(targetarr)
    targetSeq_all.append(targetarr_all)
    #print(fname, fileName)

#print(len(dataSeq))
#data=dataSeq

seq = np.arange(len(dataSeq))
np.random.shuffle(seq)
data = shuffle_data(dataSeq, seq)
targets = shuffle_data(targetSeq, seq)
targets_all = shuffle_data(targetSeq_all, seq)


#data = np.array(dataSeq) 
#target = np.array(targetSeq)


data_count = len(data)
train_percent = 1.0
train_data_count = int(train_percent * data_count)
#test_data_count = data_count - train_data_count

train_data = data[:train_data_count]
#test_data = data[train_data_count:]
train_targets = targets[:train_data_count]
train_targets_all = targets_all[:train_data_count]
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

data1 = tf.placeholder(tf.float32, [None, None,2])
target1= tf.placeholder(tf.float32, [None, NOS_CLASSES])
_index = tf.placeholder(tf.int32, [None, ])

bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
_,((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, data1,sequence_length=_index, dtype=tf.float32)
#val, state = tf.nn.dynamic_rnn(fw_cell, data1,sequence_length=_index, dtype=tf.float32)multiply
#output1 = tf.concat([output_fw, output_bw], axis=-1)
output1 = tf.multiply(output_fw, output_bw, name='multiply')
#print(state)
#something can be added
output = tf.layers.dense(inputs=output1, units=NOS_CLASSES)
# output1 = tf.nn.softmax(output, name='softmax')

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
        curr_target_all = train_targets_all[j:(i+1)*batch_size]
        result = sess.run(output, 
            {
                data1: data,
                _index: length
            }
        )
        print("prediction is {} while actual class is {} ".format(CLASSES[np.argmax(result)],CLASSES[np.argmax(curr_target)]))

        if (np.max(result) > 0.75):
            predicted = np.argmax(result)
        else:
            predicted = 10
        
        actual = np.argmax(curr_target_all)
        print(predicted, actual)
        predArr.append(predicted)
        trueArr.append(actual)
        if(predicted==actual):
            acc=acc+1
        print ("accuracy is:{}".format(acc))
        #total_loss_train+=loss_train
        #print("Training loss after {0} epoch is {1}".format(epoch+1, total_loss_train))
    end_time = time.time()

print (CLASSES)
cm = confusion_matrix(trueArr, predArr)
print(accuracy_score(trueArr, predArr))
print(precision_score(trueArr, predArr, average='weighted'))
print(recall_score(trueArr, predArr, average='weighted'))
print(f1_score(trueArr, predArr, average='weighted'))

print(cm)

CLASSES_ALL = ('Up','Down','Left','Right','Star','X','Rectangle','Caret','CheckMark','Circle', 'Unclassified')

df_cm = pd.DataFrame(cm, index = [i for i in CLASSES_ALL],
                  columns = [i for i in CLASSES_ALL])
ax = sn.heatmap(df_cm, annot=True, cmap='YlGnBu', linewidths=1, linecolor='k', square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
plt.title('Confusion Matrix')
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('True', fontweight='bold')
plt.tight_layout()
plt.savefig('/home/varunj/Desktop/fig_confusionmatrix.pdf',bbox_inches='tight')
plt.show()
