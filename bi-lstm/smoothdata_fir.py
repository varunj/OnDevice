import pickle as pk
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.signal as sig
import pdb
import pandas as pd
#clss = pk.load(open('class.pkl','rb'))
#seq = pk.load(open('seq.pkl','rb'))

def visualize(pts, new_pts):
	#pdb.set_trace()
	for i in range(0,len(pts)):
		plt.plot(pts[i][0], -(pts[i][1]), 'bo', linestyle='-')
		plt.plot(new_pts[:,0], -(new_pts[:,1]), 'ro', linestyle='-')
		plt.axis([0,640,-480,0])
	plt.show()

def firFilter(pts):
	# refer https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html

	#filterLen = 2 ** np.arange(2, 14)
	#cutOffFreq = [0.05, 0.95]
	#b = sig.firwin(filterLen, cutOffFreq, width=0.05, pass_zero=False)
	#new_pts = sig.lfilter(b, [1.0], pts, axis=0)
	new_pts = sig.savgol_filter(pts, 15, 1, axis=0)
	new_pts = np.rint(new_pts)
	#visualize(pts, new_pts)
	return new_pts
if __name__ == '__main__':
	#dataSeq=[]
	for fileName in glob.glob("./test_seq/*.txt"):
		file = pd.read_csv(fileName, delim_whitespace=True, header=None)
		fname = fileName.split('_')[-2]
		arr = np.array(file.ix[:, :])
		#dataSeq.append(arr)
		# pdb.set_trace()
		#pt = np.array(arr)
		#pt = np.reshape(pt, (pt.shape[0], pt.shape[-1]))
		# visualize(pt)
		newArr = firFilter(arr)
		visualize(arr,newArr)
