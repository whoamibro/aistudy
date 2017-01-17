#!/usr/bin/python

import cPickle as pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
	with open(filename, u'rb') as f:
		datadict = pickle.load(f)
		X = datadict[u'data']
        	Y = datadict[u'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(u"float")
        Y = np.array(Y)
    	return X,Y

def load_CIFAR10(ROOT):
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, u'data_batch_%d' % b)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
	Xtr = np.concatenate(xs)
	Ytr = np.concatenate(ys)
	del X, Y
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, u'test_batch'))
	return Xtr, Ytr, Xte, Yte

class NearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, Y):
		self.Xtr = X
		self.Ytr = Y
	def predict(self, X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)	
		for i in xrange(num_test):
			print i
			distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]),axis=1))
            # self.Xtr - X[i,:] => I1p - I2p 이고 np.square함수를 통해서 제곱을 취한후 np.sum을 통해서 sigma
            # 함수가 구현되었다. 마지막으로 np.sqrt 제곱근 함수를 사용하여 L2 distance의 결과값이 도출된다.
			min_index = np.argmin(distances)
			Ypred[i] = self.Ytr[min_index]
			print 'prediction : %d' % Ypred[i]
			print 'correct answer : %d' % Yte[i]
		return Ypred

Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')
print 'xtr'
print Xtr

print 'ytr'
print Ytr

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)

Yte_predict = nn.predict(Xte_rows)
print 'accuracy of L2 distance : %f' % (np.mean(Yte_predict == Yte))



