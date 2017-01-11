#!/usr/bin/python

import cPickle as pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
	with open(filename, u'rb') as f:
        # filename의 binary file을 ‘read’ option으로 엽니다.
		datadict = pickle.load(f)
        # 위에서 연 파일에 들어있는 content를 load하여 새로운 변수 datadict에 할당합니다.
		X = datadict[u'data']
        # ’data’라는 key에 대한 value 만을 추출해서 X에 저장합니다. ‘u’는 unicode를 의미합니다.
        Y = datadict[u'labels']
        # ‘lables’라는 key에 대한 value 많을 추출해서 Y에 저장합니다.
		X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(u"float")
        # X.reshape(10000, 3, 32, 32) -> 10000 * 3 * 32 * 32 의 4차원 배열로 reshape합니다.
        # .transpose(0, 2, 3, 1) -> 10000을 0번 index, 3을 1번 index, 32를 2번 in
        # dex, 32를 3번 index라고 생각한다면 위에서 선언된 transpose는 입력된 파라메터 순서대로
        # 기존의 순서를 바꿔버린다라고 생각하시면 될 것 같습니다. 따라서 10000 * 32 * 32 * 3의 4차원 배열로 바뀌게 됩니다.
		Y = np.array(Y)
        # 기존의 Y에 저장되어 있던 것을 numpy array의 형태로 바꿉니다.
		return X, Y

def load_CIFAR10(ROOT):
	xs = []
	ys = []
	for b in range(1,6):
		f = os.path.join(ROOT, u'data_batch_%d' % b)
        # ROOT의 경로(디렉토리의 주소)를 의미하는 값과 ‘data_batch_변수b에 들어있는 값’ 을 갖는
        # unicode를 연결하여 ‘ROOT/data_batch_변수b에 들어있는 값’ 의 문자열을 만들어냅니다.
		X, Y = load_CIFAR_batch(f)
        # 위의 load_CIFAR_batch()의 함수에서 반환되는 X, Y의 값을 load_CIFAR10()함수에서 선언된 X,Y에 할당합니다.
		xs.append(X)
        # load_CIFAR_batch로부터 반환된 값이 들어가있는 X를 xs에 할당합니다.
		ys.append(Y)
        # load_CIFAR_batch로부터 반환된 값이 들어가있는 Y를 ys에 할당합니다.
	Xtr = np.concatenate(xs)
    # xs에 ‘[array([[[[‘ 같은 형식으로 저장이 되어있었다면, Xtr에는 ‘[array(‘를 벗겨낸 배열의 형태만
    # 저장이 된다고 생각하시면 될 것같습니다. 껍데기를 벗겨내고 속알맹이만 뺴낸다고 생각하면 되지 않을까 합니다.
	Ytr = np.concatenate(ys)
    # 위의 Xtr과 마찬가지로 ‘[array([‘의 형식이 ‘[array(‘가 벗겨진 배열의 형태만 저장이 됩니다.
	del X, Y # X, Y는 이제 쓸모가 없기 때문에 지웁니다.
	Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, u'test_batch'))
    # training 데이터처럼 test 데이터들도 load_CIFAR_batch()함수를 사용해서 Xte, Yte변수에 할당합니다.
	return Xtr, Ytr, Xte, Yte

class NearestNeighbor(object):
	def __init__(self):
		pass # null statement라고 생각하면 될 것 같습니다.

	def train(self, X, Y):
		self.Xtr = X
		self.Ytr = Y
        # 파라메터로 넘겨받은 데이터들(X)과 라벨들(Y)의 값을 각각 Xtr, Ytr에 저장합니다.
	def predict(self, X):
		num_test = X.shape[0]
        # num_test의 값에 10000의 값이 들어갑니다. (test 이미지들은 10000개)
		Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)	
		for i in xrange(num_test): # 루프가 10000번 돕니다.
			print i
			distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            # 현재이미지와 training에 사용된 이미지들의 거리를 계산하는 것으로 이 계산을 통해서 L1 distance를 구합니다.
			min_index = np.argmin(distances)
            # L1 distance의 값들이 저장된 distances 배열에서 가장 작은 값의 index를 min_index라는 변수에 할당합니다.
			Ypred[i] = self.Ytr[min_index]
            # 입력된 사진에 대해서 예측한 label의 값을 Ypred[]배열에 저장합니다.
		return Ypred

Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
# 4차원 배열을 [[32*32*3] * 50000개]의 형태로 변환합니다.
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)
# 4차원 배열을 [[32*32*3] *10000개]의 형태로 변환합니다.

nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)

Yte_predict = nn.predict(Xte_rows)
print 'accuracy : %f' % (np.mean(Yte_predict == Yte))
# Yte_predict와 Yte의 동일 index에 대해서 들어있는 값이 일치하는 경우의 개수를 구하여 10000으로 나눈 결과값을 정확도로 보여줍니다.

