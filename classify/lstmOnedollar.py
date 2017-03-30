
from math import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import NullFormatter

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import np_utils
from keras.optimizers import RMSprop

yone = np.load('./onedollardata/label.npy')
trajonet = np.load('./onedollardata/trajdata.npy')
ximgt = np.load('./onedollardata/imgdata.npy')

print yone.shape
print trajonet[0].shape
print ximgt.shape



# make sure the length is same

maxlen = 0
xtraintemp = []
for i in range(len(trajonet)):
        if maxlen<trajonet[i].shape[0]:
                    maxlen = trajonet[i].shape[0]
print maxlen
for i in range(len(trajonet)):
        if trajonet[i].shape[0]<209:
                    tt = np.concatenate((trajonet[i],np.zeros((209-trajonet[i].shape[0],2))),axis = 0)
        xtraintemp.append(tt)
Xone = np.array(xtraintemp).reshape((-1,maxlen,2))
print Xone.shape

np.random.seed(100)
maskTrain = np.random.choice(5280,2640,replace=False)
afterTrain = np.setdiff1d(range(5280),maskTrain)
maskValidation = np.random.choice(afterTrain,1320,replace=False)
maskTest = np.setdiff1d(afterTrain,maskValidation)
print maskTrain

numClass = 16
Xtrain = Xone[maskTrain,:,:]
Xvalid = Xone[maskValidation,:,:]
Xtest = Xone[maskTest,:,:]
Ytrain = np_utils.to_categorical(yone[maskTrain],numClass)
Yvalid = np_utils.to_categorical(yone[maskValidation],numClass)
Ytest = np_utils.to_categorical(yone[maskTest],numClass)
print Xtrain.shape, Xvalid.shape, Xtest.shape
print Ytrain.shape, Yvalid.shape, Ytest.shape


model = Sequential()
model.add(LSTM(132, input_shape=(maxlen, 2)))
model.add(Dense(numClass,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = RMSprop(),metrics=['accuracy'])
print "training.."
model.fit(Xtrain,Ytrain,batch_size = 20,nb_epoch = 20, validation_data=[Xvalid,Yvalid])
