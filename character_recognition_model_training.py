# !/usr/bin/python
# _*_ coding:utf8 _*_

'''
description: deep learning for classifier
code date: 2018/06/02
athor: TTang
'''

# switch the backend
import os
os.environ['KERAS_BACKEND']='tensorflow'

# import some libs
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def start(k_fold):

	# initialize parameters
	seed=7
	np.random.seed(seed)
	nb_classes=4

	# load dataset
	X=np.load(os.getcwd()+'/datasets/X.npy')
	Y=np.load(os.getcwd()+'/datasets/Y.npy')

	# define 5-fold cross validation test harness
	skf=StratifiedKFold(n_splits=k_fold,shuffle=True,random_state=seed)
	splitted_indices=skf.split(X,Y)
	cvscores=[]

	for index,(train_indices,val_indices) in enumerate(splitted_indices):
		i=index+1
		print('This is the %dth training'%(i))
		X_train=X[train_indices].astype('float64')
		X_val=X[val_indices].astype('float64')
		Y_train=Y[train_indices]
		Y_val=Y[val_indices]

		# reshape dataset
		X_train=X_train.astype('float64').reshape(X_train.shape[0],-1)/255
		X_val=X_val.astype('float64').reshape(X_val.shape[0],-1)/255


		# one hot encoding
		Y_train=np_utils.to_categorical(Y_train,10)
		Y_val=np_utils.to_categorical(Y_val,10)

		model=Sequential([
			Dense(output_dim=32,input_dim=784),
			Activation('relu'),
			Dense(output_dim=10),
			Activation('softmax')
			])

		rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

		model.compile(
			loss='categorical_crossentropy',
			# optimizer='rmsprop',
			optimizer=rmsprop,
			# optimizer='sgd',
			metrics=['accuracy']
			)

		# train modle
		print("Training.....")
		history=model.fit(X_train,Y_train,nb_epoch=10,batch_size=100)
		# model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,callbacks=cbks,validation_data=(x_test, y_test))
		# print("training loss:",history.history['loss'])
		# print(type(history.history['loss']))
		# print(len(history.history['loss']))

		# plot the loss and accuracy of training
		fig=plt.figure(i)
		plt.plot(range(0,len(history.history['loss'])),(history.history['loss']))
		plt.plot(range(0,len(history.history['loss'])),(history.history['acc']))
		plt.xlabel('Epochs')
		plt.ylabel('Loss and Accuracy')
		plt.title('Loss and accuracy of training')
		plt.legend(['Loss','Acc'],loc='center right')
		# plt.axis([0,len((history.history['loss'])),0,max((history.history['loss']))])
		plt.grid(True)
		# plt.show()
		fig.savefig(os.getcwd()+'/'+'performance_images/'+'character_recognition_performance_'+str(i)+'.png')

		# test model
		print("Test......")
		accuracy=model.evaluate(X_val,Y_val,batch_size=100)
		# print("The accuracy is %s: %.2f%%" % (model.metrics_names[1],accuracy[1]*100))
		print("The accuracy is : %.2f%%" % (accuracy[1]*100))
		cvscores.append(accuracy[1]*100)

	print("The mean accuracy and std are : %.2f%% (+/- %.2f%%)" % (np.mean(cvscores),np.std(cvscores)))

	# save trained model
	model.save(os.getcwd()+'/models/model_character_recognition.h5')
	print('Finishing saving model')

	plot_model(
		model,show_shapes=True,show_layer_names=True,
		to_file=os.getcwd()+'/structure_character_recognition_model.png'
		)

if __name__=='__main__':

	# initialize parameters
	k_fold=10
	start(k_fold)