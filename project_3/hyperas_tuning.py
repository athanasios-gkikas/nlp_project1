from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LeakyReLU, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.model_selection import train_test_split

import keras.layers
import data_loaders
import os
import numpy as np
import metrics

def data():
	cwd = os.getcwd()
	valX, valY = data_loaders.import_embeddings("val")

	encoder = LabelEncoder()
	encoder.classes_ = np.load(cwd + "/dataset/labelEncoder.npz")['arr_0']

	pTrainX, pTestX, pTrainY, pTestY = train_test_split(
		valX, valY, test_size=0.2, random_state=11)

	return np.array(pTrainX), np.array(pTrainY), np.array(pTestX), np.array(pTestY), encoder

def create_model(pTrainX, pTrainY, pTestX, pTestY, encoder) :

	activation = {{choice(['relu', 'leakyrelu'])}}
	model = Sequential()

	model.add(Dense(512, input_dim=pTrainX.shape[1]))

	if activation == 'relu' :
		model.add(Activation('relu'))
	else :
		model.add(LeakyReLU())

	model.add(Dropout({{choice([0.0, 0.2, 0.5])}}))

	model.add(Dense(256))
	
	if activation == 'relu' :
		model.add(Activation('relu'))
	else :
		model.add(LeakyReLU())

	model.add(Dropout({{choice([0.0, 0.2, 0.5])}}))

	model.add(Dense(128))
	
	if activation == 'relu' :
		model.add(Activation('relu'))
	else :
		model.add(LeakyReLU())

	model.add(Dropout({{choice([0.0, 0.2, 0.5])}}))

	model.add(Dense(128))

	if activation == 'relu' :
		model.add(Activation('relu'))
	else :
		model.add(LeakyReLU())

	model.add(Dropout({{choice([0.0, 0.2, 0.5])}}))

	model.add(Dense(256))
	
	if activation == 'relu' :
		model.add(Activation('relu'))
	else :
		model.add(LeakyReLU())

	model.add(Dropout({{choice([0.0, 0.2, 0.5])}}))

	model.add(Dense(512))
	
	if activation == 'relu' :
		model.add(Activation('relu'))
	else :
		model.add(LeakyReLU())

	model.add(Dropout({{choice([0.0, 0.2, 0.5])}}))

	model.add(Dense(len(encoder.classes_)))
	model.add(Activation('softmax'))

	model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['categorical_accuracy'])

	result = model.fit(pTrainX, pTrainY,
		epochs=20,
		batch_size= 128,
		validation_data=(pTestX, pTestY),
		callbacks=[EarlyStopping(monitor='val_loss',
			min_delta=0, patience=2, verbose=0, mode='auto')],
		verbose=1)

	score, acc = model.evaluate(pTestX, pTestY, verbose=0)
	print('Best val acc of epoch:', acc)
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def main():
	best_run, best_model = optim.minimize(model=create_model, 
											data=data, 
											algo=tpe.suggest, 
											max_evals=50, 
											trials=Trials())

	print("Best performing model chosen hyper-parameters:")
	print(best_run)

	print(best_model.summary())

if __name__ == '__main__' :
	main()