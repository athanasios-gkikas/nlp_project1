from __future__ import print_function
from sklearn.preprocessing import LabelEncoder

import data_loaders
import os
import numpy as np
import metrics

from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
	"""
	Data providing function:

	This function is separated from create_model() so that hyperopt
	won't reload data for each evaluation run.
	"""
	cwd = os.getcwd()
	valX, valY = data_loaders.import_embeddings("val")

	encoder = LabelEncoder()
	encoder.classes_ = np.load(cwd + "/dataset/labelEncoder.npz")['arr_0']

	return valX, valY, encoder

def create_model(valX, valY, encoder) :

	pTrain = (valX, valY)

	y = np.argmax(pTrain[1], axis=1)

	for i in range(0, len(encoder.classes_)) :
		mask = np.count_nonzero(y == i)
		print(encoder.inverse_transform([i]), " ", mask, " ",
			"{0:.2}".format(mask / pTrain[1].shape[0]))

	activation = {{choice(['relu', 'elu'])}}
	model = Sequential()
	model.add(Dense(512, input_dim=pTrain[0].shape[1]))
	model.add(Activation(activation))
	model.add(Dropout({{choice([0.2, 0.5])}}))
	model.add(Dense(256))
	model.add(Activation(activation))
	model.add(Dropout({{choice([0.2, 0.5])}}))

	# If we choose 'four', add an additional fourth layer
	if {{choice(['three', 'four'])}} == 'four':
		model.add(Dense(128))
		model.add(Activation(activation))

	model.add(Dense(len(encoder.classes_)))
	model.add(Activation('softmax'))

	model.compile(
		optimizer={{choice(['adam', 'sgd'])}},
		loss='categorical_crossentropy',
		metrics=['categorical_accuracy'])

	result = model.fit(pTrain[0], pTrain[1],
		epochs={{choice([3, 5, 7])}}, 
		batch_size={{choice([64, 128])}},
		verbose=1)

	#get the highest validation accuracy of the training epochs
	validation_acc = np.amax(result.history['categorical_accuracy']) 
	print('Best val acc of epoch:', validation_acc)
	return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def main():
	best_run, best_model = optim.minimize(model=create_model, 
											data=data, 
											algo=tpe.suggest, 
											max_evals=5, 
											trials=Trials())
	x_val, y_val, encoder = data()
	# print("Evalutation of best performing model:")
	# print(best_model.evaluate(X_test, Y_test))
	print("Best performing model chosen hyper-parameters:")
	print(best_run)

	print(best_model.summary())


if __name__ == '__main__' :
	main()