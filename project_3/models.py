from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from sklearn.utils import class_weight

import data_generators
import data_loaders
import metrics
import csv
import os

import numpy as np

def build_mlp1(pInputDim, pNumClasses) :

    model = Sequential(name='mlp1')
    model.add(Dense(512, input_dim=pInputDim))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(pNumClasses))
    model.add(Activation('softmax'))

    model.compile(
        optimizer=Adam(lr=0.001),
        loss='categorical_crossentropy')

    return model

def train_model(pModel, pTrain, pVal, pLabelEnc) :

    y = np.argmax(pTrain[1], axis=1)

    freq = np.zeros([1, len(pLabelEnc.classes_)])

    for i in range(0, len(pLabelEnc.classes_)) :
        mask = np.count_nonzero(y == i)
        freq[0,i] = mask / pTrain[1].shape[0]

    weight = np.median(freq) / freq

    for i in range(0, len(pLabelEnc.classes_)) :
        mask = np.count_nonzero(y == i)
        percentage = mask / pTrain[0].shape[0]

        print(i, " ", pLabelEnc.inverse_transform([i]), " ",
            "{0:.2}%".format(percentage * 100.0),
            " weight: {0:.2}".format(weight[0,i]))

    pModel.summary()

    csv_logger = CSVLogger('dataset/' + pModel.name + '.log')

    epochs = 5

    history = pModel.fit(pTrain[0], pTrain[1],
        callbacks=[
            metrics.Metrics(len(pLabelEnc.classes_), (pVal[0], pVal[1]), pLabelEnc),
            csv_logger,],
        validation_data=(pVal[0], pVal[1]),
        epochs=epochs, batch_size=128, verbose=1,)

    pModel.save('dataset/' + pModel.name)

    return pModel