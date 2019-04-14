from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

import data_generators
import data_loaders
import metrics
import numpy as np

def train_mlp1(pTrain, pVal, pLabelEnc) :

    y = np.argmax(pTrain[1], axis=1)

    for i in range(0, len(pLabelEnc.classes_)) :
        mask = np.count_nonzero(y == i)
        print(pLabelEnc.inverse_transform([i]), " ", mask, " ",
            "{0:.2}".format(mask / pTrain[1].shape[0]))

    model = Sequential()
    model.add(Dense(64, input_dim=pTrain[0].shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(pLabelEnc.classes_)))
    model.add(Activation('softmax'))

    model.compile(
        optimizer=Adam(lr=0.001),
        loss='categorical_crossentropy')

    model.summary()

    model.fit(pTrain[0], pTrain[1],
        callbacks=[metrics.Metrics(len(pLabelEnc.classes_), (pVal[0], pVal[1]), pLabelEnc)],
        epochs=3, batch_size=128, verbose=1)

    return model

def evaluate_model(pModel, pTest, pEncoder) :

    metric = metrics.Metrics(
        len(pEncoder.classes_), (pTest[0], pTest[1]), pEncoder)

    metric.getMetrics(pModel)

    return