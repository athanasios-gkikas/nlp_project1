from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

import data_generators
import data_loaders
import metrics

def train_mlp1(pTrain, pVal, pLabelEnc) :

    model = Sequential()
    model.add(Dense(256, input_dim=pTrain[0].shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(19))
    model.add(Activation('softmax'))

    model.compile(
        optimizer=Adam(lr=0.001),
        loss='categorical_crossentropy')

    model.summary()

    model.fit(pTrain[0], pTrain[1],
        callbacks=[metrics.Metrics(19, (pVal[0], pVal[1]), pLabelEnc)],
        epochs=5, batch_size=32, verbose=1)

    return model