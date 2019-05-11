from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import EarlyStopping
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, GRU, Dense, TimeDistributed, SpatialDropout1D
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import layers
import numpy as np
import tensorflow as tf


def data():
    cwd = os.getcwd()
    val_data = np.load(cwd + "/dataset/val.npz")
    valX = val_data['a']
    valY = val_data['b']

    pTrainX, pTestX, pTrainY, pTestY = train_test_split(valX, valY, test_size=0.2, random_state=11)

    encoder = LabelEncoder()
    encoder.classes_ = np.load(cwd + "/dataset/labelEncoder.npz")['arr_0']

    return np.array(pTrainX), np.array(pTrainY), np.array(pTestX), np.array(pTestY), encoder


def create_model(pTrainX, pTrainY, pTestX, pTestY, encoder):

    batch_size = {{choice([64, 128])}}

    inputs = Input(shape=(50,), name='input', dtype=tf.string)
    elmo = layers.ElmoLayer(50, batch_size)(inputs)
    dropout1 = SpatialDropout1D({{choice([0.0, 0.2, 0.5])}})(elmo)
    blstm1 = Bidirectional(GRU(50, return_sequences=True, name='lstm1'))(dropout1)
    dropout2 = SpatialDropout1D({{choice([0.0, 0.2, 0.5])}})(blstm1)
    blstm2 = Bidirectional(GRU(50, return_sequences=True, name='lstm2'))(dropout2)

    if {{choice(['two', 'three'])}} == "three":
        blstm3 = Bidirectional(GRU(50, return_sequences=True, name='lstm2'))(blstm2)
        output = TimeDistributed(Dense(encoder.classes_, activation='softmax'))(blstm3)
    else:
        output = TimeDistributed(Dense(encoder.classes_, activation='softmax'))(blstm2)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    result = model.fit(pTrainX, pTrainY,
                       epochs=20,
                       batch_size=batch_size,
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


if __name__ == '__main__':
    main()
