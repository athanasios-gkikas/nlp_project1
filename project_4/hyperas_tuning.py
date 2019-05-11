from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import EarlyStopping
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Model
from keras.layers import Input, Dropout, Bidirectional, GRU, Dense, TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import layers
import numpy as np
import tensorflow as tf
import data_generator
import math

def data():
    cwd = os.getcwd()
    val_data = np.load(cwd + "/dataset/val.npz")
    valX = val_data['a']
    valY = val_data['b']

    pTrainX, pTestX, pTrainY, pTestY = train_test_split(valX, valY, test_size=0.2)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(cwd + "/dataset/labelEncoder.npz")['arr_0']
    return pTrainX, pTrainY, pTestX, pTestY, encoder


def create_model(pTrainX, pTrainY, pTestX, pTestY, encoder):

    inputs = Input(shape=(50,), name='input', dtype=tf.string)
    elmo = layers.ElmoLayer(50, 128)(inputs)
    dropped_embeddings = Dropout({{choice([0.0, 0.2, 0.5])}})(elmo)
    var_dropout1 = {{choice([0.0, 0.2, 0.5])}}
    blstm1 = Bidirectional(GRU(50, dropout=var_dropout1, recurrent_dropout=var_dropout1, return_sequences=True, name='lstm1'))(dropped_embeddings)
    dropout2 = Dropout({{choice([0.0, 0.2, 0.5])}})(blstm1)
    blstm2 = Bidirectional(GRU(50, return_sequences=True, name='lstm2'))(dropout2)

    if {{choice(['two', 'three'])}} == "three":
        var_dropout2 = {{choice([0.0, 0.2, 0.5])}}
        blstm3 = Bidirectional(GRU(50, dropout=var_dropout2, recurrent_dropout=var_dropout2, return_sequences=True, name='lstm2'))(blstm2)
        output = TimeDistributed(Dense(len(encoder.classes_), activation='softmax'))(blstm3)
    else:
        output = TimeDistributed(Dense(len(encoder.classes_), activation='softmax'))(blstm2)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    train_gen = data_generator.data_stream([pTrainX, pTrainY], 128, len(encoder.classes_))
    dev_gen = data_generator.data_stream([pTestX, pTestY], 128, len(encoder.classes_))

    stopper = EarlyStopping(monitor='val_loss',
                            min_delta=0, patience=3,
                            verbose=0, mode='auto',
                            restore_best_weights=True)

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=math.ceil(len(pTrainX) / 128),
        validation_data=dev_gen,
        validation_steps=math.ceil(len(pTestX) / 128),
        callbacks=[stopper, ],
        epochs=10,
        verbose=1,
        max_queue_size=100,
        workers=1,
        use_multiprocessing=False, )

    score, acc = model.evaluate_generator(
        generator=dev_gen,
        steps=math.ceil(len(pTestX) / 128),
        verbose=0)

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
