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

    batchSize = 111
    seq = 25
    inputs = Input(shape=(seq,), name='input', dtype=tf.string)
    elmo = layers.ElmoLayer(seq, batchSize)(inputs)
    gru1 = Bidirectional(GRU({{choice([25, 50, 100])}}, dropout={{choice([0.0, 0.2, 0.5])}}, recurrent_dropout={{choice([0.0, 0.2, 0.5])}}, return_sequences=True, name='gru1'))(elmo)

    if {{choice(['two', 'three'])}} == "three":
        gru2 = Bidirectional(GRU({{choice([25, 50, 100])}}, dropout={{choice([0.0, 0.2, 0.5])}}, recurrent_dropout={{choice([0.0, 0.2, 0.5])}}, return_sequences=True, name='lstm2'))(gru1)
        output = TimeDistributed(Dense(len(encoder.classes_), activation='softmax'))(gru2)
    else:
        output = TimeDistributed(Dense(len(encoder.classes_), activation='softmax'))(gru1)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    train_gen = data_generator.data_stream([pTrainX, pTrainY], batchSize, len(encoder.classes_))
    dev_gen = data_generator.data_stream([pTestX, pTestY], batchSize, len(encoder.classes_))

    stopper = EarlyStopping(monitor='val_loss',
                            min_delta=0, patience=2,
                            verbose=0, mode='auto',
                            restore_best_weights=True)

    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=math.ceil(len(pTrainX) / batchSize),
        validation_data=dev_gen,
        validation_steps=math.ceil(len(pTestX) / batchSize),
        callbacks=[stopper, ],
        epochs=8,
        verbose=1,
        max_queue_size=100,
        workers=1,
        use_multiprocessing=False, )

    score, acc = model.evaluate_generator(
        generator=dev_gen,
        steps=math.ceil(len(pTestX) / batchSize),
        verbose=0)

    print('Best val acc of epoch:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def main():
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    print(best_model.summary())


if __name__ == '__main__':
    main()
