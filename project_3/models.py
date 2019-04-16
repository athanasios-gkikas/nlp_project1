from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

import data_generators
import data_loaders
import metrics
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger
import numpy as np

def train_mlp1(pTrain, pVal, pLabelEnc) :

    y = np.argmax(pTrain[1], axis=1)

    for i in range(0, len(pLabelEnc.classes_)) :
        mask = np.count_nonzero(y == i)
        print(pLabelEnc.inverse_transform([i]), " ", mask, " ",
            "{0:.2}".format(mask / pTrain[1].shape[0]))

    model = Sequential()
    model.add(Dense(512, input_dim=pTrain[0].shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(len(pLabelEnc.classes_)))
    model.add(Activation('softmax'))

    model.compile(
        optimizer=Adam(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    model.summary()

    csv_logger = CSVLogger('training.log')

    epochs = 10

    history = model.fit(pTrain[0], pTrain[1],
        callbacks=[metrics.Metrics(len(pLabelEnc.classes_), (pVal[0], pVal[1]), pLabelEnc), csv_logger], validation_data=(pVal[0], pVal[1]),
        epochs=epochs, batch_size=128, verbose=1)

    model_eval_init = model.evaluate(
        pVal[0],
        pVal[1],
        verbose=1
    )

    print("Train Loss     : {0:.5f}".format(history.history['loss'][-1]))
    print("Validation Loss: {0:.5f}".format(history.history['val_loss'][-1]))
    print("Test Loss      : {0:.5f}".format(model_eval_init[0]))
    print("---")
    print("Train Accuracy     : {0:.5f}".format(history.history['categorical_accuracy'][-1]))
    print("Validation Accuracy: {0:.5f}".format(history.history['val_categorical_accuracy'][-1]))
    print("Test Accuracy      : {0:.5f}".format(model_eval_init[1]))

    # Plot train and validation error per epoch.
    plot_history(hs={'Init model': history}, epochs=epochs, metric='loss')
    plot_history(hs={'Init model': history}, epochs=epochs, metric='categorical_accuracy')

    return model

def evaluate_model(pModel, pTest, pEncoder) :

    metric = metrics.Metrics(
        len(pEncoder.classes_), (pTest[0], pTest[1]), pEncoder)

    metric.getMetrics(pModel)

    return


def plot_history(hs, epochs, metric):
    plt.clf()
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['font.size'] = 16
    for label in hs:
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric))
        plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric))
    x_ticks = np.arange(0, epochs + 1)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.show()

