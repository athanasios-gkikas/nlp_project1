from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from keras import backend as k

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class Metrics(Callback) :

    def __init__(self, pNumClasses, pValSet, pLabelEnc, **kwargs):

        super(Metrics, self).__init__(**kwargs)

        self.valX = pValSet[0]
        self.valY = pValSet[1]
        self.numClasses = pNumClasses

        self.labelEnc = pLabelEnc

        self.per_class_history = np.zeros([100, pNumClasses, 4])
        self.macro_history = np.zeros([100, 4])

    def getMetrics(self, pModel) :

        prediction = pModel.predict(self.valX)
        gt = np.argmax(self.valY, axis=1)
        prediction = np.argmax(prediction, axis=1)

        per_class_metrics = np.zeros([self.numClasses, 4])
        macro_metrics = np.zeros([1, 4])

        for i in range(0, self.numClasses) :
            trueY = gt == i
            predY = prediction == i

            per_class_metrics[i, 0] = precision_score(trueY, predY)
            per_class_metrics[i, 1] = recall_score(trueY, predY)
            per_class_metrics[i, 2] = f1_score(trueY, predY)
            per_class_metrics[i, 3] = accuracy_score(trueY, predY)

        macro_metrics[0, 0] = np.mean(per_class_metrics[:, 0])
        macro_metrics[0, 1] = np.mean(per_class_metrics[:, 1])
        macro_metrics[0, 2] = np.mean(per_class_metrics[:, 2])
        macro_metrics[0, 3] = np.mean(per_class_metrics[:, 3])

        return per_class_metrics, macro_metrics

    def on_epoch_begin(self, epoch, logs=None) :
        pass

    def on_epoch_end(self, epoch, logs=None) :

        per_class_metrics, macro_metrics = self.getMetrics(self.model)

        self.per_class_history[epoch, :,:] = per_class_metrics
        self.macro_history[epoch, :] = macro_metrics

        for i in range(0, self.numClasses) :

            label = self.labelEnc.inverse_transform([i])

            print(i, "class ", label,
                " precision: {0:.4f}".format(self.per_class_history[epoch,i,0]),
                " recall: {0:.4f}".format(self.per_class_history[epoch,i,1]),
                " f1: {0:.4f}".format(self.per_class_history[epoch,i,2]),
                " accuracy: {0:.4f}".format(self.per_class_history[epoch,i,3]))

        print("With <UNK> macro f1: {0:.4f}".format(self.macro_history[epoch, 2]),
         "macro accuracy: {0:.4f}".format(self.macro_history[epoch, 3]))

        index = self.labelEnc.transform(['UNK',])[0]

        arr1 = self.per_class_history[epoch,0:index,2]
        arr2 = self.per_class_history[epoch,index + 1:,2]

        print("Without <UNK> macro f1: {0:.4f}".format(np.mean(np.concatenate((arr1, arr2)))))

        arr1 = self.per_class_history[epoch,0:index,3]
        arr2 = self.per_class_history[epoch,index + 1:,3]

        print("Without <UNK> macro accuracy: {0:.4f}".format(np.mean(np.concatenate((arr1, arr2)))))

        return

    def on_batch_end(self, batch, logs=None) :
        pass

    def on_train_begin(self, logs=None) :
        pass

    def on_train_end(self, logs=None) :
        cwd = os.getcwd()
        np.savez_compressed(cwd + "/dataset/per_class_history", self.per_class_history)
        np.savez_compressed(cwd + "/dataset/macro_history", self.macro_history)
        return

def evaluate_model(pModel, pTest, pEncoder) :

    cwd = os.getcwd()

    metric = Metrics(
        len(pEncoder.classes_), (pTest[0], pTest[1]), pEncoder)

    metric.getMetrics(pModel)

    model_eval_init = pModel.evaluate(
        pTest[0], pTest[1], verbose=1 )

    with open(cwd + '/dataset/' + pModel.name + '.log', 'r') as log :
        rows = csv.DictReader(log)
        history = {}
        for i, row in enumerate(rows) :
            for key,value in row.items() :
                if key in history :
                    history[key] += [float(value),]
                else :
                    history[key] = [float(value),]

    epochs = len(history['epoch'])

    print("Train Loss     : {0:.5f}".format(history['loss'][-1]))
    print("Validation Loss: {0:.5f}".format(history['val_loss'][-1]))
    print("Test Loss      : {0:.5f}".format(model_eval_init[0]))
    print("---")
    print("Train Accuracy     : {0:.5f}".format(history['categorical_accuracy'][-1]))
    print("Validation Accuracy: {0:.5f}".format(history['val_categorical_accuracy'][-1]))
    print("Test Accuracy      : {0:.5f}".format(model_eval_init[1]))
    plot_history(hs={'Init model': history}, epochs=epochs, metric='categorical_accuracy')

    # Plot train and validation error per epoch.
    plot_history(hs={'Init model': history}, epochs=epochs, metric='loss')

    return

def plot_history(hs, epochs, metric):
    plt.clf()
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['font.size'] = 16

    for label in hs:
        plt.plot(hs[label][metric], label='{0:s} train {1:s}'.format(label, metric))
        plt.plot(hs[label]['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric))

    x_ticks = np.arange(0, epochs + 1)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.show()
