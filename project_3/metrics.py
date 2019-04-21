from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, average_precision_score
from keras import backend as K
from gensim.models import KeyedVectors

import sklearn.metrics as sk_metrics
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import data_loaders
import data_generators

class Metrics(Callback) :

    def __init__(self, pNumClasses, pValSet, pLabelEnc, **kwargs):

        super(Metrics, self).__init__(**kwargs)

        self.valX = pValSet[0]
        self.valY = pValSet[1]
        self.numClasses = pNumClasses

        self.labelEnc = pLabelEnc

        self.per_class_history = np.zeros([100, pNumClasses, 3])
        self.macro_history = np.zeros([100, 4])

    def getMetrics(self, pModel) :

        prediction = pModel.predict(self.valX)
        gt = np.argmax(self.valY, axis=1)
        pred = np.argmax(prediction, axis=1)

        per_class_metrics = np.zeros([self.numClasses, 3])
        macro_metrics = np.zeros([1, 4])

        for i in range(0, self.numClasses) :
            trueY = gt == i
            predY = pred == i

            per_class_metrics[i, 0] = precision_score(trueY, predY)
            per_class_metrics[i, 1] = recall_score(trueY, predY)
            per_class_metrics[i, 2] = f1_score(trueY, predY)

        macro_metrics[0, 0] = np.mean(per_class_metrics[:, 0])
        macro_metrics[0, 1] = np.mean(per_class_metrics[:, 1])
        macro_metrics[0, 2] = np.mean(per_class_metrics[:, 2])

        index = self.labelEnc.transform(['UNK',])[0]

        arr1 = per_class_metrics[0:index,2]
        arr2 = per_class_metrics[index + 1:,2]

        macro_metrics[0, 3] = np.mean(np.concatenate((arr1, arr2)))

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
                " f1: {0:.4f}".format(self.per_class_history[epoch,i,2]),)

        print("With <UNK> macro f1: {0:.4f}".format(self.macro_history[epoch, 2]))
        print("Without <UNK> macro f1: {0:.4f}".format(self.macro_history[epoch, 3]))

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
        len(pEncoder.classes_),
        (pTest[0], pTest[1]), pEncoder)

    metric.getMetrics(pModel)

    prediction = pModel.predict(pTest[0])
    plot_score(pTest[1], prediction, len(pEncoder.classes_))

    loss, acc = pModel.evaluate(pTest[0], pTest[1], verbose=1)

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

    per_class_history = np.load(cwd + "/dataset/per_class_history.npz")['arr_0']
    macro_history = np.load(cwd + "/dataset/macro_history.npz")['arr_0']

    per_class_history = per_class_history[:epochs - 1, :, :]

    plt.figure()

    print(len(pEncoder.classes_))
    plt.subplot(2,2,1)

    for i in range(0, 5) :
        label = pEncoder.inverse_transform([i])[0]
        plt.plot(per_class_history[:epochs - 1, i, 2], label=label)

    plt.xlabel('epoch')
    plt.ylabel('F1 macro')
    plt.grid()
    plt.legend()
    plt.subplot(2,2,2)

    for i in range(5, 10) :
        label = pEncoder.inverse_transform([i])[0]
        plt.plot(per_class_history[:epochs - 1, i, 2], label=label)

    plt.grid()
    plt.legend()
    plt.subplot(2,2,3)

    for i in range(10, 15) :
        label = pEncoder.inverse_transform([i])[0]
        plt.plot(per_class_history[:epochs - 1, i, 2], label=label)

    plt.grid()
    plt.legend()
    plt.subplot(2,2,4)
    for i in range(15, 18) :
        label = pEncoder.inverse_transform([i])[0]
        plt.plot(per_class_history[:epochs - 1, i, 2], label=label)

    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(macro_history[:epochs - 1, 2], label='macro F1 with <UNK> token')
    plt.plot(macro_history[:epochs - 1, 3], label='macro F1 without <UNK> token')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(loc="best")
    plt.legend()
    plt.grid()
    plt.show()

    print("Train Loss     : {0:.5f}".format(history['loss'][-1]))
    print("Dev Loss: {0:.5f}".format(history['val_loss'][-1]))
    print("Test Loss      : {0:.5f}".format(loss))
    print("---")
    print("Train Accuracy     : {0:.5f}".format(history['categorical_accuracy'][-1]))
    print("Dev Accuracy: {0:.5f}".format(history['val_categorical_accuracy'][-1]))
    print("Test Accuracy      : {0:.5f}".format(acc))

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(1, 2, 1)
    plot_history(hs={'Init model': history}, epochs=epochs, metric='categorical_accuracy')
    plt.grid()

    plt.subplot(1, 2, 2)
    plot_history(hs={'Init model': history}, epochs=epochs, metric='loss')

    plt.grid()
    plt.show()

    return

def plot_score(pTest, pPred, pNumClasses) :

    y_tets_arg = np.argmax(pTest, axis=1)
    y_pred_arg = np.argmax(pPred, axis=1)

    print('accuracy:', accuracy_score(y_tets_arg, y_pred_arg))
    print('macro-f1-score:', f1_score(y_tets_arg, y_pred_arg, average='macro'))
    print('micro-f1-score:', f1_score(y_tets_arg, y_pred_arg, average='micro')) 
    print(sk_metrics.classification_report( y_tets_arg, y_pred_arg))

    y_test = pTest.ravel()
    y_pred = pPred.ravel()

    precision = {}
    recall = {}
    average_precision = {}

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test, y_pred)

    average_precision["micro"] = average_precision_score(
        y_test, y_pred, average="micro")

    plt.figure()

    plt.plot(recall["micro"], precision["micro"], color='navy',
        label='average AUC= {0:0.2f}'.format(average_precision['micro']))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")

    plt.grid()
    plt.show()

    return

def plot_history(hs, epochs, metric):
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['font.size'] = 16

    for label in hs:
        plt.plot(hs[label][metric], label='train {0:s}'.format(metric))
        plt.plot(hs[label]['val_{0:s}'.format(metric)], label='val {0:s}'.format(metric))

    x_ticks = np.arange(0, epochs + 1)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()

def test_model(pModel, pEncoder, embeddings, pSentence) :

    x, y = data_generators.buildData_word2vec(pSentence, 5, embeddings, pEncoder)

    predict = pModel.predict(x)

    y_tets_arg = np.argmax(y, axis=1)
    y_pred_arg = np.argmax(predict, axis=1)

    print(pEncoder.inverse_transform(list(y_tets_arg)))
    print(pEncoder.inverse_transform(list(y_pred_arg)))
    return