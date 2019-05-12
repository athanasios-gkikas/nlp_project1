from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from keras import backend as K
from gensim.models import KeyedVectors

import data_generator
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
import json
import gzip

class Metrics(Callback) :

    def __init__(self, pValSet, pBatchSize, pLabelEnc, **kwargs):
        super(Metrics, self).__init__(**kwargs)
        self.valX = pValSet[0]
        self.valY = np.argmax(pValSet[1], axis=2).ravel()
        self.labelEnc = pLabelEnc
        self.tags = self.labelEnc.inverse_transform( \
            [tag for tag in range(len(self.labelEnc.classes_))])
        self.batchSize = pBatchSize
        self.mReport = []

    def getMetrics(self, pModel) :
        pred = np.zeros(self.valY.shape, dtype=int)
        batches = math.ceil(self.valX.shape[0] / self.batchSize)
        seqLen = self.valX.shape[1]

        for i in range(batches) :
            sample = np.zeros((self.batchSize, seqLen), dtype=self.valX.dtype)

            offset = i * self.batchSize
            end = offset + self.batchSize

            if end > self.valX.shape[0]:
                tmp = self.valX[offset:, :]
                sample[:tmp.shape[0], :] = tmp
            else :
                sample = self.valX[offset:end, :]

            prediction = np.argmax(pModel.predict(sample, steps=1), axis=2).ravel()

            offset = i * seqLen * self.batchSize
            end = offset + seqLen * self.batchSize

            if end > pred.shape[0] :
                pred[offset:] = prediction[:pred.shape[0] - offset]
            else :
                pred[offset:end] = prediction

        print(classification_report(self.valY, pred,
            target_names=self.tags))

        return classification_report(self.valY, pred,
            target_names=self.tags, output_dict=True)

    def on_epoch_begin(self, epoch, logs=None) :
        pass

    def on_epoch_end(self, epoch, logs=None) :
        report = self.getMetrics(self.model)
        self.mReport.append(report)
        return

    def on_batch_end(self, batch, logs=None) :
        pass

    def on_train_begin(self, logs=None) :
        pass

    def on_train_end(self, logs=None) :
        with gzip.GzipFile(os.getcwd() + "\\dataset\\report.json.gz", "wb") as file:
            file.write(json.dumps(self.mReport, indent=4).encode('utf-8'))
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