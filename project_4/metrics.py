from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, average_precision_score
import data_generator
import sklearn.metrics as sk_metrics
from keras.callbacks import Callback
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import math
import json
import gzip


class Metrics(Callback):

    def __init__(self, val_set, batch_size, label_enc, **kwargs):
        super(Metrics, self).__init__(**kwargs)
        self.valX = val_set[0]
        self.valY = np.argmax(val_set[1], axis=2).ravel()
        self.labelEnc = label_enc
        self.tags = self.labelEnc.inverse_transform([tag for tag in range(len(self.labelEnc.classes_))])
        self.batchSize = batch_size
        self.report = []
        self.epochs = 0

    def get_metrics(self, model):
        pred = np.zeros(self.valY.shape, dtype=int)
        batches = math.ceil(self.valX.shape[0] / self.batchSize)
        seq_len = self.valX.shape[1]

        for i in range(batches):
            sample = np.zeros((self.batchSize, seq_len), dtype=self.valX.dtype)

            offset = i * self.batchSize
            end = offset + self.batchSize

            if end > self.valX.shape[0]:
                tmp = self.valX[offset:, :]
                sample[:tmp.shape[0], :] = tmp
            else:
                sample = self.valX[offset:end, :]

            prediction = np.argmax(model.predict(sample, steps=1), axis=2).ravel()

            offset = i * seq_len * self.batchSize
            end = offset + seq_len * self.batchSize

            if end > pred.shape[0]:
                pred[offset:] = prediction[:pred.shape[0] - offset]
            else:
                pred[offset:end] = prediction

        print("Accuracy: ", accuracy_score(self.valY, pred))
        print(classification_report(self.valY, pred, target_names=self.tags))

        return classification_report(self.valY, pred, target_names=self.tags, output_dict=True)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        report = self.get_metrics(self.model)
        self.report.append(report)
        return

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        with gzip.GzipFile(os.getcwd() + "\\dataset\\report.json.gz", "wb") as file:
            file.write(json.dumps(self.report, indent=4).encode('utf-8'))
        return


def evaluate_model(model):
    encoder = LabelEncoder()
    encoder.classes_ = np.load(os.getcwd() + "/dataset/labelEncoder.npz")['arr_0']
    per_class_history = load_per_class_data(model, encoder)
    accuracy_loss_metrics(per_class_history)
    return
    plt.figure()
    print(len(encoder.classes_))
    plt.subplot(2, 2, 1)
    per_class_plots(0, 5, encoder, per_class_history)
    plt.subplot(2, 2, 2)
    per_class_plots(5, 10, encoder, per_class_history)
    plt.subplot(2, 2, 3)
    per_class_plots(10, 15, encoder, per_class_history)
    plt.subplot(2, 2, 4)
    per_class_plots(15, 18, encoder, per_class_history)
    plt.show()

    return

def accuracy_loss_metrics(pPerClassHistory):
    with open(os.getcwd() + '/dataset/logger.log', 'r') as log:
        rows = csv.DictReader(log)
        history = {}
        for i, row in enumerate(rows):
            for key, value in row.items():
                if key in history:
                    history[key] += [float(value), ]
                else:
                    history[key] = [float(value), ]

    epochs = len(history['epoch'])

    plt.figure()
    plt.subplot(1, 3, 1)
    plot_history(hs={'Init model': history}, epochs=epochs, metric='categorical_accuracy')
    plt.grid()

    plt.subplot(1, 3, 2)
    plot_history(hs={'Init model': history}, epochs=epochs, metric='loss')

    plt.grid()
    #plt.show()

    plt.subplot(1, 3, 3)
    macro_f1_per_epoch = calculate_macro_f1(pPerClassHistory, True)

    plt.plot(list(macro_f1_per_epoch.keys()), list(macro_f1_per_epoch.values()), label='macro F1 with PAD')
    plt.xlabel('Epochs')
    #plt.ylabel('Scores')
    plt.legend(loc="best")
    macro_f1_per_epoch = calculate_macro_f1(pPerClassHistory, False)
    plt.plot(list(macro_f1_per_epoch.keys()), list(macro_f1_per_epoch.values()), label='macro F1 without PAD')
    plt.xlabel('Epochs')
    #plt.ylabel('Scores')
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    return


def load_per_class_data(model, encoder):
	data = model.import_json(os.getcwd() + "\\dataset\\report")
	per_class_history = {}
	epoch_counter = 1
	for i in range(0, 18):
		per_class_history[i] = defaultdict(list)
	for epoch in data:
		for i in range(0, 18):
			label = encoder.inverse_transform([i])[0]
			per_class_history[i][epoch_counter] = epoch[label]['f1-score']
		epoch_counter = epoch_counter + 1

	return per_class_history


def calculate_macro_f1(per_class_history, pWithPAD):
	epochs = len(per_class_history[0])
	f1_per_epoch = {}
	for epoch in range(1, epochs + 1):
		total_f1 = 0
		for i in range(0, (18 if pWithPAD else 17)):
			total_f1 = total_f1 + per_class_history[i][epoch]
		f1_per_epoch[epoch] = total_f1/(18 if pWithPAD else 17)

	return f1_per_epoch


def per_class_plots(start, end, encoder, per_class_history):
    for i in range(start, end):
        label = encoder.inverse_transform([i])[0]
        if label == "__PAD__" :
            label = "PAD"
        plt.plot(list(per_class_history[i].keys()), list(per_class_history[i].values()), label=label)

    plt.xlabel('epoch')
    plt.ylabel('F1 macro')
    plt.grid()
    plt.legend()

    return


def plot_history(hs, epochs, metric):
	plt.rcParams['figure.figsize'] = [10, 5]
	plt.rcParams['font.size'] = 16

	for label in hs:
		plt.plot(hs[label][metric], label='train {0:s}'.format(metric))
		plt.plot(hs[label]['val_{0:s}'.format(metric)], label='val {0:s}'.format(metric))

	x_ticks = np.arange(0, epochs + 1)
	x_ticks[0] += 1
	plt.xticks(x_ticks)
	plt.ylim((0, 1))
	plt.xlabel('Epochs')
	plt.ylabel('Loss' if metric == 'loss' else 'Accuracy')
	plt.legend()