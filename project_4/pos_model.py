import layers
import tensorflow as tf
import numpy as np
import os
import metrics
import pyconll
import gzip
import json
import gc
import data_generator
import math
import metrics

from random import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, GRU
from keras.layers import Dense, TimeDistributed, BatchNormalization, concatenate
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from nltk import ngrams

class PoStagger:

    def __init__(self, seq_len=25):
        self.model = None
        self.seqLen = seq_len
        self.batchSize = 128
        self.labelEncoder = None
        self.root = os.getcwd() + "\\dataset\\"
        self.name = "base"
        return

    def numClasses(self):
        return len(self.labelEncoder.classes_)

    def load_data(self, pData, pPadding):
        x = []
        y = []
        max_len = {}

        for sentence in pData:
            xx = []
            yy = []

            if len(sentence) in max_len:
                max_len[len(sentence)] += 1
            else:
                max_len[len(sentence)] = 1

            for i, token in enumerate(sentence):
                xx.append(token.form if token.form is not None else "__NONE__")
                yy.append(token.upos if token.upos is not None else "__NONE__")

            for i in range(len(xx), pPadding):
                xx.append('__PAD__')
                yy.append('__PAD__')
            x.append(xx)
            y.append(yy)

        tmpx = []
        tmpy = []
        for i, sentence in enumerate(x):
            if len(sentence) > pPadding:
                gx = ngrams(sentence, pPadding)
                gy = ngrams(y[i], pPadding)
                for gram in gx:
                    tmpx.append(list(gram))
                for gram in gy:
                    tmpy.append(list(gram))
            else:
                tmpx.append(sentence)
                tmpy.append(y[i])
        # print("Max len sentence: ", max_len.items())
        return tmpx, tmpy

    def export_json(self, pFile, pData):
        with gzip.GzipFile(pFile + ".json.gz", "wb") as file:
            file.write(json.dumps(pData, indent=4).encode('utf-8'))
        return

    def import_json(self, pFile):
        with gzip.GzipFile(pFile + ".json.gz", "rb") as file:
            data = json.loads(file.read().decode('utf-8'))
        return data

    def export_arr(self, pFile, pX, pY):
        np.savez_compressed(pFile, a=pX, b=pY)
        return

    def import_arr(self, pFile):
        data = np.load(pFile + ".npz")
        return data['a'], data['b']

    def build_labels(self, pTrain, pVal, pTest, pDev):
        labels = {}
        for dataset in [pTrain, pVal, pTest, pDev]:
            for sentence in dataset:
                for token in sentence:
                    labels[token] = token

        print("Num labels: ", len(labels.keys()))

        label_encoder = LabelEncoder()
        label_encoder.fit([l for l in labels.keys()])

        return label_encoder

    def build_categ_labels(self, pY):
        y = np.zeros((len(pY), self.seqLen, self.numClasses()))

        for i, tags in enumerate(pY):
            for j, tag in enumerate(tags):
                y[i, j, self.labelEncoder.transform([tag])[0]] = 1.0

        return y

    def compile_dataset(self):
        train_path = self.root + "en_ewt-ud-train.conllu"
        dev_path = self.root + "en_ewt-ud-dev.conllu"
        test_path = self.root + "en_ewt-ud-test.conllu"

        train_data = pyconll.load_from_file(train_path)
        dev_data = pyconll.load_from_file(dev_path)
        test_data = pyconll.load_from_file(test_path)

        trainX, trainY = self.load_data(train_data, self.seqLen)
        devX, devY = self.load_data(dev_data, self.seqLen)
        testX, testY = self.load_data(test_data, self.seqLen)

        xtrain, xval, ytrain, yval = train_test_split(
            trainX, trainY, test_size=0.1)

        self.labelEncoder = self.build_labels(ytrain, devY, testY, yval)
        np.savez_compressed(self.root + "labelEncoder", self.labelEncoder.classes_)

        self.export_arr(self.root + "train", np.array(xtrain), self.build_categ_labels(ytrain))
        self.export_arr(self.root + "val", np.array(xval), self.build_categ_labels(yval))
        self.export_arr(self.root + "dev", np.array(devX), self.build_categ_labels(devY))
        self.export_arr(self.root + "test", np.array(testX), self.build_categ_labels(testY))

        return

    def load_model(self):
        self.model.load_weights(self.root + "base_model")
        return

    def predict(self, pSentence) :
        sentence = pSentence
        initLen = len(pSentence)
        for i in range(len(pSentence), self.seqLen):
            sentence.append('__PAD__')

        sample = np.zeros((self.batchSize, self.seqLen), dtype='|S6')
        sample[0,:] = sentence

        prediction = self.model.predict(sample, steps=1)
        prediction = np.argmax(prediction[0, :], axis=1).ravel()

        return self.labelEncoder.inverse_transform(list(prediction))[:initLen]

    def test_model(self) :
        devX, devY = self.import_arr(self.root + "test")
        metric = metrics.Metrics((devX, devY), self.batchSize, self.labelEncoder)
        metric.get_metrics(self.model)
        return

    def compile_base_model(self) :
        gc.collect()
        self.labelEncoder = LabelEncoder()
        self.labelEncoder.classes_ = np.load(self.root + "labelEncoder.npz")['arr_0']

        inputs = Input(shape=(self.seqLen,), name='input', dtype=tf.string)
        elmo = layers.ElmoLayer(self.seqLen, self.batchSize)(inputs)
        gru1 = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.5, return_sequences=True, name='gru1'))(elmo)
        bn1 = BatchNormalization()(gru1)
        residual = concatenate([elmo, bn1])
        gru2 = Bidirectional(GRU(100, dropout=0.2, recurrent_dropout=0.5, return_sequences=True, name='gru2'))(residual)
        bn2 = BatchNormalization()(gru2)
        output = TimeDistributed(Dense(self.numClasses(), activation='softmax'))(bn2)
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy',])
        model.summary()

        self.model = model
        self.name = "base_model"
        return


    def compile_model(self):
        self.compile_base_model()
        return

    def train_model(self):
        trainX, trainY = self.import_arr(self.root + "train")
        devX, devY = self.import_arr(self.root + "dev")
        '''
        trainX = trainX[:1000]
        trainY = trainY[:1000]
        devX = devX[:1000]
        devY = devY[:1000]
        '''
        train_gen = data_generator.data_stream([trainX, trainY], self.batchSize, self.numClasses())
        dev_gen = data_generator.data_stream([devX, devY], self.batchSize, self.numClasses())

        stopper = EarlyStopping(monitor='val_loss',
                                min_delta=0, patience=5,
                                verbose=0, mode='auto',
                                restore_best_weights=True)

        csv_logger = CSVLogger(self.root + 'logger.log')

        self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=math.ceil(len(trainX) / self.batchSize),
            validation_data=dev_gen,
            validation_steps=math.ceil(len(devX) / self.batchSize),
            callbacks=[ \
                metrics.Metrics((devX, devY), self.batchSize, self.labelEncoder), \
                stopper, csv_logger],
            epochs=20,
            verbose=1,
            max_queue_size=100,
            workers=1,
            use_multiprocessing=False, )

        self.model.save_weights(self.root + self.name)
        return
