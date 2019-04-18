
from gensim.models import KeyedVectors
from keras.utils import to_categorical, np_utils
from sklearn.preprocessing import LabelEncoder

import data_loaders
import gensim
import numpy as np
import os


def export_word2vec_dataset() :

    data_loaders.exportCorpus()

    train, val, dev, test = data_loaders.importCorpus()
    encoder = build_labels(train, val, dev, test)

    cwd = os.getcwd()

    np.savez_compressed(cwd + "/dataset/labelEncoder", encoder.classes_)

    print("Loading pre-trained word2vec embeddings...")

    embeddings = KeyedVectors.load_word2vec_format(
        cwd + "/dataset/GoogleNews-vectors-negative300.bin", binary=True)

    print("Extracting embeddings...")

    trainX, trainY = buildData_word2vec(train, 5, embeddings, encoder)
    testX, testY = buildData_word2vec(test, 5, embeddings, encoder)
    valX, valY = buildData_word2vec(val, 5, embeddings, encoder)
    devX, devY = buildData_word2vec(dev, 5, embeddings, encoder)
    data_loaders.export_embeddings(trainX, trainY, "train")
    data_loaders.export_embeddings(testX, testY, "test")
    data_loaders.export_embeddings(valX, valY, "val")
    data_loaders.export_embeddings(devX, devY, "dev")

    return

def buildData_word2vec(pData, pWindow, pEmbeddings, pLabelEncoder) :

    x = np.zeros((len(pData), 300 * pWindow), dtype=np.float32)

    for i, pair in enumerate(pData) :
        for j, token in enumerate(pair[0]) :
            if token != "__PAD__" :
                if token in pEmbeddings.vocab :
                    x[i, j * 300 : j * 300 + 300] = pEmbeddings[token]
                else :
                    x[i, j * 300 : j * 300 + 300] = pEmbeddings["UNK"]
                    if j == int(pWindow / 2) :
                        pair[0][j] = "UNK"
                        pair[1] = "UNK"

    return x, to_categorical(pLabelEncoder.transform([tag[1] for tag in pData]))

def build_labels(pTrain, pVal, pTest, pDev) :

    labels = {}
    labels["UNK"] = "UNK"

    for dataset in [pTrain, pVal, pTest, pDev] :
        for token in dataset :
            labels[token[1]] = token[1]

    print("Num labels: ", len(labels.keys()))

    label_encoder = LabelEncoder()
    label_encoder.fit([l for l in labels.keys()])

    return label_encoder