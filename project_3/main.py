from sklearn.preprocessing import LabelEncoder
from keras.models import load_model

import data_generators
import data_loaders
import models
import metrics
import os
import numpy as np
from gensim.models import KeyedVectors

def stats(pY, pLabelEnc) :

    y = np.argmax(pY, axis=1)

    for i in range(0, len(pLabelEnc.classes_)) :
        mask = np.count_nonzero(y == i)
        print(i, " ", pLabelEnc.inverse_transform([i])[0], " ", mask)

    return

def main() :

    data_loaders.exportCorpus()
    data_generators.export_word2vec_dataset()

    cwd = os.getcwd()
    trainX, trainY = data_loaders.import_embeddings("train")
    valX, valY = data_loaders.import_embeddings("val")
    testX, testY = data_loaders.import_embeddings("test")
    devX, devY = data_loaders.import_embeddings("dev")

    encoder = LabelEncoder()
    encoder.classes_ = np.load(cwd + "/dataset/labelEncoder.npz")['arr_0']

    model = models.build_mlp(trainX.shape[1], len(encoder.classes_))
    model = models.train_model(model, (trainX, trainY), (devX, devY), encoder)
    model.load_weights(cwd + "/dataset/mlp1")
    metrics.evaluate_model(model, (testX, testY), encoder)

    '''
    
    cwd = os.getcwd()

    embeddings = KeyedVectors.load_word2vec_format(
        cwd + "/dataset/GoogleNews-vectors-negative300.bin", binary=True)

    sentence1 = [
        [['__PAD__', '__PAD__', 'I', 'love', 'you'], 'X'],
        [['__PAD__', 'I', 'love', 'you', '__PAD__'], 'VERB'],
        [['I', 'love', 'you', '__PAD__', '__PAD__'], 'PRON'],]

    sentence2 = [
        [['__PAD__', '__PAD__', 'Let', 'make', 'love'], 'VERB'],
        [['__PAD__', 'Let', 'make', 'love', '__PAD__'], 'VERB'],
        [['Let', 'make', 'love', '__PAD__', '__PAD__'], 'NOUN']]

    sentence = [
        [['__PAD__', '__PAD__', 'Google', 'is', 'a'], 'PROPN'],
        [['__PAD__', 'Google', 'is', 'a', 'nice'], 'AUX'],
        [['Google', 'is', 'a', 'nice', 'search'], 'DET'],
        [['is', 'a', 'nice', 'search', 'engine'], 'ADJ'],
        [['a', 'nice', 'search', 'engine', '.'], 'NOUN'],
        [['nice', 'search', 'engine', '.', '__PAD__'], 'NOUN'],
        [['search', 'engine', '.', '__PAD__', '__PAD__'], 'PUNCT'],]

    metrics.test_model(model, encoder, embeddings, sentence1)
    metrics.test_model(model, encoder, embeddings, sentence2)
    '''

    return

if __name__ == '__main__' :
    main()