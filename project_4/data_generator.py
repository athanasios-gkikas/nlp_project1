import threading as th
import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

class threadsafe_iter :

    def __init__(self, it) :
        self.it = it
        self.lock = th.Lock()

    def __iter__(self) :
        return self

    def __next__(self) :
        with self.lock :
            return self.it.next()

def threadsafe_generator(f) :
    def g(*g, **kw) :
        return threadsafe_iter(f(*g, **kw))
    return g

#@threadsafe_generator
def data_stream(pDataPairs, pBatchSize, pNumClasses) :

    currBatch = 0
    x = pDataPairs[0]
    y = pDataPairs[1]

    while True :

        batchX = np.zeros((pBatchSize, x.shape[1]), dtype=x.dtype)
        batchY = np.zeros((pBatchSize, x.shape[1], pNumClasses))

        for i in range(pBatchSize) :

            offset = (currBatch + i) % len(x)

            batchX[i, :] = x[offset, :]
            batchY[i, :, :] = y[offset, :, :]

        currBatch += pBatchSize

        if currBatch > len(x) - 1 :
            x, y = shuffle(x, y)
            currBatch = 0

        yield batchX, batchY

    return