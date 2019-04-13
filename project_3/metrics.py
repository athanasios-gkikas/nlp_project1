from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

import numpy as np

class Metrics(Callback) :

    def __init__(self, pNumClasses, pValSet, pLabelEnc, **kwargs):

        super(Metrics, self).__init__(**kwargs)

        self.valX = pValSet[0]
        self.valY = pValSet[1]
        self.numClasses = pNumClasses
        self.labelEnc = pLabelEnc
        self.fscores = np.zeros((100, self.numClasses))

    def getMetrics(self, pModel) :

        prediction = pModel.predict(self.valX)
        gt = np.argmax(self.valY, axis=1)
        prediction = np.argmax(prediction, axis=1)

        for i in range(0, self.numClasses) :
            trueY = gt == i
            predY = prediction == i
            label = self.labelEnc.inverse_transform([i])
            print("class ", label, " : {0:.4f}".format(f1_score(predY, trueY)))

        return

    def on_epoch_begin(self, epoch, logs=None) :
        pass

    def on_epoch_end(self, epoch, logs=None) :

        self.getMetrics(self.model)

        #self.getMetrics(epoch, prediction, gt)

        #macro_fscore = np.mean(self.fscores[epoch,:])

        #print(" val macro-fscore: ", "{0:.4f}".format(macro_fscore))

        #if(macro_ji > 0.98) :
        #    self.model.stop_training = True

        return

    def on_batch_end(self, batch, logs={}) :
        pass

    def on_train_begin(self, logs=None) :
        pass

    def on_train_end(self, logs=None) :
        return
