import keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Layer

class ElmoLayer(Layer) :
    def __init__(self, seq_len, batchSize, **kwargs) :
        self.elmo = None
        self.name = "ELMo"
        self.seqLen = seq_len
        self.batchSize = batchSize
        super(ElmoLayer, self).__init__(**kwargs)

    def build(self, input_shape) :
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2',
            trainable=True, name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(
            scope="^{}_module/.*".format(self.name))

        super(ElmoLayer, self).build(input_shape)

    def call(self, x, mask=None) :
        return self.elmo(inputs={
            "tokens": tf.squeeze(tf.cast(x, tf.string)),
            "sequence_len": tf.constant(self.batchSize * [self.seqLen])
        }, as_dict=True, signature='tokens')['elmo']

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1024

    def compute_mask(self, inputs, mask=None):
        return None