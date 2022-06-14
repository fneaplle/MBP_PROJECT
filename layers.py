import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa


weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)

def Dense(units, activation=None):
    op = tf.keras.layers.Dense(units=units, activation=activation, use_bias=True, kernel_initializer=weight_initializer, bias_initializer='zeros', 
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    return op

def Droupout(rate):
    op = tf.keras.layers.Dropout(rate=rate, noise_shape=None, seed=None)
    return op
                                 
def Conv1D(cls, filters, kernel_size, strides=1, padding='valid', activation=None, use_bias=True):
    op = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, data_format='channels_last',
                              dilation_rate=1, activation=None, use_bias=use_bias,
                              kernel_initializer=cls.weights_initializer, bias_initializer='zeros',
                              kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                              kernel_constraint=None, bias_constraint=None)

    return op

def DeConv1D(cls, filters, kernel_size, strides=1, padding='valid', use_bias=True):
    op = tf.keras.layers.Conv2DTranspose(filters= filters, kernel_size=(1, kernel_size), strides=(1, strides), padding=padding, 
                                          output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=use_bias,
                                          kernel_initializer=cls.weights_initializer, bias_initializer='zeros', kernel_regularizer=None, 
                                          bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
      
    return op

def BatchNormalization(trainable=True, virtual_batch_size=None):
    op = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                              gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
                                              fused=None, trainable=trainable, virtual_batch_size=virtual_batch_size, adjustment=None, name=None)

    return op

def BatchNormalization(trainable=True, virtual_batch_size=None):
    op = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                              beta_initializer='zeros', gamma_initializer='ones',
                                              moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                              beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                              gamma_constraint=None, renorm=False, renorm_clipping=None, renorm_momentum=0.99,
                                              fused=None, trainable=trainable, virtual_batch_size=virtual_batch_size, adjustment=None, name=None)
      
    return op

def normalization(name):
   if name =='none':
       return lambda: lambda x: x
   elif name == 'batch_norm':
       return keras.layers.BatchNormalization()
   elif name == 'instance_norm':
       return tfa.layers.InstanceNormalization()
   elif name == 'layer_norm':
       return keras.layers.LayerNormalization()

def Activation(x, activation):
   if activation == 'relu':
       return tf.keras.activations.relu(x, alpha=0.0, max_value=None, threshold=0)
   elif activation == 'leaky_relu':
       return tf.keras.activations.relu(x, alpha=0.2, max_value=None, threshold=0)
   elif activation == 'sigmoid':
       return tf.keras.activations.sigmoid(x)
   elif activation == 'softmax':
       return tf.keras.activations.softmax(x, axis=-1)
   elif activation == 'tanh':
       return tf.keras.activations.tanh(x)
   else:
       raise ValueError('please check the name of the activation')


