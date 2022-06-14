import tensorflow as tf
import tensorflow.keras as keras
import layers
import pipeline
from sklearn.decomposition import IncrementalPCA


def figure_layer(ip):
    fig = plt.figure()
    fig.title('figure in layers')
    plt.savefig('./figure_in_layers')

'''
#(2,6000) -> (,32)
def embedding(input_shape=(2,6000)):
    inp = tf.keras.layers.Input(shape=input_shape)
    h = tf.keras.layers.Flatten()(inp)
    h = tf.keras.layers.Dense(30)(h)
    return tf.keras.Model(inputs=inp, outputs=h)
'''

class embedding():
    def __init__(self):
        self.rbf_pca = IncrementalPCA(n_components=30)
    def __call__(self, inp, training=False):
        if training==True:
            inp = inp.numpy()
            print(inp.shape)
            inp = inp.reshape((1, 6000*2))
            print(inp.shape)
            self.rbf_pca.partial_fit(inp)
            return self.rbf_pca.tranform(inp)
        else:
            inp = inp.numpy()
            inp = inp.reshape((1, 6000*2))
            return self.rbf_pca.transform(inp)

def generator(input_shape=(30,)):
    def _downsample(inp, norm):
        inp = tf.keras.layers.Conv2D(128, kernel_size=(1, 2), padding='same', strides=(1, 2))(inp)
        if norm != 'none':
            inp = layers.normalization(norm)(inp)
        inp = layers.Activation(inp, activation='leaky_relu')
        return inp
    
    def _upsample(inp, norm, drop_rate=0.5, apply_dropout=False):
        inp = tf.keras.layers.Conv2DTranspose(128, kernel_size=(1, 2), padding='same', strides=(1, 2))(inp)
        if norm != 'none':
            inp = layers.normalization(norm)(inp)
        if apply_dropout:
            inp = layers.Dropout(rate=drop_rate)
        inp = layers.Activation(inp, activation='relu')
        return inp
    
    inp = tf.keras.layers.Input(shape=input_shape)    
    h = inp
    h = tf.expand_dims(h, axis=1)
    h = tf.expand_dims(h, axis=3)
    h = tf.keras.layers.ZeroPadding2D(padding=(0,1))(h)

    #downsample step
    skips = []
    for i in range(5):
        if i==0:
            h = _downsample(h, 'none')
        else:
            h = _downsample(h, 'layer_norm')
        skips.append(h)
    skips = reversed(skips[:-1])

    #upsample step

    for skip in skips:
        h = _upsample(h, 'layer_norm')
        h = tf.keras.layers.Concatenate()([h, skip])

    h = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same', strides=1)(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(30)(h)

    return keras.Model(inputs=inp, outputs=h)

def discriminator(input_shape_inp=(30,)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape_inp, name='inp_signal')
    tar = tf.keras.layers.Input(shape=input_shape_inp, name='tar_signal')
    
    inp_h = tf.expand_dims(inp, axis=1)
    inp_h = tf.expand_dims(inp_h, axis=-1)

    tar_h = tf.expand_dims(tar, axis=1)
    tar_h = tf.expand_dims(tar_h, axis=-1)

    h = tf.keras.layers.concatenate([inp_h, tar_h])
    h = tf.keras.layers.Conv2D(1, (1, 10))(h)
    h = layers.BatchNormalization('layer_norm')(h)
    h = layers.Activation(h, activation='leaky_relu')
    h = tf.keras.layers.Conv2D(1, (1, 3))(h)
    
    last = tf.squeeze(h)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss 

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


if __name__=="__main__":
    generator_model = generator()
    discriminator_model = discriminator()
    embedding_model = embedding() 
    for X, y in pipeline.train_dataset.take(1):
        em = embedding_model(X, training=True)
        gr = generator_model(em)
        print(gr.shape)
        dr = discriminator_model((gr, y))
        print(dr.shape)
