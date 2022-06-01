import tensorflow as tf
import tensorflow.keras as keras
import layers
import pipeline

def generator(input_shape=(2, 6000)):
    def _downsample(ip):
        ip = tf.keras.layers.Conv2D(1, (2, 5971))(ip)
        return ip

    h = inputs = keras.Input(shape=input_shape)
    h = tf.expand_dims(h, axis=3)
    
    h = _downsample(h)
    h = tf.squeeze(h)
    return keras.Model(inputs=inputs, outputs=h)

def discriminator(input_shape_inp=(30,)):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=input_shape_inp, name='trg_signal')
    h = tf.expand_dims(inp, axis=1)
    h = tf.expand_dims(h, axis=-1)
    down = tf.keras.layers.Conv2D(1, (1, 10))(h)
    last = tf.keras.layers.Conv2D(1, (1, 3))(down)
    return tf.keras.Model(inputs=inp, outputs=last)

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
    
    for X, y in pipeline.train_dataset.batch(1).take(1):
        gr = generator_model(X)
        dr = discriminator_model(gr)
        print(gr)
        print(dr)
