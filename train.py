import tensorflow as tf
import module
import pipeline
import time
from IPython import display
import matplotlib.pyplot as plt
import os

discriminator = module.discriminator()
generator = module.generator()
embedding = module.embedding()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
embedding_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 embedding_optimizer=embedding_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 embedding=embedding)

def generate_images(generator, embedded_inp, example_input, example_tar, epoch):
    prediction = generator(embedded_inp, training=True)
    plt.figure(figsize=(15,3))
    display_list = [example_input[0][0][:500], example_input[0][1][:500], example_tar[0], prediction[0]]
    title = ['Input_EKG','Input_PPG', 'Ground True', 'Predicted Image']

    for i in range(4):
        plt.subplot(1,4, i+1)
        plt.title(title[i])
        plt.plot(display_list[i])
        plt.axis('off')
    plt.savefig(f'./figures/train/image{epoch}.png')
    print('save figure complete!')

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        embedding_inp = embedding(input_image, training=False)
        gen_output = generator(embedding_inp, training=True)

        disc_real_output = discriminator([embedding_inp, target], training=True)
        disc_generated_output = discriminator([gen_output, target], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = module.generator_loss(disc_generated_output, gen_output, target)
        disc_loss = module.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))


        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs):
    for epoch in range(epochs):
        print("Epoch : ", epoch)

        start = time.time()
        display.clear_output(wait=True)
        
        if (epoch+1)%20 == 0:
            for example_input, example_target in pipeline.train_dataset:
                embedded_inp = embedding(example_input)
                generate_images(generator, embedded_inp, example_input, example_target, epoch+1)

        for n, (input_image, target) in pipeline.train_dataset.enumerate():
            print(f'{n}step data training....')
            train_step(input_image, target, epoch)
        
        if(epoch+1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

if __name__=='__main__':
    fit(pipeline.train_dataset, 200)

