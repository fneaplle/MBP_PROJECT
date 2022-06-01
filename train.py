import tensorflow as tf
import module
import pipeline
import time
from IPython import display
import matplotlib.pyplot as plt
import os

discriminator = module.discriminator()
generator = module.generator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(generator, example_input, example_tar, epoch):
    prediction = generator(example_input, training=True)
    plt.figure(figsize=(15,3))
    display_list = [example_input[0][0][:500], example_input[0][1][:500], example_tar[0], prediction[0]]
    title = ['Input_EKG','Input_PPG', 'Ground True', 'Predicted Image']

    for i in range(4):
        plt.subplot(1,4, i+1)
        plt.title(title[i])
        plt.plot(display_list[i])
        plt.axis('off')
    plt.savefig(f'./figures/train/image{epoch}.png')

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([target], training=True)
        disc_generated_output = discriminator([gen_output], training=True)

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
        
        for example_input, example_target in pipeline.test_dataset.take(1):
            generate_images(generator, example_input, example_target, epoch)

        for n, (input_image, target) in pipeline.train_dataset.enumerate():
            train_step(input_image, target, epoch)
        
        if(epoch+1) % 20 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

if __name__=='__main__':
    fit(pipeline.train_dataset, 40)







