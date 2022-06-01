import tensorflow as tf
import matplotlib.pyplot as plt
import os
import module
import pipeline
from train import generate_images

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
    plt.savefig(f'./figures/prediction/image{epoch}.png')

generator = module.generator()
discriminator = module.discriminator()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on a few examples from the test dataset
for n, (inp, tar) in pipeline.test_dataset.take(5).enumerate():
  generate_images(generator, inp, tar, n)
