import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt

train_pattern = glob.glob('/data/MBP_DATA_TEST/*/*[!l].npy')
label_pattern = glob.glob('/data/MBP_DATA_TEST/*/*label.npy')
test_label_pattern = glob.glob('/data/MBP_DATA/10/*label.npy')
test_inp_pattern = glob.glob('/data/MBP_DATA/10/*[!l].npy')


def load_numpy(load_file):
    data = tf.numpy_function(np.load, [load_file], tf.float64)
    return data

X_dataset = tf.data.Dataset.list_files(train_pattern, shuffle=False)
X_dataset = X_dataset.map(load_numpy)


y_dataset = tf.data.Dataset.list_files(label_pattern, shuffle=False)
y_dataset = y_dataset.map(load_numpy)

train_dataset = tf.data.Dataset.zip((X_dataset, y_dataset)).batch(12, drop_remainder=True)
train_dataset = train_dataset.map(lambda x,y : (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))

X_test_dataset = tf.data.Dataset.list_files(test_inp_pattern, shuffle=False)
X_test_dataset = X_test_dataset.map(load_numpy)


y_test_dataset = tf.data.Dataset.list_files(test_label_pattern, shuffle=False)
y_test_dataset = y_test_dataset.map(load_numpy)

test_dataset = tf.data.Dataset.zip((X_test_dataset, y_test_dataset)).batch(12, drop_remainder=True)
test_dataset = test_dataset.map(lambda x,y : (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))

if __name__=="__main__":
    import module
    gen = module.generator()
    dis = module.discriminator()

    for n, (input_image, target) in test_dataset.enumerate():
        print(input_image.shape)
        fig = plt.figure()
        plt.plot(input_image[0][0][:500])
        fig.savefig('fig.png', dpi=300)
        break 
