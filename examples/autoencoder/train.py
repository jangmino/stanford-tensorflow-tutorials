import tensorflow as tf
import numpy as np

from utils import *
from autoencoder import *

batch_size = 100
batch_shape = (batch_size, 28, 28, 1)
num_visualize = 10

lr = 0.01
num_epochs = 50

def calculate_loss(original, reconstructed):
    return tf.div(tf.reduce_sum(tf.square(tf.subtract(reconstructed,
                                                 original))), 
                  tf.constant(float(batch_size)))

def train(dataset):
    input_image, reconstructed_image = autoencoder(batch_shape)
    loss = calculate_loss(input_image, reconstructed_image)
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    #f0, f1, f2, f3 = extra[0], extra[1], extra[2], extra[3]
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(init)

        dataset_size = len(dataset.train.images)
        print("Dataset size:", dataset_size)
        num_iters = (num_epochs * dataset_size)//batch_size
        print("Num iters:", num_iters)
        for step in range(num_iters):
            input_batch  = get_next_batch(dataset.train, batch_size)
            loss_val,  _= session.run([loss, optimizer],
                                       feed_dict={input_image: input_batch})

            #print('f0_', np.sqrt(np.sum(f0_**2)))
            if step % 1000 == 0:
                #print('f0_', f0_)
                #print('f1_', f1_)
                #print('f2_', f2_)
                #print('f3_', f3_)
                #print('re_', re_)
                print("Loss at step", step, ":", loss_val)

        test_batch = get_next_batch(dataset.test, batch_size)
        reconstruction = session.run(reconstructed_image,
                                     feed_dict={input_image: test_batch})
        visualize(test_batch, reconstruction, num_visualize)

if __name__ == '__main__':
    dataset = load_dataset()
    train(dataset)
    
