import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def discriminator(x,c):
    with tf.name_scope('discriminator'):
        W_fc1 = weight_variable([x.shape[1]+c.shape[1], 128])
        b_fc1 = bias_variable([128])
        W_fc2 = weight_variable([128, 1])
        b_fc2 = bias_variable([1])
        x_context = tf.concat([x,c],1)
        h_fc1 = tf.nn.relu(tf.matmul(x_context,W_fc1) + b_fc1)
        logits = tf.matmul(h_fc1,W_fc2) + b_fc2
        prob = tf.nn.sigmoid(logits)
        return prob,logits

def generator(z,c):
    with tf.name_scope('generator'):
        W_fc1 = weight_variable([z.shape[1]+c.shape[1], 128])
        b_fc1 = bias_variable([128])
        W_fc2 = weight_variable([128,784])
        b_fc2 = bias_variable([784])
        z_context = tf.concat([z,c],1)
        h_fc1 = tf.nn.relu(tf.matmul(z_context,W_fc1) + b_fc1)
        h_fc2 = tf.matmul(h_fc1,W_fc2) + b_fc2
        prob = tf.nn.sigmoid(h_fc2)
        return prob

def weight_variable(shape):
    initial = tf.truncated_normal([int(shape[0]),int(shape[1])], stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for j, sample in enumerate(samples):
        ax = plt.subplot(gs[j])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    c = tf.placeholder(tf.float32, [None, 10])
    z = tf.placeholder(tf.float32, [None, 100])
    g = generator(z,c)
    d_real,d_logits_real = discriminator(x,c)
    d_fake,d_logits_fake = discriminator(g,c)
    if not os.path.exists('gan_iamges/'):
        os.makedirs('gan_images/')

    with tf.name_scope('loss'):
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))
        d_loss = d_loss_real + d_loss_fake
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

    train_var = tf.trainable_variables()
    d_var = [var for var in train_var if 'discriminator' in var.name]
    g_var = [var for var in train_var if 'generator' in var.name]

    with tf.name_scope('adam_optimizer'):
        d_train = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_var)
        g_train = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_var)

    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         j=0
         for i in range(1000000):
             batch_x, batch_c = mnist.train.next_batch(64)
             batch_z = np.random.uniform(-1., 1., [64, 100])
             _, d_loss_curr = sess.run([d_train, d_loss], feed_dict={x: batch_x, z: batch_z, c:batch_c})
             _, g_loss_curr = sess.run([g_train, g_loss],feed_dict={z: batch_z, c:batch_c})

             if i % 1000 == 0:
                 print('Iteration: {}'.format(i))
                 print('discriminator loss: {:.4}'.format(d_loss_curr))
                 print('Generator loss: {:.4}'.format(g_loss_curr))
                 print()

                 num_sample = 16
                 z_sample = np.random.uniform(-1., 1., [num_sample, 100])
                 c_sample = np.zeros(shape=[num_sample,10])
                 c_sample[:,9] = 1
                 images = sess.run(g, feed_dict={z: z_sample, c:c_sample})
                 fig = plot(images)
                 plt.savefig('gan_images/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                 plt.close(fig)
                 j+=1

if __name__ == '__main__': main()
