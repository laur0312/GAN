from __future__ import division

import os

import numpy as np
import scipy.misc
import tensorflow as tf
from six.moves import xrange


def save_images(images, image_path):
    num_samples = images.shape[0]
    in_img_height = images.shape[1]
    in_img_width = images.shape[2]
    # make sure images are gray scale
    assert images.shape[3] == 1

    # num of row and column of out_img, out_num_row <= out_num_column
    out_num_row = int(np.floor(np.sqrt(num_samples)))
    out_num_column = int(np.ceil(np.sqrt(num_samples)))

    out_img = np.zeros((in_img_height * out_num_row, in_img_width * out_num_column))
    for index, image in enumerate(images):
        i = index % out_num_column
        j = index // out_num_column
        out_img[j * in_img_height:(j + 1) * in_img_height, i * in_img_width:(i + 1) * in_img_width] = image[:, :, 0]

    scipy.misc.imsave(image_path, np.squeeze(out_img))


class CDCGAN(object):
    model_name = 'CDCGAN'

    def __init__(self, sess, batch_size=64, noise_dim=100, label_dim=10,
                 dataset_name='mnist', result_dir=None, model_dir=None):

        """
        Args:
            sess:               TensorFlow session.
            batch_size:         Size of batch.
            noise_dim:          Dimension of noise.
            dataset_name:       Name of dataset.
            result_dir:         Directory to save result.
            model_dir:          Directory to save model.
        """

        self.sess = sess
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.samples_num = label_dim
        self.result_folder = os.path.join(result_dir, self.model_name)
        self.model_folder = os.path.join(model_dir, self.model_name)

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        if dataset_name == 'mnist':
            self.img_height = 28
            self.img_width = 28
            self.img_color_dim = 1

            self.data_X, self.data_Y = self.load_mnist()
            self.batch_index = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def build_model(self):
        """ Graph Input """
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size, self.img_height, self.img_width, self.img_color_dim],
                                     name='inputs')
        self.noise = tf.placeholder(tf.float32,
                                    [None, self.noise_dim],
                                    name='noise')
        self.labels = tf.placeholder(tf.float32,
                                     [self.batch_size, self.label_dim],
                                     name='labels')

        """ Loss Function """
        G = self.generator(self.noise, self.labels, is_training=True, reuse=False)
        D_real = self.discriminator(self.inputs, self.labels, is_training=True, reuse=False)
        D_fake = self.discriminator(G, self.labels, is_training=True, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))

        # loss for discriminator
        self.d_loss = d_loss_real + d_loss_fake

        # loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

        """ Training """
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        print(d_vars)
        print(g_vars)

        # define optimizers for discriminator
        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=d_vars)

        # define optimizers for generator
        self.g_optim = tf.train.AdamOptimizer(0.0002 * 5, beta1=0.5).minimize(self.g_loss, var_list=g_vars)

        """ Testing """
        self.sample_noise = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim))
        self.fake_images = self.generator(self.noise, self.labels, is_training=False, reuse=True)

    def discriminator(self, image, label, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            label_conv = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])

            # Add label as condition
            image = self.conv_concat(image, label_conv)

            # Hidden Layer
            net = self.lrelu(
                self.batch_norm(
                    self.conv2d(image, 64 + self.label_dim, scope='d_conv_1'),
                    is_training=is_training, scope='d_batch_nom_1'))

            # Add label as condition
            # net = self.conv_concat(net, label_conv)

            # Hidden Layer
            net = self.lrelu(
                self.batch_norm(
                    self.conv2d(net, 128 + self.label_dim, scope='d_conv_2'),
                    is_training=is_training, scope='d_batch_nom_2'))

            net = tf.reshape(net, [self.batch_size, -1])

            # Add label as condition
            # net = self.concat([net, label], 1)

            # Hidden Layer
            net = self.lrelu(
                self.batch_norm(
                    self.linear(net, 1024, scope='d_linear_3'),
                    is_training=is_training, scope='d_batch_norm_3'))

            # Add label as condition
            # net = self.concat([net, label], 1)

            # Output Layer
            out = self.linear(net, 1, scope='d_linear_4')

            return out

    def generator(self, noise, label, is_training=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            label_conv = tf.reshape(label, [self.batch_size, 1, 1, self.label_dim])

            # Add label as condition
            noise = self.concat([noise, label], 1)

            # Hidden Layer
            net = tf.nn.relu(
                self.batch_norm(
                    self.linear(noise, 1024, scope='g_linear_1'),
                    is_training=is_training, scope='g_batch_norm_1'))

            # Add label as condition
            # net = self.concat([net, label], 1)

            # Hidden Layer
            net = tf.nn.relu(
                self.batch_norm(
                    self.linear(net, 7 * 7 * 128, scope='g_linear_2'),
                    is_training=is_training, scope='g_batch_norm_2'))

            net = tf.reshape(net, [-1, 7, 7, 128])

            # Add label as condition
            # net = self.conv_concat(net, label_conv)

            # Hidden Layer
            net = tf.nn.relu(
                self.batch_norm(
                    self.deconv2d(net, [self.batch_size, 14, 14, 64], scope='g_deconv_3'),
                    is_training=is_training, scope='g_batch_norm_3'))

            # Add label as condition
            # net = self.conv_concat(net, label_conv)

            # Output Layer
            out = tf.nn.sigmoid(
                self.deconv2d(net, [self.batch_size, 28, 28, 1], scope='g_deconv_4'))

            return out

    def train(self, epochs):
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        for epoch in xrange(epochs):

            for index in xrange(0, self.batch_index):
                batch_inputs = self.data_X[index * self.batch_size: (index + 1) * self.batch_size]
                batch_noise = np.random.uniform(-1, 1, size=(self.batch_size, self.noise_dim)).astype(np.float32)
                batch_labels = self.data_Y[index * self.batch_size: (index + 1) * self.batch_size]

                # train discriminator
                _, d_loss = self.sess.run([self.d_optim, self.d_loss],
                                          feed_dict={
                                              self.inputs: batch_inputs,
                                              self.noise: batch_noise,
                                              self.labels: batch_labels})

                # train generator
                _, g_loss = self.sess.run([self.g_optim, self.g_loss],
                                          feed_dict={self.noise: batch_noise,
                                                     self.labels: batch_labels})

                print('Epoch: {:02d}; index: {:04d}, d_loss: {:.8}; g_loss: {:.8}'
                      .format(epoch, index, d_loss, g_loss))

            self.test(epoch)

        saver.save(self.sess, os.path.join(self.model_folder, self.model_name + '.model'), global_step=epochs)

    def test(self, epoch):
        seed = 111
        np.random.seed(seed)
        selected_samples = np.random.choice(self.batch_size, self.samples_num)

        for index in xrange(self.label_dim):
            label = np.zeros(self.batch_size, dtype=np.int64) + index
            label_one_hot = np.zeros((self.batch_size, self.label_dim))
            label_one_hot[np.arange(self.batch_size), label] = 1

            samples = self.sess.run(self.fake_images,
                                    feed_dict={self.noise: self.sample_noise,
                                               self.labels: label_one_hot})

            samples = samples[selected_samples, :, :, :]

            if index == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        save_images(all_samples, './{}/{:02d}.png'.format(self.result_folder, epoch))

    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def batch_norm(self, x, is_training, scope='batch_norm'):
        return tf.contrib.layers.batch_norm(x,
                                            decay=0.9,
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            is_training=is_training,
                                            scope=scope)

    def linear(self, input_, output_size, stddev=0.02, bias_init=0.0, scope='linear'):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope):
            w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable('bias', [output_size],
                                   initializer=tf.constant_initializer(bias_init))

            return tf.matmul(input_, w) + bias

    def conv2d(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, bias_init=0.0, scope='conv'):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            bias = tf.get_variable('bias', [output_dim],
                                   initializer=tf.constant_initializer(bias_init))

            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
            conv = tf.reshape(tf.nn.bias_add(conv, bias), conv.get_shape())

            return conv

    def deconv2d(self, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, bias_init=0.0, scope='deconv'):
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable('bias', [output_shape[-1]],
                                   initializer=tf.constant_initializer(bias_init))

            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
            deconv = tf.reshape(tf.nn.bias_add(deconv, bias), deconv.get_shape())

            return deconv

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def conv_concat(self, x, y):
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return self.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def load_mnist(self):
        data_dir = os.path.join('./data', 'mnist')

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_y = loaded[8:].reshape(60000).astype(np.int)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_x = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_y = loaded[8:].reshape(10000).astype(np.int)

        x = np.concatenate((train_x, test_x), axis=0)
        y = np.concatenate((train_y, test_y), axis=0)

        seed = 111
        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)

        x_out = x / 255.

        y_out = np.zeros((len(y), self.label_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_out[i, y[i]] = 1.0

        return x_out, y_out
