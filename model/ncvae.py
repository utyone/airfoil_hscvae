import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import numpy as np
import tensorflow_probability as tfp

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

def full_connected(x, weight_shape, initializer):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)


def reconstruction_loss(original, reconstruction, eps=1e-10):
    """
    The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution
    induced by the decoder in the data space). This can be interpreted as the number of "nats" required for
    reconstructing the input when the activation in latent is given.
    Adding 1e-10 to avoid evaluation of log(0.0)
    """
    _tmp = original * tf.log(eps + reconstruction) + (1 - original) * tf.log(eps + 1 - reconstruction)
    return -tf.reduce_sum(_tmp, 1)

def reconstruction_loss_shape(original, reconstruction, eps=1e-10):
    """
    The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution
    induced by the decoder in the data space). This can be interpreted as the number of "nats" required for
    reconstructing the input when the activation in latent is given.
    Adding 1e-10 to avoid evaluation of log(0.0)
    """
    return tf.compat.v1.losses.mean_squared_error(original, reconstruction)

def latent_loss(latent_mean, latent_log_sigma_sq):
    """
    The latent loss, which is defined as the Kullback Leibler divergence between the distribution in latent space
    induced by the encoder on the data and some prior. This acts as a kind of regularizer. This can be interpreted as
    the number of "nats" required for transmitting the the latent space distribution given the prior.
    """
    latent_log_sigma_sq = tf.clip_by_value(latent_log_sigma_sq, clip_value_min=-1e-10, clip_value_max=1e+2)
    return -0.5 * tf.reduce_sum(1 + latent_log_sigma_sq - tf.square(latent_mean) - tf.exp(latent_log_sigma_sq), axis=1)


class ConditionalVAE(object):
    """ Conditional VAE
    - Encoder: input (2d vector) -> FC x 3 -> latent
    - Decoder: latent -> FC x 3 -> output (2d vector)
    """

    def __init__(self, label_size, network_architecture, activation=tf.nn.sigmoid, learning_rate=0.001,
                 batch_size=100, save_path=None, load_model=None, max_grad_norm=1, latent=4, ini=True):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input
            n_z: dimensionality of latent space
        :param float learning_rate: learning rate
        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param int batch_size:
        """
        self.network_architecture = network_architecture
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.label_size = label_size
        self.max_grad_norm = max_grad_norm

        # Initializer
        if "relu" in self.activation.__name__:
            self.ini = variance_scaling_initializer()
        else:
            self.ini = xavier_initializer()

        # Create network
        self._create_network(latent,ini=True, )

        # Summary
        tf.compat.v1.summary.scalar("loss", self.loss)
        # Launch the session
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False))
        # Summary writer for tensor board
        self.summary = tf.compat.v1.summary.merge_all()
        if save_path:
            self.writer = tf.compat.v1.summary.FileWriter(save_path, self.sess.graph)
        # Load model
        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.sess, load_model)

    def _create_network(self, latent, ini=False):
        """ Create Network, Define Loss Function and Optimizer """

        dropout_ratio = 0.2
        
        # tf Graph input
        if(ini):
            self.x = tf.compat.v1.placeholder(tf.float32, [None, self.network_architecture["n_input"]], name="input")
            self.y = tf.compat.v1.placeholder(tf.float32, [None, self.label_size], name="output")

        # Build conditional input
        _layer = tf.concat([self.x, self.y], axis=1)

        # Encoder network to determine mean and (log) variance of Gaussian distribution in latent space
        with tf.compat.v1.variable_scope("encoder"):
            # full connected 1
            _layer = full_connected(_layer, [self.network_architecture["n_input"] + self.label_size,
                                             self.network_architecture["n_hidden_encoder_1"]], self.ini)
            _layer = self.activation(_layer)
            _layer = tf.nn.dropout(_layer, rate=dropout_ratio)
            # full connected 2
            _layer = full_connected(_layer, [self.network_architecture["n_hidden_encoder_1"],
                                             self.network_architecture["n_hidden_encoder_2"]], self.ini)
            _layer = self.activation(_layer)
            _layer = tf.nn.dropout(_layer, rate=dropout_ratio)
            self.z_mean = tf.layers.dense(_layer, units=self.network_architecture["n_z"], activation=lambda x: tf.nn.l2_normalize(x, axis=-1))
            ## N-VAE
            self.z_log_sigma_sq = full_connected(_layer, [self.network_architecture["n_hidden_encoder_2"],
                                                          self.network_architecture["n_z"]], self.ini)
            self.z_var = self.z_log_sigma_sq
        # Draw one sample z from Gaussian distribution
        eps = tf.compat.v1.random_normal((self.batch_size, self.network_architecture["n_z"]), mean=0, stddev=1, dtype=tf.float32)
        ## z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_var)), eps))

        # print(self.z.shape)
        _layer = tf.concat([self.z, self.y], axis=1)

        # Decoder to determine mean of Bernoulli distribution of reconstructed input
        with tf.variable_scope("decoder"):
            # full connected 1
            _layer = full_connected(_layer, [self.network_architecture["n_z"] + self.label_size,
                                             self.network_architecture["n_hidden_decoder_1"]], self.ini)
            _layer = self.activation(_layer)
            _layer = tf.nn.dropout(_layer, rate=dropout_ratio)
            # full connected 2
            _layer = full_connected(_layer, [self.network_architecture["n_hidden_decoder_1"],
                                             self.network_architecture["n_hidden_decoder_2"]], self.ini)
            _layer = self.activation(_layer)
            _layer = tf.nn.dropout(_layer, rate=dropout_ratio)
            # full connected 3 to output
            #_logit = full_connected(_layer, [self.network_architecture["n_hidden_decoder_2"],
            #                                 self.network_architecture["n_input"]], self.ini, activation=tf.nn.sigmoid())
            _logit = full_connected(_layer, [self.network_architecture["n_hidden_decoder_2"],
                                             self.network_architecture["n_input"]], self.ini)
            #self.x_decoder_mean = tf.nn.sigmoid(_logit)
            self.x_decoder_mean = tf.nn.tanh(_logit)

        # Define loss function
        with tf.name_scope('loss'):
            self.re_loss = tf.reduce_mean(reconstruction_loss_shape(original=self.x, reconstruction=self.x_decoder_mean))
            
            ## N-VAE
            self.latent_loss = tf.reduce_mean(latent_loss(self.z_mean, self.z_log_sigma_sq))
            self.loss = tf.where(tf.math.is_nan(self.re_loss), 0.0, self.re_loss)  + \
               1.*tf.where(tf.math.is_nan(self.latent_loss), 0.0, self.latent_loss)

        # Define optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        if self.max_grad_norm:
            _var = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, _var), self.max_grad_norm)
            self.train = optimizer.apply_gradients(zip(grads, _var))
        else:
            self.train = optimizer.minimize(self.loss)
        # saver
        self.saver = tf.compat.v1.train.Saver()

    def reconstruct(self, inputs, label):
        """Reconstruct given data. """
        assert len(inputs) == self.batch_size
        assert len(label) == self.batch_size
        return self.sess.run(self.x_decoder_mean, feed_dict={self.x: inputs, self.y: label})

    def encode(self, inputs, label):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z, feed_dict={self.x: inputs, self.y: label})

    def encode2(self, inputs, label):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})

    def encode2s(self, inputs, label):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z_log_sigma_sq, feed_dict={self.x: inputs, self.y: label})

    def encode3(self, inputs, label):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z_var, feed_dict={self.x: inputs, self.y: label})

    def decode(self, label, z=None, std=0.01, mu=0):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.
        """
        return self.sess.run(self.x_decoder_mean, feed_dict={self.z: z, self.y: label})


if __name__ == '__main__':
    import os

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    ConditionalVAE(10)
