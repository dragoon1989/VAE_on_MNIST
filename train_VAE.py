import tensorflow as tf

import VAE


############### global configs ################
n_hidden = tf.placeholder(tf.int32, shape=())
z_dim = tf.placeholder(tf.int32, shape=())
keep_prob = tf.placeholder(tf.float32, shape=())

############### the input pipeline part ################



############### build the training pipeline ################
X = tf.placeholder(tf.uint8, shape=(None, IMG_X*IMG_Y))

miu, std = VAE.Encoder_FC(X=X, n_hidden=n_hidden, z_dim=zdim, keep_prob=keep_prob)

# sampling code from generated distributions
z = miu + std * tf.random.normal(shape=tf.shape(miu), mean=0.0, stddev=1.0, dtype=tf.float32)

