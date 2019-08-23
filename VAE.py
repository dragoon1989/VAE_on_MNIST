import tensorflow as tf
import tensorflow.keras as keras


# FC encoder part
def Encoder_FC(X, n_hidden, z_dim):
	''' input:	X --- input MNIST image as batch of 1D vectors (shape=(B,L), dtype=tf.int32)
				n_hidden --- FC layer hidden units
				z_dim --- output code dimensions
		output:	miu --- the sampling distribution's mean as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32) 
				std --- the sampling distribution's sigma as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32) '''
	# first convet X to dtype=tf.float32 ?
	X = tf.to_float(X)
	# build 1st FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')(X)
	# build 2nd FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')(feature)
	# output layer
	output = keras.layers.Dense(units=2*z_dim, 
								kernel_initializer='he_normal', 
								activation=None)(feature)
	# mean part (miu)
	miu = output[:, 0:z_dim]
	# sigma part (std), must >= 0
	std = output[:, z_dim:]
	std = tf.math.softplus(std) + 1e-6
	# over
	return tf.to_float(miu), tf.to_float(std)

# FC decoder
def Decoder_FC(z, n_hidden, z_dim, img_size):
	''' input:	z --- the input code as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32)
				n_hidden --- FC layer hidden units
				z_dim --- code dimensions
				img_size --- length of MNIST image, L 
		output:	logits_before_softmax --- logits to generate pixelwise likelihood of reconstruction (shape=(B,L,256), dtype=tf.float32)
				likelihood --- pixelwise likelihood of reconstruction (shape=(B,L,256), dtype=tf.float32)
				Xr --- the reconstructed MNIST image as batch of 1D vectors (shape=(B,L), dtype=tf.int32) '''
	# build 1st FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')(z)
	# build 2nd FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')(feature)
	# output layer, here we use a 1x1 conv kernel to generate the logits
	feature = keras.layers.Dense(units=img_size*256,
								 kernel_initializer='he_normal', 
								 activation=None)(feature)
	batch_size =tf.shape(feature)[0]
	feature = tf.reshape(feature, shape=(batch_size, img_size, 1, 1))	# reshape to BHWC format
	logits_before_softmax = keras.layers.Conv2D(filters=256,
												kernel_size=(1,1),
												strides=(1,1),
												padding='SAME',
												activation=None)(feature)
	# reshape back to shape=(B, L, 256)
	logits_before_softmax = tf.reshape(logits_before_softmax, shape=(batch_size,img_size,256))
	# compute the pixelwise likelihood predictions
	likelihood = tf.nn.softmax(logits_before_softmax, axis=-1)
	# reconstruct the image
	Xr = tf.math.argmax(likelihood, axis=-1, output_type=tf.int32)
	# over
	return logits_before_softmax, likelihood, Xr

# compute the reconstruction loss
def ReconLoss(X, logits_before_softmax):
	''' input:	X --- input MNIST image as batch of 1D vectors (shape=(B,L), dtype=tf.int32)
				logits_before_softmax --- logits to generate pixelwise likelihood of reconstruction (shape=(B,L,256), dtype=tf.float32)
		output:	batch_loss --- batch of reconstruction loss (shape=(B,)) '''
	# the cross entropy describes the likelihood between 2 distributions (p and q), which is what we want to maximize here : KL(p||q) = Sigma{p*log(q/p)}
	# the 'cross entropy loss' usually used in DL platform can be expressed as :
	# H(p,q) = -Sigma{plog(q)}
	# so KL(p||q) = H(p) + H(p,q)
	# where p is our known distribution to fit (the posterior distribution of variable X), so H(p) is constant
	# that is to say, KL(p||q) and H(p,q) is just the same here
	# so we just use cross-entropy loss as loss metric, which we would like to minimize
	batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=X, 
							logits=logits_before_softmax, name='reconstruction_loss')
	batch_loss = tf.reduce_sum(batch_loss, -1)
	# over
	return batch_loss

# compute the KL divergence between prior and posterior distributions
def KL(miu, std):
	''' input:	miu ---the sampling distribution's mean as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32) 
				std ---the sampling distribution's sigma as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32)
		output:	kl --- KL divergence of a batch (shape=(B,) '''
	kl = 0.5*tf.reduce_sum(tf.math.square(miu) + tf.math.square(std) - tf.math.log(1e-8 + tf.math.square(std)) - 1, axis=1)
	# over
	return kl
	