import tensorflow as tf
import tensorflow.keras as keras


# FC encoder part
def Encoder_FC(X, n_hidden, z_dim, keep_prob):
	''' input:	X --- input MNIST image as a 1D vector (dtype=tf.uint8)
				n_hidden --- FC layer hidden units
				z_dim --- output code dimensions
				keep_prob --- keep prob for dropout
		output:	miu --- the sampling distribution's mean as 1D vector (dtype=tf.float32) 
				std --- the sampling distribution's sigma as 1D vector (dtype=tf.float32) '''
	# build 1st FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 activation='relu')(X)
	# dropout
	feature = keras.layers.Dropout(1-keep_prob)(feature)
	# build 2nd FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 activation='relu')(feature)
	# dropout
	feature = keras.layers.Dropout(1-keep_prob)(feature)
	# output layer
	output = keras.layers.Dense(units=2*z_dim, 
								activation=None)(feature)
	# mean part (miu)
	miu = tf.to_float(output[:, 0:z_dim])
	# sigma part (std), must >= 0
	std = output[:, z_dim:]
	std = tf.to_float(tf.math.softplus(std)) + 1e-6
	# over
	return miu, std

# FC decoder
def Decoder_FC(z, n_hidden, img_size, keep_prob):
	''' input:	z --- the input code as 1D vector (dtype=tf.float32)
				n_hidden --- FC layer hidden units
				img_size --- length of MNIST image as 1D vector 
				keep_prob --- keep prob for dropout
		output:	Xr --- the reconstructed MNIST image as 1D vector (dtype=tf.float32) '''
	# get code dimensions
	z_dim = z.get_shape()[1]
	# build 1st FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 activation='relu')(z)
	# dropout
	feature = keras.layers.Dropout(1-keep_prob)(feature)
	# build 2nd FC layer
	feature = keras.layers.Dense(units=n_hidden,
								 activation='relu')(feature)
	# dropout
	feature = keras.layers.Dropout(1-keep_prob)(feature)
	# output layer
	Xr = keras.layers.Dense(units=img_size, activation='softmax')(feature)
	Xr = Xr * 256
	# over
	return Xr

# compute the reconstruction loss part
def ReconLoss(X, Xr):
	''' input:	X --- input MNIST image as a 1D vector (dtype=tf.uint8)
				Xr --- the reconstructed MNIST image as 1D vector (dtype=tf.float32)
		output:	loss --- output loss '''
	# we use cross-entropy as loss metric
	batch_loss = keras.losses.sparse_categorical_crossentropy(y_true=tf.to_float(X), y_pred=Xr)
	# over
	return tf.reduce_mean(batch_loss)


	