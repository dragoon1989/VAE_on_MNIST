import tensorflow as tf
import tensorflow.keras as keras


# FC encoder part
def Encoder_FC(X, n_hidden, z_dim):
	''' input:	X --- input MNIST image as batch of 1D vectors (shape=(B,L), dtype=tf.int32)
				n_hidden --- FC layer hidden units
				z_dim --- output code dimensions
		output:	miu --- the sampling distribution's mean as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32) 
				std --- the sampling distribution's sigma as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32) '''
	# create a scope for encoder variable nodes
	# we will add these variables to a collection for save and restore
	with tf.variable_scope('encoder-fc'):
		# first convet X to dtype=tf.float32 ?
		X = tf.to_float(X)
		# build 1st FC layer
		fc1 = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')
		# save the variables
		tf.add_to_collection('encoder_var', fc1.variables)
		# apply the layer
		feature = fc1.apply(X)
		
		# build 2nd FC layer
		fc2 = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')
		# save the variables
		tf.add_to_collection('encoder_var', fc2.variables)
		# apply the layer
		feature = fc2.apply(feature)
		
		# build the output layer
		out_layer = keras.layers.Dense(units=2*z_dim, 
									   kernel_initializer='he_normal', 
									   activation=None)
		# save the variables
		tf.add_to_collection('encoder_var', out_layer.variables)
		# apply the layer
		output = out_layer.apply(feature)
		
		# mean part (miu)
		miu = output[:, 0:z_dim]
		# sigma part (std), must >= 0
		std = output[:, z_dim:]
		std = tf.math.softplus(std) + 1e-6
		# over
		return tf.to_float(miu), tf.to_float(std)
		
# CNN encoder part
def Encoder_CNN(X, z_dim):
	''' input:	X --- input MNIST image as batch of 2D gray images (shape=(B,Ny,Nx), dtype=tf.uint8)
				z_dim --- output code dimensions
		output:	miu --- the sampling distribution's mean as batch of 2D images (shape=(B,Ny/4, Nx/4, z_dim), dtype=tf.float32) 
				std --- the sampling distribution's sigma as batch of 2D images (B,Ny/4, Nx/4, z_dim), dtype=tf.float32) '''
	# create a scope for encoder variable nodes
	# we will add these variables to a collection for save and restore
	with tf.variable_scope('encoder-cnn'):
		w_init = tf.initializers.he_normal()
		b_init = tf.initializers.constant(0.0)
		# first convet X to dtype=tf.float32
		X = tf.to_float(X)
		# rescale X to [-0.5, 0.5]
		X = (X - 128)/256
		# reshape X to BHWC format
		X = tf.reshape(X, shape=(tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2], 1))
		# build 1st conv layer (3x3x32)
		conv1_w = tf.get_variable(name='conv1_w',
								  shape=(3,3,1,32),
								  dtype=tf.float32, 
								  initializer=w_init,
								  trainable=True)
		conv1_b = tf.get_variable(name='conv1_b',
								  shape=(1,1,32),
								  dtype=tf.float32,
								  initializer=b_init,
								  trainable=True)
		# apply the conv layer
		feature = tf.nn.conv2d(input=X,
							   filter=conv1_w,
							   strides=1,
							   padding='SAME')
		feature = tf.relu(feature + conv1_b)
		# save the variables
		tf.add_to_collection('encoder_var', conv1_w)
		tf.add_to_collection('encoder_var', conv1_b)
		
		# 1st maxpool
		feature = tf.nn.maxpool(value=feature,
								ksize=2,
								strides=2,
								padding='SAME')
		
		# build 2nd conv layer (3x3x64)
		conv2_w = tf.get_variable(name='conv2_w',
								  shape=(3,3,32,64),
								  dtype=tf.float32, 
								  initializer=w_init,
								  trainable=True)
		conv2_b = tf.get_variable(name='conv2_b',
								  shape=(1,1,64),
								  dtype=tf.float32,
								  initializer=b_init,
								  trainable=True)
		# apply the conv layer
		feature = tf.nn.conv2d(input=feature,
							   filter=conv2_w,
							   strides=1,
							   padding='SAME')
		feature = tf.relu(feature + conv2_b)
		# save the variables
		tf.add_to_collection('encoder_var', conv2_w)
		tf.add_to_collection('encoder_var', conv2_b)
		
		# 2nd maxpool
		feature = tf.nn.maxpool(value=feature,
								ksize=2,
								strides=2,
								padding='SAME')

		# build the output layer (1 x 1 x (2*z_dim) conv)
		out_w = tf.get_variable(name='out_w',
								shape=(1,1,64,2*z_dim),
								dtype=tf.float32, 
								initializer=w_init,
								trainable=True)
		out_b = tf.get_variable(name='out_b',
								shape=(1,1,2*z_dim),
								dtype=tf.float32,
								initializer=b_init,
								trainable=True)
		# apply the output layer
		feature = tf.nn.conv2d(input=feature,
							   filter=out_w,
							   strides=1,
							   padding='SAME')
		output = feature + out_b
		# save the variables
		tf.add_to_collection('encoder_var', out_w)
		tf.add_to_collection('encoder_var', out_b)

		# get mean part (miu)
		miu = output[:,:,:, 0:z_dim]
		# sigma part (std), must >= 0
		std = output[:,:,:, z_dim:]
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
	# create a scope for decoder variable nodes
	# we will add these variables to a collection for save and restore
	with tf.variable_scope('decoder-fc'):
		# build 1st FC layer
		fc1 = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')
		# save the layer
		tf.add_to_collection('decoder_var', fc1.variables)
		# apply the layer
		feature = fc1.apply(z)
		# build 2nd FC layer
		fc2 = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')
		# save the layer
		tf.add_to_collection('decoder_var', fc2.variables)
		# apply the layer
		feature = fc2.apply(feature)
		
		# output layer
		out_layer = keras.layers.Dense(units=img_size*256,
									   kernel_initializer='he_normal', 
									   activation=None)
		# save the layer
		tf.add_to_collection('decoder_var', out_layer.variables)
		# apply the layer
		feature = out_layer.apply(feature)
		
		# reshape the output to shape=(B, L, 256)
		logits_before_softmax = tf.reshape(feature, shape=(tf.shape(feature)[0], img_size, 256))
		# compute the pixelwise likelihood predictions
		likelihood = tf.nn.softmax(logits_before_softmax, axis=-1)
		# reconstruct the image
		Xr = tf.math.argmax(likelihood, axis=-1, output_type=tf.int32)
		# over
		return logits_before_softmax, likelihood, Xr

# Bernoulli Distribution Decoder
def Decoder_FC_B(z, n_hidden, z_dim, img_size):
	''' input:	z --- the input code as batch of 1D vectors (shape=(B,z_dim), dtype=tf.float32)
				n_hidden --- FC layer hidden units
				z_dim --- code dimensions
				img_size --- length of MNIST image, L 
		output:	likelihood --- pixelwise likelihood of reconstruction (shape=(B,L), dtype=tf.float32)
				Xr --- the reconstructed MNIST image as batch of 1D vectors (shape=(B,L), dtype=tf.uint8) '''
	# create a scope for decoder variable nodes
	# we will add these variables to a collection for save and restore
	with tf.variable_scope('decoder-fc-bi'):
		# build 1st FC layer
		fc1 = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')
		# save the layer
		tf.add_to_collection('decoder_var', fc1.variables)
		# apply the layer
		feature = fc1.apply(z)
		# build 2nd FC layer
		fc2 = keras.layers.Dense(units=n_hidden,
								 kernel_initializer='he_normal', 
								 activation='relu')
		# save the layer
		tf.add_to_collection('decoder_var', fc2.variables)
		# apply the layer
		feature = fc2.apply(feature)
		
		# output layer
		out_layer = keras.layers.Dense(units=img_size,
									   kernel_initializer='he_normal', 
									   activation=None)
		# save the layer
		tf.add_to_collection('decoder_var', out_layer.variables)
		# apply the layer
		feature = out_layer.apply(feature)

		# compute the pixelwise likelihood predictions (use sigmoid)
		likelihood = tf.math.sigmoid(feature)
		# reconstruct the image
		Xr = tf.cast(likelihood*256, dtype=tf.uint8)
		# over
		return likelihood, Xr

# CNN decoder
def Decoder_CNN_B(z, z_dim, img_size):
	''' input:	z --- the input code as batch of 2D images (shape=(B,Ny,Nx,z_dim), dtype=tf.float32)
				z_dim --- code dimensions
				img_size --- size of MNIST image (img_size x img_size)
		output:	likelihood --- pixelwise likelihood of reconstruction (shape=(B,img_size, img_size), dtype=tf.float32)
				Xr --- the reconstructed MNIST image as batch of 1D vectors (shape=(B,img_size, img_size), dtype=tf.uint8) '''
	# create a scope for decoder variable nodes
	# we will add these variables to a collection for save and restore
	with tf.variable_scope('decoder-cnn'):
		w_init = tf.initializers.he_normal()
		b_init = tf.initializers.constant(0.0)
		# build 1st deconv layer
		deconv1_w = tf.get_variable(name='deconv1_w',
									shape=(2,2,z_dim,z_dim),
									dtype=tf.float32, 
									initializer=w_init,
									trainable=True)
		deconv1_b = tf.get_variable(name='deconv1_b',
									shape=(1,1,z_dim),
									dtype=tf.float32,
									initializer=b_init,
									trainable=True)
		# apply the deconv layer
		# the auto-determined output size is correct, no need to specify output_shape
		feature = tf.nn.conv2d_transpose(value=z,
										 filter=deconv1_w,
										 strides=2,
										 padding='VALID')
		feature = feature + deconv1_b
		# save the variables
		tf.add_to_collection('decoder_var', deconv1_w)
		tf.add_to_collection('decoder_var', deconv1_b)
		
		# build 1st conv layer (3x3x64)
		conv1_w = tf.get_variable(name='conv1_w',
								  shape=(3,3,z_dim,64),
								  dtype=tf.float32, 
								  initializer=w_init,
								  trainable=True)
		conv1_b = tf.get_variable(name='conv1_b',
								  shape=(1,1,64),
								  dtype=tf.float32,
								  initializer=b_init,
								  trainable=True)
		# apply the conv layer
		feature = tf.nn.conv2d(input=feature,
							   filter=conv1_w,
							   strides=1,
							   padding='SAME')
		feature = tf.nn.relu(feature + conv1_b)
		# save the variables
		tf.add_to_collection('decoder_var', conv1_w)
		tf.add_to_collection('decoder_var', conv1_b)
		
		# build the 2nd unpooling layer
		deconv2_w = tf.get_variable(name='deconv2_w',
									shape=(2,2,64,64),
									dtype=tf.float32, 
									initializer=w_init,
									trainable=True)
		deconv2_b = tf.get_variable(name='deconv2_b',
									shape=(1,1,64),
									dtype=tf.float32,
									initializer=b_init,
									trainable=True)
		# apply the deconv layer
		# the auto-determined output size is correct, no need to specify output_shape
		feature = tf.nn.conv2d_transpose(value=feature,
										 filter=deconv2_w,
										 strides=2,
										 padding='VALID')
		feature = feature + deconv2_b
		# save the variables
		tf.add_to_collection('decoder_var', deconv2_w)
		tf.add_to_collection('decoder_var', deconv2_b)
		
		# build 2nd conv layer (3x3x32)
		conv2_w = tf.get_variable(name='conv2_w',
								  shape=(3,3,64,32),
								  dtype=tf.float32, 
								  initializer=w_init,
								  trainable=True)
		conv2_b = tf.get_variable(name='conv2_b',
								  shape=(1,1,32),
								  dtype=tf.float32,
								  initializer=b_init,
								  trainable=True)
		# apply the conv layer
		feature = tf.nn.conv2d(input=feature,
							   filter=conv2_w,
							   strides=1,
							   padding='SAME')
		feature = tf.nn.relu(feature + conv2_b)
		# save the variables
		tf.add_to_collection('decoder_var', conv2_w)
		tf.add_to_collection('decoder_var', conv2_b)
		
		# build the output layer (conv 1x1x1)
		out_w = tf.get_variable(name='out_w',
								shape=(1,1,32,1),
								dtype=tf.float32, 
								initializer=w_init,
								trainable=True)
		out_b = tf.get_variable(name='out_b',
								shape=(1,1,1),
								dtype=tf.float32,
								initializer=b_init,
								trainable=True)
		# apply the output layer
		feature = tf.nn.conv2d(input=feature,
							   filter=out_w,
							   strides=1,
							   padding='SAME')
		feature = feature + out_b
		# save the variables
		tf.add_to_collection('decoder_var', out_w)
		tf.add_to_collection('decoder_var', out_b)

		# compute the pixelwise likelihood predictions (use sigmoid)
		likelihood = tf.math.sigmoid(feature)
		likelihood = tf.reshape(likelihood, (-1,img_size, img_size))
		# reconstruct the image
		Xr = tf.cast(likelihood*256, dtype=tf.uint8)
		# over
		return likelihood, Xr

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

# compute the reconstruction loss for Bernoulli distribution
def ReconLoss_B(X, y):
	''' input:	X --- input MNIST image as batch of 1D vectors (shape=(B,L), dtype=tf.int32)
				y --- pixelwise likelihood of reconstruction (shape=(B,L), dtype=tf.float32)
		output:	batch_loss --- batch of reconstruction loss (shape=(B,)) '''
	# rescale X to [0,1]
	X = tf.to_float(X)/256
	# compute cross entropy
	batch_loss = X*tf.math.log(y) + (1-X)*tf.math.log(1-y)
	batch_loss = -batch_loss
	# compute batch loss
	batch_loss = tf.reduce_sum(batch_loss, -1)
	# over
	return batch_loss

# compute the reconstruction loss for Bernoulli distribution
def ReconLossCNN_B(X, y):
	''' input:	X --- input MNIST image as batch of 2D gray images (shape=(B,Ny,Nx), dtype=tf.int32)
				y --- pixelwise likelihood of reconstruction (shape=(B,Ny,Nx), dtype=tf.float32)
		output:	batch_loss --- batch of reconstruction loss (shape=(B,)) '''
	# rescale X to [0,1]
	X = tf.to_float(X)/256
	# flatten inputs
	X = tf.reshape(X, (tf.shape(X)[0], -1))
	y = tf.reshape(y, (tf.shape(X)[0], -1))
	# compute cross entropy
	batch_loss = X*tf.math.log(y) + (1-X)*tf.math.log(1-y)
	batch_loss = -batch_loss
	# compute batch loss
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

def KL_CNN(miu, std):
	''' input:	miu ---the sampling distribution's mean as batch of 2D images (shape=(B,Ny,Nx,z_dim), dtype=tf.float32) 
				std ---the sampling distribution's sigma as batch of 2D images (shape=(B,Ny,Nx,z_dim), dtype=tf.float32)
		output:	kl --- KL divergence of a batch (shape=(B,) '''
	# reshape inputs
	miu = tf.reshape(miu, (tf.shape(miu)[0], -1))
	std = tf.reshape(std, (tf.shape(std)[0], -1))
	kl = 0.5*tf.reduce_sum(tf.math.square(miu) + tf.math.square(std) - tf.math.log(1e-8 + tf.math.square(std)) - 1, axis=1)
	# over
	return kl
	