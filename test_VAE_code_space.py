import os
import sys
import getopt

import numpy as np
import tensorflow as tf

import VAE
from read_MNIST import MNIST_IMG_X
from read_MNIST import MNIST_IMG_Y
from read_MNIST import load_mnist
from read_MNIST import BuildPipeline


############### global configs ################
n_hidden = 512
z_dim = 2

# tensorboard summary will be saved as summary_path/summary_name
summary_path = './tensorboard/'
summary_name = 'summary-test'
# current optimal model checkpoint will be restored from model_path
model_path = './ckpts/'

############### build the VAE decoder pipeline ################
z_plh = tf.placeholder(tf.float32, shape=(None, z_dim))

# decode
_, _, Xr = VAE.Decoder_FC(z_plh, n_hidden, z_dim, MNIST_IMG_X*MNIST_IMG_Y)

# add summary hooks here
with tf.name_scope('summary'):
	tf.summary.image(name='reconstruction', tensor=tf.dtypes.cast(Xr,tf.uint8))



# build the test process
def test(sess, z_min, z_max, samples, summary_op, summary_writer):
	'''
	input:	sess : tf session to run the test
			z_min: min of each code dimension
			z_max: max of each code dimension
			samples: num of samples along each code dimension
			summary_op: summary operation to do
			summary_writer : summary writer
	'''
	# generate code Z as numpy array
	z1 = np.linspace(z_min, z_max, samples)
	z2 = z1
	Z1, Z2 = np.meshgrid(z1, z2)
	z = np.vstack((Z1.ravel(),Z2.ravel())).transpose()
	# feed the Z code to the decoder and get the reconstructed images
	Xr_, summary_buff_ = sess.run([Xr, summary_op], feed_dict={z_plh: z})
	# summary
	summary_writer.add_summary(summary=summary_buff_)
	# over
	return

	
	

###################### main entrance ######################
if __name__ == "__main__":
	# set tensorboard summary path
	try:
		options, args = getopt.getopt(sys.argv[1:], '', ['logdir='])
	except getopt.GetoptError:
		print('invalid arguments!')
		sys.exit(-1)
	for option, value in options:
		if option == '--logdir':
			summary_name = value

	with tf.Session() as sess:
		# restore the decoder part
		decoder_saver = tf.train.Saver(model_path + 'decoder.ckpt')
		# build the tensorboard summary
		summary_writer = tf.summary.FileWriter(summary_path+summary_name)
		summary_op = tf.summary.merge_all()

		# test
		test(sess, -10.0 10.0, 10, summary_op, summary_writer)
		
	# finished
	print('++++++++++++++++++++++++++++++++++++++++')
