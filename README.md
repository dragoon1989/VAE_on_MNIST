# VAE_on_MNIST
study VAE on MNIST dataset

# 1. use fc layers as encoder and decoder
encoder --- fc1->activation->dropout->
		  fc2->activation->dropout->
		  generate miu's and sigma's

decoder --- fc1->activation->dropout->
		  fc2->activation->dropout->
		  generate output
