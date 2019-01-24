#Arpan Bag

#Helper file that contains all the required user defined functions for Linear Regression



## Initialization
import numpy as np
import tensorflow as tf
import os



#=================== Feature Normalization =========================

#FEATURENORMALIZE Normalizes the features in X 
#   FEATURENORMALIZE(X) returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation
#   is 1. This is often a good preprocessing step to do when
#   working with learning algorithms.

def featureNormalize(X):
	X_norm = X
	mu = np.zeros((1, X.shape[1]))
	sigma = np.zeros((1, X.shape[1]))
	
	# =================================CODE HERE ======================
	#Instructions: First, for each feature dimension, compute the mean
    #           of the feature and subtract it from the dataset,
    #           storing the mean value in mu. Next, compute the 
    #           standard deviation of each feature and divide
    #           each feature by it's standard deviation, storing
    #           the standard deviation in sigma. 
	#
    #           Note that X is a matrix where each column is a 
    #           feature and each row is an example. You need 
    #           to perform the normalization separately for 
    #           each feature. 
	
	mu = np.mean(X,axis=0)
	sigma = np.std(X,ddof=1,axis=0)
	X_norm = (X_norm - mu)/sigma
	
	return X_norm, mu, sigma
	

	
	
	
	
	
	
	
#======================= Gradient Descent =========================	
#GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(x, y, theta, learning_rate, num_iters) updates theta by
#   taking num_iters gradient steps with specified learning rate
	
def gradientDescent(X, Y, learning_rate, num_iters):

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	#Hide tensorflow log	

	#Initialize some useful values
	m = len(Y) # number of training examples
	J_history = np.zeros((num_iters, 1))
	
	Tf_X = tf.placeholder(tf.float32,X.shape)
	Tf_Y = tf.placeholder(tf.float32,Y.shape)
	
	Tf_theta = tf.Variable(tf.ones([X.shape[1],1]))

	Y_ = tf.matmul(Tf_X,Tf_theta)
	
	cost = 1/(2*m)*tf.reduce_sum(tf.square(Y_ - Y))
	
	training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	
	init = tf.global_variables_initializer()
	
	sess = tf.Session()
	
	sess.run(init)
	
	for iter in range(num_iters):

		# =========================== CODE HERE ==============================
		
		sess.run(training_step,feed_dict={Tf_X:X,Tf_Y:Y})

		# ============================================================

		# Save the cost J in every iteration    
		J_history[iter] = sess.run(cost,feed_dict={Tf_X:X,Tf_Y:Y})

	theta = sess.run(Tf_theta)	
	
	return theta, J_history	
	
	
	


