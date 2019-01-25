#Arpan Bag

#Helper file that contains all the required user defined functions for Linear Regression



## Initialization
import numpy as np
import tensorflow as tf
import os

	
	
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
	
	
	


