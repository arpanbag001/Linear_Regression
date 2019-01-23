#Arpan Bag

#Helper file that contains all the required user defined functions for Linear Regression



## Initialization
import numpy as np
import pdb


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
#   theta = GRADIENTDESCENT(x, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha
	
def gradientDescent(X, Y, theta, alpha, num_iters):	
	#Initialize some useful values
	m = len(Y) # number of training examples
	J_history = np.zeros((num_iters, 1))
	
	for iter in range(num_iters):

		# =========================== CODE HERE ==============================
		# Instructions: Perform a single gradient step on the parameter vector
		#               theta. 
    
		# Hint: While debugging, it can be useful to print out the values
		#       of the cost function (computeCost) and gradient here.
		#

		derivativePart = 1/m*np.sum((np.dot(X,theta) - Y)*X,axis=0,keepdims=True)
		
		theta = theta - (alpha*derivativePart.T)

		

		# ============================================================

		# Save the cost J in every iteration    
		J_history[iter] = computeCost(X, Y, theta)

	return theta, J_history	
	
	
	
	

	
	

	
#======================= Cost Function =========================	
#COMPUTECOST Compute cost for linear regression with multiple variables
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
	
	
def computeCost(X, Y, theta):
	#Initialize some useful values
	m = len(Y) #Number of training examples
	
	J = 0	#Cost

	# =========================== CODE HERE ========================
	# Instructions: Compute the cost of a particular choice of theta
	#               Should set J to the cost.

	
	J = 1/(2*m)*np.sum(np.square(np.dot(X,theta) - Y))

	return J

























	
	
	
	
	
	
	
	
	
       
