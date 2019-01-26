#Arpan Bag

#Helper file that contains all the required user defined functions for Linear Regression



## Initialization
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import os

	
	
#======================= Gradient Descent =========================	
#	GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, Y, learning_rate, num_iters) updates theta by
#   taking num_iters gradient steps with specified learning rate	
	
	
def gradientDescent(X, Y, learning_rate, num_iters):	
	
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	#Hide tensorflow log
	
	#Initialize some useful values
	m = len(Y) # number of training examples
	
	model = Sequential()	#We are using sequential model
	model.add(Dense(units = 1))		#Add only one dense layer, without any activation function (which makes the activation function linear a = g(z) = z by default.
	model.compile(loss='mean_squared_error', optimizer=optimizers.SGD(lr=learning_rate))	#Compile the model
	
	hist = model.fit(X,Y,epochs=num_iters)	#Train the model, and save the History in a variable
	
	weights = model.get_weights()[0]	#Get weights from the trained model
	bias = model.get_weights()[1]	#Get biases from the trained model
	J_history = np.asarray(hist.history['loss'])	#Get cost history from the variable we stored the training history in
	
	return model, weights, bias, J_history
	
	
	
	
	
	
	
	
	
	

	
	
	
	
	
	
	
	
#================================= Prediction ======================================	
#Predicts the output (Y) for the given Input (X) and the Keras model	
	
def predict(model,pred_x):

	pred_y = model.predict(pred_x)

	return pred_y
	
	





	
	
	


