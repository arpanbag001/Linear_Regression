##Arpan Bag
##Linear Regression

## Initialization
import numpy as np
import os
import pdb






print("*****Linear Regression*******\n\n\n")
input("Press enter to select the Input data file.\n")



## Load Data


print('Loading data ...\n')
data = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + "\\" +"Sample_Data.txt",delimiter=",") #The file containing the training data. The Last column is the output (Y) and the rest of the columns are input (X) 

X = data[:,:-1] #Inputs
Y = data[:,-1]	 #Outputs
m = len(Y);		 #Number of training examples
num_features = X.shape[1]	 #No of features, which is the dimension of X

#Print out some data points
print('\nData loaded.\nFirst 10 examples from the dataset: \n')
print(np.column_stack((X,Y))[0:10,:])
input('\nPress enter to start Feature Normalization.\n')




## ================ Part 1: Feature Normalization ================