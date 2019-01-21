%%Arpan Bag
%%Linear Regression

%% Initialization
%% Clear and Close Figures
clear ; close all; clc


fprintf('*****Linear Regression*******\n\n\n');
fprintf('Press enter to select the Input data file.\n');
pause;

%% Load Data

%Open the file selection dialogue
[inputFileName,inputFilePath] = uigetfile({
   '*.txt','Text (*.txt)'; ...
   '*.*',  'All Files (*.*)'}, ...
   'Select the Data file');
   
fprintf('\nSelected file: %s \nPress Enter to load data.\n',inputFileName);
pause;
   
fprintf('Loading data ...\n');
data = load([inputFilePath '\' inputFileName]);	%The sample file containing the training data. The ям?rst column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house. 
X = data(:,1:end-1);	%Inputs
Y = data(:,end);		%Outputs
m = length(Y);		%Number of training examples
num_features = size(X,2); %No of features, which is the dimension of X


% Print out some data points
fprintf('\nData loaded.\nFirst 10 examples from the dataset: \n');
disp([X(1:10,:),Y(1:10,:)])
fprintf('\nPress enter to start Feature Normalization.\n');
pause;



%% ================ Part 1: Feature Normalization ================


% Scale features and set them to zero mean
fprintf('\nNormalizing Features ...\n');
[X, mu, sigma] = featureNormalize(X);		% Returned X = Feature normalized training inputs, X = (X-mu)/sigma, where mu = mean of X, sigma = standard deviation of X


fprintf('\nFeature Normalization complete. Press enter to start Gradient Descent.\n');
pause;

% Add x0 intercept term to X (As x0 = 1 for all the examples)
X = [ones(m, 1) X];



%% ================ Part 2: Gradient Descent ================

%Input alpha and number of iteration values

alpha = input("Enter learning rate \nOr \npress Enter to use default (0.01): ");
if (isempty(alpha))
	alpha = 0.01;		%Default Learning rate
end
fprintf('\nLearning rate: %g\n',alpha);

num_iters = input("\nEnter number of iterations \nOr \npress Enter to use default (500): ");
if (isempty(num_iters))
	num_iters = 500;	%Default Number if iterations
else
	num_iters = floor(num_iters);	%Convert into integer
end
fprintf('\nNumber of iterations: %d\n',num_iters);





fprintf('\nRunning gradient descent ...\n');	

% Init Theta and Run Gradient Descent 
theta = zeros(num_features+1, 1);
[theta, J_history] = gradientDescent(X, Y, theta, alpha, num_iters);

fprintf('\nGradient Descent complete. Press enter to display results.\n');
pause;

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('\nTheta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


%% ======================== Prediction ===========================


% Estimate the price of a 1850 sq-ft, 3 bedroom house
% ======================== CODE HERE =========================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

fprintf('\nModel training complete. Press enter to start Prediction.\n');
pause;
pred_x = zeros(1,num_features);

for i = 1:num_features
    pred_x(i)=input(strcat("Enter feature no. ",int2str(i),": "));
end

pred_y = [1,(pred_x-mu)./sigma]*theta;

% ============================================================

fprintf('\nPredicted output: \n%f\n', pred_y);

fprintf('\n\nProgram paused. Press enter to Exit.\n');
pause;






























