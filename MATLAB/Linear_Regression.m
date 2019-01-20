%%Arpan Bag
%%Linear Regression

%% Initialization
%% Clear and Close Figures
clear ; close all; clc



%% Load Data
fprintf('Loading data ...\n');
data = load('Sample_Data.txt');	%The sample file containing the training data. The ﬁrst column is the size of the house (in square feet), the second column is the number of bedrooms, and the third column is the price of the house. 
X = data(:, 1:2);	%Inputs
y = data(:, 3);		%Outputs
m = length(y);		%Number of training examples



% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Part 1: Feature Normalization ================


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);		% Returned X = Feature normalized training inputs, X = (X-mu)/sigma, where mu = mean of X, sigma = standard deviation of X



% Add x0 intercept term to X (As x0 = 1 for all the examples)
X = [ones(m, 1) X];



%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');	

% Choose some alpha value
alpha = 0.01;		%Learning rate
num_iters = 500;	%Number if iterations


% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);


% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1850 sq-ft, 3 bedroom house
% ======================== CODE HERE =========================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = [1,([1650 3]-mu)./sigma]*theta;

% ============================================================

fprintf(['Predicted price of a 1850 sq-ft, 3 bedroom house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to Exit.\n');
pause;






























