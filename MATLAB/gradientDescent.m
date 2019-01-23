%%Arpan Bag
%%The Gradient Descent function

function [theta, J_history] = gradientDescent(X, Y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, Y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values
m = length(Y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % =========================== CODE HERE ==============================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	
	derivativePart = 1/m* sum((X*theta - Y).*X);
	theta = theta - (alpha.*derivativePart)';

	

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, Y, theta);

end


end
