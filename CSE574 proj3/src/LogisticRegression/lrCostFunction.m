function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y);              % number of training examples
J = 0;
grad = zeros(size(theta));
hypothesis = sigmoid(X*theta);
J = -(1/m)*(y'*log(hypothesis) + (1-y)'*log(1-hypothesis)) + (lambda/(2*m))*sum(theta(2:end).^2) ;
grad(1) = (1/m) * (X(:,1)' * (hypothesis - y));
grad(2:end) = (1/m) * (X(:,2:end)' * (hypothesis - y)) + (lambda/m)*theta(2:end);
grad = grad(:);

end
