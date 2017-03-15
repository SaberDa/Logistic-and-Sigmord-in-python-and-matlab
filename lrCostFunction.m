function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


htheta = sigmoid(X*theta);
J = 1/m*sum(-y .* log(htheta) - (1-y) .* log(1-htheta)) + lambda/(2*m)* sum(theta(2:end) .^ 2);

temp = theta;
temp(1) = 0;
grad = 1/m*(X' * (htheta - y) + lambda*temp);

grad = grad(:);

end
