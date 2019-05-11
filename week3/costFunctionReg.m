function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
th = theta
th(1) = []
h0 = sigmoid(X*theta) ;
j1 = (lambda/(2*m)) * th' * th ;
j2 = (-y'*log(h0)-((1-y)'*log(1-h0)))/m ;
J = j1 + j2 ;
grad0 = X'*(h0-y)/m ;
grad = grad0 + (lambda*theta/m) ;
grad(1) = grad0(1) ;
h_theta = sigmoid(X*theta);


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
