function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
accum_1 = zeros(size(Theta1));
accum_2 = zeros(size(Theta2));

% Recoding vector y
Y = zeros(m,num_labels);
for j = 1:m
    Y(j,(y(j)+1)) = 1;
end
% Add ones to the X data matrix
X = [ones(m, 1) X];
hidden = sigmoid(X * Theta1');    
% Add ones to the hidden layer data matrix
hidden = [ones(m, 1) hidden];
output = sigmoid(hidden * Theta2');
for i = 1:m
    J = J + (-(1/m)*(Y(i,:)*log(output(i,:)') + (1-Y(i,:))*log(1-output(i,:)')));
end
J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
% Learning Algorithm
X = X(:,2:end);
for t = 1:m
% Step 1:Forward Pass
    a_1 = X(t,:)';
    a_1 = [1 ; a_1];
    
    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1 ; a_2];
    
    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    
% Step 2:Error Calculation for Output Layer
    delta_3 = a_3 - Y(t,:)';
    
% Step 3:Error Calculation for Hidden Layer    
    delta_2 = (Theta2'*delta_3) .* sigmoidGradient([1 ; z_2]);
    delta_2 = delta_2(2:end);
    
% Step 4:Accumulate the Gradient
    accum_1 = accum_1 + delta_2 * a_1';
    accum_2 = accum_2 + delta_3 * a_2';
end
% Step 5:Unregularized gradient
    Theta1_grad = (1/m) .* accum_1;
    Theta2_grad = (1/m) .* accum_2;
% Step 6:Adding Regularization term
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
