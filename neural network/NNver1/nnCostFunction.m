function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = nnCostFunction(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1) X];

% foward propagation
a2 = sigmoid(Theta1 * X');
a2 = [ones(m,1) a2'];

h_theta = sigmoid(Theta2 * a2'); % h_theta equals z3

% y(k) - the great trick - we need to recode the labels as vectors containing only values 0 or 1
yk = zeros(num_labels, m); 
for i=1:m,
  yk(y(i),i)=1;
end

% follow the form
J = (1/m) * sum ( sum (  (-yk) .* log(h_theta)  -  (1-yk) .* log(1-h_theta) ));

% Backprop

for t=1:m,
        a1 = X(t,:); % X already have bias
        z2 = Theta1 * a1';

        a2 = sigmoid(z2);
        a2 = [1 ; a2]; % add bias

        z3 = Theta2 * a2;

        a3 = sigmoid(z3); % final activation layer a3 == h(theta)
           
        z2=[1; z2]; % bias

        delta_3 = a3 - yk(:,t); % y(k) trick - getting columns of t element
        delta_2 = (Theta2' * delta_3) .* sigmoid(z2) .* (1 - sigmoid(z2));

        % skipping sigma2(0) 
        delta_2 = delta_2(2:end); 

        Theta2_grad = Theta2_grad + delta_3 * a2';
        Theta1_grad = Theta1_grad + delta_2 * a1;
end;

Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end)); 

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end