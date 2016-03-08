% Stat 841 Assignment 2 Question 1(b)

% Initialization
clear; close all; clc;

% Setup the parameters you will use for this exercise
input_layer_size = 9;
hidden_layer_size = 4;
num_labels = 2;

% Load training data
load('sediment_stat841.mat');
X = Xtrain';
y = class_train;

m = size(X, 1);

epsilon_init = 0.12;
initial_Theta1 = rand(hidden_layer_size, 1 + input_layer_size) * 2 * epsilon_init - epsilon_init;
initial_Theta2 = rand(num_labels, 1 + hidden_layer_size) * 2 * epsilon_init - epsilon_init;
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 30);
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

pred = predict(Theta1, Theta2, X);
error_rate = 1 - mean(double(pred == y))