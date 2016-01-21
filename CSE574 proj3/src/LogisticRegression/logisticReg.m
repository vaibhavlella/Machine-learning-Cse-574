%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data.mat'); % training data stored in arrays train_X, train_Y_M
m = size(train_X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = train_X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(train_X, train_Y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, train_X);
pred = pred - 1;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == train_Y)) * 100);
ErrorLrTrain = (sum(train_Y ~= pred)/size(train_Y,1))*100;
%% ================= Evaluation ==============
predTest = predictOneVsAll(all_theta, test_X);
predTest = predTest - 1;
ErrorLrTest = (sum(test_Y ~= predTest)/size(test_Y,1))*100;

templr=all_theta';
Wlr=templr(1:end-1,:);
blr=templr(end,:);

save logisticReg.mat;