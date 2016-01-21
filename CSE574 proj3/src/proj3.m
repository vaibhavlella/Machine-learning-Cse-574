%% Matrix containing train and test data
load('dataraw.mat');

%% Getting out feature matrix and label output matrix from train data 
train_X=mnisttrain(:,1:end-1);
train_Y=mnisttrain(:,end);

%% Getting out feature matrix and label output matrix from test data 
test_X=mnisttest(:,1:end-1);
test_Y=mnisttest(:,end);
save data.mat;
clc;clear;

%% Calling Logistic Regression
logisticReg;

clc;clear;

%% Calling Neural Network with one hidden layer
neuralNetwork;
clc;clear;

%% Convolutional Neural Network
test_example_CNN;
clc;clear;

%% Activation Function 
h = 'sigmoid';
%% Credentials
UBitName=['v' 'a' 'i' 'b' 'h' 'a' 'v' 'l'];
personNumber=['5' '0' '1' '6' '9' '8' '5' '9'];

%% Generating .mat for Format Checker
load('logisticReg.mat','Wlr','blr','ErrorLrTrain','ErrorLrTest')
load('neuralnet.mat','Wnn1','bnn1','Wnn2','bnn2','ErrorNNTrain','ErrorNNTest')
load('cnn.mat','ErrorCnnTrain','ErrorCnnTest')
save proj3.mat;
