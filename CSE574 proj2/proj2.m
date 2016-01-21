%% Feature and Label Extraction and Data Partition
fileId=fopen('Querylevelnorm.txt');
fmt=['%f %*s' repmat('%*d:%f',1,46) '%*[^\n]'];
data=textscan(fileId,fmt,'collectoutput',1);
fclose(fileId);
m = cell2mat(data);
InputVector = m(:,2:47);
TargetVector=m(:,1);
TrainingInput=InputVector(1:55699,:);           % training data is 80% of total data
TrainingTarget=TargetVector(1:55699,:);         
ValidationInput=InputVector(55700:62661,:);        % testing data is 10% of total data
ValidationTarget=TargetVector(55700:62661,:);
TestingInput=InputVector(62662:69623,:);     % validation data is 10% of total data
TestingTarget=TargetVector(62662:69623,:);   % all three data sets do not overlap
Erms1 = zeros(1,12);
Erms2 = zeros(1,12);
D = size(InputVector,2);
N1 = size(TrainingInput,1);
trainInd1 = zeros(N1,1);
trainInd1(:,:) = 1:N1;
N2 = size(ValidationInput,1);
validInd1 = zeros(N2,1);
validInd1(:,:) = N1+1:N1+N2;
%%Training model parameters M, µj, Σj, λ,η
M1=20;
                                      % For different values of M
PhiMat1 = ones(N1,M1);                         %Initializing Design Matrix
PhiMat2 = ones(N2,M1);                           %Initializing Design Matrix
[IDX1,C1] = kmeans(TrainingInput,M1);          %Centers for Rbfs
C1(1,:)=1;
mu1=C1';
Var1 = diag(var(TrainingInput)) + (4*eye(46));        %Gaussian Spread Matrix
Sigma1=zeros(D,D,M1);    
for i=1:M1
       Sigma1(:,:,i)=Var1;
end
    for m = 2:M1                               %Constructing Design Matrix
        for n = 1:N1
            PhiMat1(n,m) = rbf(TrainingInput(n,:),C1(m,:),Var1);
        end
    end
    for m = 2:M1                                              %Constructing Design Matrix
        for n = 1:N2
            PhiMat2(n,m) = rbf(ValidationInput(n,:),C1(m,:),Var1);
        end
     end    
%Calculating Closed Form Maximum Likelihood Solution for w
i=1;
%for lambda1=0.1:0.1:1
lambda1=0.6;
Wml = zeros(M1,1);                           %Initializing Weight Matrix                  
Wml = (lambda1*eye(M1) + (PhiMat1' * PhiMat1)) \ (PhiMat1'* TrainingTarget);         %Calculating Maximum Likelihood Solution for w without Regularization
%Wml = (PhiMat1' * PhiMat1) \ (PhiMat1'* TrainingTarget);         %Calculating Maximum Likelihood Solution for w with Regularization
w1=Wml;                   
%Calculating sum-of-square errors on Training Set
hypothesisTr = PhiMat1 * Wml;
sqrErrorsTr = (TrainingTarget - hypothesisTr).^2;
EDw = 0.5 * sum(sqrErrorsTr);
%EWw = 0.5 * sum(Wml.^2);
%Ew = EDw + lambda1 * EWw;
%Erms(M1) = sqrt((2*EDw)/N1);
Erms1(i) = sqrt((2*EDw)/N1);
trainPer1 = Erms1(i);
%Calculating sum-of-square errors on Validation Set
hypothesisV = PhiMat2 * Wml;
sqrErrorsV = (ValidationTarget - hypothesisV).^2;
EDw = 0.5 * sum(sqrErrorsV);
%EWw = 0.5 * sum(Wml.^2);
%Ew = EDw + lambda1 * EWw;
%Erms2(M1) = sqrt((2*EDw)/N2);
Erms2(i) = sqrt((2*EDw)/N2);
validPer1 = Erms2(i);
%i=i+1;
%end
%%Stochastic Gradient Descent
E=100000;
eta1=(6/N1)*ones(1,E);
weight=zeros(M1,1);
dw1=zeros(M1,E);
Erms=zeros(1,E);
w01=weight;
for iter = 1:E
    if iter>N1
        DelW = (eta1(iter)) * ((PhiMat1(iter-N1,:)' * ( TrainingTarget(iter-N1,1) - weight'*PhiMat1(iter-N1,:)')) - lambda1*weight);
weight = weight + DelW;
dw1(:,iter)=DelW;
h = PhiMat1 * weight;
sqrErr = (TrainingTarget - h).^2;
Err = 0.5 * sum(sqrErr);
Erms(iter) = sqrt((2*Err)/N1);

%if iter>1 && iter<E && Erms(iter) < Erms(iter-1) 
%        eta1(iter+1) = eta1(iter)*1.05;
%end
else
DelW = (eta1(iter)) * ((PhiMat1(iter,:)' * ( TrainingTarget(iter,1) - weight'*PhiMat1(iter,:)')) - lambda1*weight);
weight = weight + DelW;
dw1(:,iter)=DelW;
h = PhiMat1 * weight;
sqrErr = (TrainingTarget - h).^2;
Err = 0.5 * sum(sqrErr);
Erms(iter) = sqrt((2*Err)/N1);

if iter>1 && iter<E && Erms(iter) < Erms(iter-1) 
        eta1(iter+1) = eta1(iter)*1.05;
end
    end
  %{
        elseif Erms(iter) > Erms(iter-1)
        eta1(iter) = eta1(iter)*0.5;
       
        while(Erms(iter)>Erms(iter-1))
        weight(:,iter) = weight(:,iter-1);
        eta1(iter) = eta1(iter)*0.5;
        DelW = (eta1(iter)) * ((PhiMat1(iter,:)' * ( TrainingTarget(iter,1) - weight(:,iter)'*PhiMat1(iter,:)')) - lambda1*weight(:,iter));
        weight(:,iter) = weight(:,iter) + DelW;
        dw1(:,iter)=DelW;
        h = PhiMat1 * weight(:,iter);
        sqrErr = (TrainingTarget - h).^2;
        Err = 0.5 * sum(sqrErr);
        Erms(iter) = sqrt((2*Err)/N1);
        end
        eta1(iter+1) = eta1(iter)*0.8;
        
    else
            eta1(iter+1) = eta1(iter);
            %}
end
load synthetic.mat;
SynInp1 = x';
SynTrgt1=t;
SynTrainingInput=SynInp1(1:1400,:);           % training data is 70% of total data
SynTrainingTarget=SynTrgt1(1:1400,:);         
SynValidationInput=SynInp1(1401:2000,:);        % Validation data is 30% of total data
SynValidationTarget=SynTrgt1(1401:2000,:);
SynErms1 = zeros(1,12);
SynErms2 = zeros(1,12);
SD = size(SynInp1,2);
SN1 = size(SynTrainingInput,1);
trainInd2 = zeros(SN1,1);
trainInd2(:,:) = 1:SN1;
SN2 = size(SynValidationInput,1);
validInd2 = zeros(SN2,1);
validInd2(:,:) = SN1+1:SN1+SN2;
%Training model parameters M, µj, Σj, λ,η
M2=5;
                                      % For different values of M
SynPhiMat1 = ones(SN1,M2);                         %Initializing Design Matrix
SynPhiMat2 = ones(SN2,M2);                           %Initializing Design Matrix
[SIDX1,SC1] = kmeans(SynTrainingInput,M2);          %Centers for Rbfs
SC1(1,:)=1;
mu2=SC1';
Var2 = diag(var(SynTrainingInput));        %Gaussian Spread Matrix
Sigma2=zeros(SD,SD,M2);    
for i=1:M2
       Sigma2(:,:,i)=Var2;
end
    for m = 2:M2                               %Constructing Design Matrix
        for n = 1:SN1
            SynPhiMat1(n,m) = rbf(SynTrainingInput(n,:),SC1(m,:),Var2);
        end
    end
    for m = 2:M2                                              %Constructing Design Matrix
        for n = 1:SN2
            SynPhiMat2(n,m) = rbf(SynValidationInput(n,:),SC1(m,:),Var2);
        end
     end    
%Calculating Closed Form Maximum Likelihood Solution for w
j=1;
%for lambda1=0.1:0.1:1
lambda2=0.4;
SWml = zeros(M2,1);                           %Initializing Weight Matrix                  
SWml = (lambda2*eye(M2) + (SynPhiMat1' * SynPhiMat1)) \ (SynPhiMat1'* SynTrainingTarget);         %Calculating Maximum Likelihood Solution for w without Regularization
%SWml = (SynPhiMat1' * SynPhiMat1) \ (SynPhiMat1'* SynTrainingTarget);         %Calculating Maximum Likelihood Solution for w with Regularization
w2=SWml;                   
%Calculating sum-of-square errors on Training Set
SynhypothesisTr = SynPhiMat1 * SWml;
SynsqrErrorsTr = (SynTrainingTarget - SynhypothesisTr).^2;
SEDw = 0.5 * sum(SynsqrErrorsTr);
%SEWw = 0.5 * sum(SWml.^2);
%SEw = SEDw + lambda2 * SEWw;
%SynErms(M1) = sqrt((2*SEDw)/SN1);
SynErms1(j) = sqrt((2*SEDw)/SN1);
trainPer2 = SynErms1(j);
%Calculating sum-of-square errors on Validation Set
SynhypothesisV = SynPhiMat2 * SWml;
SynsqrErrorsV = (SynValidationTarget - SynhypothesisV).^2;
SEDw = 0.5 * sum(SynsqrErrorsV);
%SEWw = 0.5 * sum(SWml.^2);
%SEw = SEDw + lambda2 * SEWw;
%SynErms2(M1) = sqrt((2*SEDw)/SN2);
SynErms2(j) = sqrt((2*SEDw)/SN2);
validPer2 = SynErms2(j);
j=i+1;
%end
E2=SN1;
eta2=(0.8/SN1)*ones(1,E2);
weight1=zeros(M2,1);
Syndw1=zeros(M2,E2);
SynErms=zeros(1,E2);
w02=weight1;
for iter = 1:E2
SynDelW = (eta2(iter)) * ((SynPhiMat1(iter,:)' * ( SynTrainingTarget(iter,1) - weight1'*SynPhiMat1(iter,:)')) - lambda2*weight1);
weight1 = weight1 +SynDelW;
dw2(:,iter)=SynDelW;
h1 = SynPhiMat1 * weight1;
sqrErr1 = (SynTrainingTarget - h1).^2;
Err1 = 0.5 * sum(sqrErr1);
Erms5(iter) = sqrt((2*Err1)/SN1);

%if iter>1 && iter<E2 && Erms5(iter) < Erms5(iter-1) 
%        eta2(iter+1) = eta2(iter)*1.05;
%end
end
figure;                                             % open a new figure window
plot(Erms,'rx', 'MarkerSize', 10);                  % Plot the data
ylabel('SynERMS2');                                     % Set the y􀀀axis label
xlabel('Model Complexity');                         % Set the x􀀀axis label
%xlabel('Lamda');
figure;                                             % open a new figure window
plot(Erms5,'rx', 'MarkerSize', 10);                  % Plot the data
ylabel('Error');                                     % Set the y􀀀axis label
xlabel('Iterations')
save proj21.mat;
clc;clear;
load proj21.mat M1 M2 N1 N2 mu1 mu2 lambda1 lambda2 w1 w2 validInd1 validInd2 validPer1 validPer2 trainInd1 trainInd2 trainPer1 trainPer2 Sigma1 Sigma2 w01 w02 dw2 dw1 eta1 eta2
save proj2.mat;

   