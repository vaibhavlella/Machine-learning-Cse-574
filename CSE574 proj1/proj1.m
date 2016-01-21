UBitName=['v' 'a' 'i' 'b' 'h' 'a' 'v' 'l'];
personNumber=['5' '0' '1' '6' '9' '8' '5' '9'];
%% Mean Calculations
mu1=mean(CSScoreUSNews,1);
mu2=mean(ResearchOverhead,1);
mu3=mean(AdminBasePay,1);
mu4=mean(Tuitionoutstate,1);

%% Variance Calculations
var1=var(CSScoreUSNews,0,1);
var2=var(ResearchOverhead,0,1);
var3=var(AdminBasePay,0,1);
var4=var(Tuitionoutstate,0,1);

%%Standard Deviation Calculations
sigma1=std(CSScoreUSNews,0,1);
sigma2=std(ResearchOverhead,0,1);
sigma3=std(AdminBasePay,0,1);
sigma4=std(Tuitionoutstate,0,1);

%% Covariance Matrix
data1234=[CSScoreUSNews,ResearchOverhead,AdminBasePay,Tuitionoutstate];
covarianceMat=cov(data1234);

%% Correlation matrix:
correlationMat = corrcoef(data1234);
%% logLikelihood
params1=[mu1 sigma1];
like1=normlike(params1,CSScoreUSNews);
params2=[mu2 sigma2];
like2=normlike(params2,ResearchOverhead);
params3=[mu3 sigma3];
like3=normlike(params3,AdminBasePay);
params4=[mu4 sigma4];
like4=normlike(params4,Tuitionoutstate);
logLikelihood=-(like1+like2+like3+like4);

%% BNgraph
BNgraph = zeros(4);
BNgraph(1,2) = 1;
BNgraph(1,3) = 1;
BNgraph(1,4) = 1;
BNgraph(2,3) = 1;
BNgraph(2,4) = 1;
BNgraph(3,4) = 1;

%% BNlogLikelihood
mean1234=[mu1,mu2,mu3,mu4];
BNlogLikelihood=-ecmnobj(data1234,mean1234,covarianceMat);
%%
figure;
scatter(CSScoreUSNews,ResearchOverhead);
xlabel('CSScoreUSNews');
ylabel('ResearchOverhead');
figure;
scatter(CSScoreUSNews,AdminBasePay);
xlabel('CSScoreUSNews');
ylabel('AdminBasePay');
figure;
scatter(CSScoreUSNews,Tuitionoutstate);
xlabel('CSScoreUSNews');
ylabel('Tuitionoutstate');
figure;
scatter(ResearchOverhead,AdminBasePay);
xlabel('ResearchOverhead');
ylabel('AdminBasePay');
figure;
scatter(AdminBasePay,Tuitionoutstate);
xlabel('AdminBasePay');
ylabel('Tuitionoutstate');
figure;
scatter(ResearchOverhead,Tuitionoutstate);
xlabel('ResearchOverhead');
ylabel('Tuitionoutstate');