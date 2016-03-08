%preprocessing data
load('Ion.trin.mat');
load('Ion.test.mat');

% for t=1:10
%     lambda=0.1*10^(-t)
%    [W1, W2, b1, b2,Err_train,Err_test]=neural_v2(Xtrain,ytrain,Xtest,ytest,0.01,lambda,4);
%    Err_train(t)=Err_train;
%    Err_test(t)=Err_test;
%    
% end
%  plot(linspace(1,10,10),Err_train,'b',linspace(1,10,10),Err_test,'k')
%     legend('train-err','test-err');
%     title('lambda vs err')

[W1, W2, b1, b2,E1, E2 ,E3,err]=neural_v2(Xtrain,ytrain,Xtest,ytest,0.01,0.0001,10)  
