clear all
load('Ion.trin.mat')
load('Ion.test.mat')
[m, n]=size(Xtrain);
[d, r]=size(Xtest);
IND=randsample(1:n,n,'false');
L=zeros(1,4);
clusters=20;
CVerr=zeros(1,clusters);
LOOerr=zeros(1,clusters);

%4 fold cv
for h=1:clusters
    for i=1:4  
        INDEX1=(i-1)*44+1; INDEX2=i*44;
        Xvad=Xtrain(:,IND(INDEX1:INDEX2)); Yvad=ytrain(IND(INDEX1:INDEX2),1);
        REM=IND; REM(:,INDEX1:INDEX2)=[];
         X=Xtrain(:,REM); Y=ytrain(REM,1);

        [y_hat_t,y_hat_vad,H_diag]=EvalRfbnn(X,Y,Xvad,h);
        [err,Xvad_group]=ErrRate(y_hat_vad,Yvad);
        L(i)=err;
    end
CVerr(h)=mean(L);
%optimal model error
[y_hat_train,y_hat_test,H]=EvalRfbnn(Xtrain,ytrain,Xtest,h);
[Err_train,g1]=ErrRate(y_hat_train,ytrain); training_error(h)=Err_train;
[Err_test,g2]=ErrRate(y_hat_test,ytest); test_error(h)=Err_test;
end

[min_err, opt_h1]=min(CVerr)
[y_hat_train,y_hat_test,H]=EvalRfbnn(Xtrain,ytrain,Xtest,opt_h1);
[Err_train,g1]=ErrRate(y_hat_train,ytrain); Err_train
[Err_test,g2]=ErrRate(y_hat_test,ytest); Err_test

figure
plot(linspace(1,clusters,clusters),training_error,'b',linspace(1,clusters,clusters),test_error,'k')
legend('train-err','test-err');
title('h vs errors')
% 
% %LOO

for h=1:clusters
    L=zeros(1,n);
    for i=1:n
        Xvad=Xtrain(:,i) ;
        X=Xtrain; X(:,i)=[] ;
        Y=ytrain; Y(i)=[];
        [y_hat_t,y_hat_vad,H_diag]=EvalRfbnn(X,Y,Xvad,h);
        [err,Xvad_group]=ErrRate(y_hat_vad,ytrain(i,1));
        L(i)=err;
        
    end
    LOOerr(h)=mean(L);

end

[min_err,opt_h2]=min(LOOerr)
[y_hat_train,y_hat_test,H]=EvalRfbnn(Xtrain,ytrain,Xtest,opt_h2);
[Err_train,g1]=ErrRate(y_hat_train,ytrain); Err_train
[Err_test,g2]=ErrRate(y_hat_test,ytest); Err_test



%CLOO

for h=1:clusters
     L=zeros(1,n);
    [y_hat_t,y_hat_vad,H_diag]=EvalRfbnn(Xtrain,ytrain,Xtrain,h);
    [err,X_group]=ErrRate(y_hat_vad,ytrain);
    CLOOerr(h)=sum(((ytrain-X_group)./(1-H_diag)).^2)/n;
    
end
[min_err,opt_h3]=min(CLOOerr)
[y_hat_train,y_hat_test,H]=EvalRfbnn(Xtrain,ytrain,Xtest,opt_h3);
[Err_train,g1]=ErrRate(y_hat_train,ytrain); Err_train
[Err_test,g2]=ErrRate(y_hat_test,ytest); Err_test


