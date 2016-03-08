function [beta train_error] = log_regression(X,Y,Xtest,Ytest)
%input: data matrix X, label matrix Y 
    %eg Y=class_train
%output: optimal beta (dx1) by newton Raphson


[d,n]=size(X);
beta=zeros(d,1); 
betaOld=ones(d,1);
P=[]; W=[];
tol=1e-6;
Y=Y-ones(n,1); % label is either Y=1 or Y=0

%if stopping criteria not met, compute beta
while(norm(beta-betaOld)>tol)
    P=[]; W=[];
    
    %compute probability
    for i=1:n
        p=exp(beta'*X(:,i))/(1+exp(beta'*X(:,i)));
        w=p*(1-p);
        P=[P; p];
       W=[W, w];
    end

    
    %diagonal W
    W=diag(W,0);
    %Z
    Z=X'*beta+W\(Y-P);
    betaOld=beta;
    beta=(X*W*X')\(X*W*Z);
    
end

%training error
[d,m]=size(Xtest);
group=zeros(m,1);
Ytest=Ytest-ones(m,1);
R=beta'*Xtest;

for j=1:m
    if (R(j)<0)
        group(j)=0; 
    end
end
train_error=sum(group~=Ytest)/m;
