function [b, b0]=HardMarg(X,y)
% X is a d-by-n matrix
%y is a n-by-1 vector

%find alpha=max of dual L
[d,n]=size(X);
S=(X*diag(y))'*(X*diag(y));
H=S;
f=-ones(n,1);
Aeq=y';
beq=0;
lb=zeros(n,1); 
ub=(1e+6)*ones(n,1); 
options=optimset('Algorithm','interior-point-convex','Display','off');
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub,[],options);

%form b
b=X*diag(y)*alpha;
%find b0;
alphatemp=[]; Xtemp=[]; ytemp=[];
for i=1:n
    if alpha(i)>0.1
        alphatemp=[alphatemp, alpha(i)];
        Xtemp=[Xtemp, X(:,i)];
        ytemp=[ytemp; y(i)];
    end
end
% alphatemp
% ytemp
k=size(ytemp,1);
I=eye(k,k);
B0=diag(inv(diag(ytemp))*I)-(b'*Xtemp)'
b0=mean(B0);

%  [alphamax,j]=max(alphatemp)
%  bverify=-b'*Xtemp(:,j)+1/ytemp(j)




