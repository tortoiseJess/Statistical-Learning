function yhat=classify(Xtest, b,b0)

n=size(Xtest,2);
ytemp=1234567*ones(n,1);
M=b'*Xtest+b0;
%classify data according to sign of M
for i=1:n
    if M(i)>0
        ytemp(i)=1;
    elseif M(i)<0
        ytemp(i)=-1;
    else 
        ytemp(i)=0;
    end
end
yhat=ytemp;
    