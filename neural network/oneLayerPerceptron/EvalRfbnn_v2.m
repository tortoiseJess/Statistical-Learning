function [y_hat_train,y_hat_test,H_diag]=EvalRfbnn_v2(X,Y,Xtest,h)
%input: X=Xtrain, Y=ylabel, Xvad=new data point, h=no. of
%clusters
%output: rbfnn model value of y model of Xvad data
%        H diag matrix elts
[d r]=size(Xtest);
[m n]=size(X);
[PHI,Mu,Sigma]=RBMatrix(X,h);
Y_new=[zeros(n,1),Y];
for k=1:n
    if Y_new(k,1)==Y(k,1)
        Y_new(k,1)=1;
    end
end

W=(PHI'*PHI)\PHI'*Y_new;
H=PHI*((PHI'*PHI)\PHI');
y_hat_train=H*Y_new;
PHIval=ones(r,h+1);

for i=1:r
    for j=1:h
        Xtemp=Xtest(:,i)-Mu(j);
        PHIval(i,j+1)=exp(-(norm(Xtemp)^2)/(2*Sigma(j)^2));
    end
end

y_hat_test=PHIval*W;
 H_diag=diag(H);
