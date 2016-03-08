function [W1, W2, b1, b2]=neural(X,Y,rho,lambda,h,itn)  %Xtest,Ytest)
%input: data matrix X, labels Y, test data matrix Xtest, test label matrix
%input: weight decay lambda
%input: no. hidden units h
%Ytest
%output: optimal weight matrix, square loss function, using back
%progpagation
%output error rate

[m,n]=size(X);
XY=[X;Y'];
Ind=randsample(1:n,n,false);
XY=XY(:,Ind);
X=XY(1:m,:);
Y=XY(m+1,:)';

%initialise weights 
%no. of hidden units in the single layer is now 4
% W1Old=ones(h,m); 
% W2Old=ones(1,h); 
W1=-0.1+0.2*rand(h,m);    b1=-0.1+0.2*rand(h,1);
W2=-0.1+0.2*rand(1,h);    b2=-0.1+0.2*rand(1,1);
%activation function: sigmoid function
epoch=1;

%while((norm(W1Old-W1)>tol || norm(W2Old-W2)>tol) && k<=100)
for epoch=1:itn
    
    for i=1:n  %online update
%         W1Old=W1; W2Old=W2;
        
        %propagate forward 
        %input layer
        y_0=X(:,i);
        z_0=1./(1.+exp(-y_0));
        %layer1
        y_1=W1*z_0+b1 ;
        for t=1:h
            z_1(t)=1/(1+exp(-y_1(t)));
        end
% %         z_1=1./(1.+exp(-y_1)) \in R^1-by-h
        %layer2 =output layer
        y_2=W2*z_1'+b2 ; %in R
       

        %back propagate
        del_2=-2*(Y(i)-y_2); %in R
        %del_1=del_2*W2'.*((1./(1.+exp(-y_1))).*(1.-(1./(1.+exp(-y_1)))))
        for k=1:h
            del_1(k)= (exp(-y_1(k))/(1+exp(-y_1(k)))^2)*del_2*W2(1,k);
        end
%del_1 %\in R^h

         %Weight change
        %DEL_W1=del_1*y_0'; %\in R^h-by-m
        for s=1:h
            for p=1:m
            DEL_W1(s,p)=del_1(s)*z_0(p);
            end
        end
        DEL_W2=del_2*z_1';
        DEL_b1=del_1;
        DEL_b2=del_2;


        %weight update
        W2=W2-rho*(DEL_W2'+2*lambda*W2);
%         W1=W1-rho*(DEL_W1+2*lambda*W1);
        for r=1:m
           W1(:,r)= W1(:,r)-rho*(DEL_W1(:,r)+2*lambda*W1(:,r));
        end
        b1=b1-rho*DEL_b1';
        b2=b2-rho*DEL_b2;


        end
epoch=epoch+1;
end

