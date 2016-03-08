function [W1, W2, b1, b2]=neural(X,Y,lambda,h)  %Xtest,Ytest)
%input: data matrix X, labels Y, test data matrix Xtest, test label matrix
%input: weight decay lambda
%input: no. hidden units h
%Ytest
%output: optimal weight matrix, square loss function, using back
%progpagation
%output error rate


tol=1e-6;
[m,n]=size(X);
rho=0.01;
i=1;
k=1;



%initialise weights 
%no. of hidden units in the single layer is now 4
W1Old=ones(h,m); 
W2Old=ones(1,h); 
W1=rand(h,m);    b1=rand(h,1);
W2=rand(1,h);    b2=rand(1,1);
%activation function: sigmoid function


%while((norm(W1Old-W1)>tol || norm(W2Old-W2)>tol) && k<=100)
for k=1:70
    for i=1:n  %online update
        W1Old=W1; W2Old=W2;
        
        %propagate forward get z_l
        y_0=X(:,i);
        z_0=y_0;
        %layer1
        y_1=W1*z_0+b1 ;
        z_1=1./(1.+exp(-y_1));
        %layer2
        y_2=W2*z_1+b2 ;
        z_2=1/(1+exp(-y_2));
        y_3=z_2;


        %error of output layer
        del_3=-2*(Y(i)-y_3); %different from notes


        %back propgate get del_l
        del_2=del_3.*(1./(1.+exp(-y_2))).*(1.-(1./(1.+exp(-y_2))));
        del_1=del_2*W2'.*((1./(1.+exp(-y_1))).*(1.-(1./(1.+exp(-y_1)))));


         %Weight change
        DEL_W1=del_1*z_0';
        DEL_W2=del_2*z_1';
        DEL_b1=del_1;
        DEL_b2=del_2;


        %weight update
        W2=W2-rho*(DEL_W2+2*lambda*W2);
        W1=W1-rho*(DEL_W1+2*lambda*W1);
        b1=b1-rho*DEL_b1;
        b2=b2-rho*DEL_b2;


        end
        k=k+1;
 end

k