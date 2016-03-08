
load('sediment_stat841.mat');
%index1=1:1413-195;
%index2=1413-195+1:1413;
%X1=randsample(index1,706,true);
%X2=randsample(index2,707,true);
%X=[Xtrain(:,X1),Xtrain(:,X2)];
%Y=[class_train(X1);class_train(X2)]-1;

Xindices = randsample(1:1218, 195);
temp = 1;
for i = Xindices
    X(:, temp) = Xtrain(:, i);
    temp = temp + 1;
end
Y = [ones(195, 1); class_train(1219:1412, :)];

%fit neural network
[W1 W2 b1 b2]=neural(X, Y,0,4);



%training error
[d,m]=size(Xtest);
group=zeros(m,1);
Ytest=class_test-1;

 for i=1:m
        y_0=Xtest(:,i);
        z_0=y_0;
        %layer1
        y_1=W1*z_0 +b1 ;
        z_1=1./(1.+exp(-y_1));
        %layer2
        y_2=W2*z_1 +b2 ;
        z_2=1/(1+exp(-y_2));
        y_3(i)=z_2 ;
 end
 y_3;
for i=1:m
    if (y_3(i)>=0.5)
        group(i)=1;
    end
end

test_error=1-sum(group~=Ytest)/m
