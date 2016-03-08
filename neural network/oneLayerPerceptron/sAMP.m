% index1=1:1413-195;
% index2=1413-195+1:1413;
% X1=randsample(index1,706,true);
% X2=randsample(index2,707,true);
% X=[Xtrain(:,X1),Xtrain(:,X2)];
% Y=[class_train(X1);class_train(X2)]-1;

XY=[Xtrain;ytrain];
XY=randsample(1:176,176,false);
X=XY(1:33,:);
Y=XY(34,:);