function err=misclassify(yhat, ytest)
%input is col vector
n=size(yhat,1);
R=sum(yhat==ytest);

err=1-R/n;
