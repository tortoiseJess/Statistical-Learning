function [err,y_group]=ErrRate(ymodel,ylabel)
%input: col vector of fitted model ymodel
%       col vector of ground truth ylabel
%output: mean misclassification error
%        ylabel of ymodel
n=size(ymodel,1);
y_group=ones(n,1);

for k=1:n
    if (ymodel(k)<=0.5)
        y_group(k,1)=0;
    end
end
err=sum(y_group~=ylabel)/n;