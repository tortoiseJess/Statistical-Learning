function [err,y_group]=ErrRate_v2(ymodel,ylabel)
%input: col vector of fitted model ymodel
%       col vector of ground truth ylabel
%output: mean misclassification error
%        ylabel of ymodel
n=size(ymodel,1);
y_group=ones(n,1);

for k=1:n
    if (ymodel(k,1)>ymodel(k,2))
        y_group(k)=0;
    end
end
err=sum(y_group~=ylabel)/n;