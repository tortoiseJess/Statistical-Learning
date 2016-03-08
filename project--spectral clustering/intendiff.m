function ID= intendiff(v)
%v is a vector of intensity values. This function outputs the pairwise
%intensity difference between v(i) and v(j) and record it in ID(i,j)
[n,~]=size(v);
IN=zeros(n,n);
for i=1:n
    IN(i,:)=v(i).*ones(1,n)-v';
end
ID=IN;
end

