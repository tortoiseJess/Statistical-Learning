function ED = pdistance(m)
%Compute the distance between two point in a m by m matrix, and store the
%restult in a m^2 by m^2 matrix. Ordering follows matlab build-in function
%ind2sub
n=m^2;
W1=zeros(n,n);
W2=zeros(n,n);
[I,J]=ind2sub([m,m],1:n);
for i=1:n
    W1(i,:)=I(i).*ones(1,n)-I;
    W2(i,:)=J(i).*ones(1,n)-J;
end
ED=(W1.^2+W2.^2);
end

