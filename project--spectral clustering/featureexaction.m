d=4;%number of features
k=49;%number of clusters or classes
m=64;%image size
Q=zeros(d,k);
for k=1:k
    p1=reshape(M,m^2,1);
    p2=reshape(g1,m^2,1);
    ind=find(p2==k);
    z=p1(ind);
    averageint=sum(z)/size(z,1);
    varint=var(z);
    area=size(z,1)/(m^2);
    gx=gradient(z,1/(m+1));
    vgx=var(gx);
    %add the distribution thing here
    Q(:,k)=[averageint;varint;area;vgx];
end

    
    
    