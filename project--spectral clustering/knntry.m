clear
clc
M=imread('peppers_color.jpg');
l=rgb2gray(M);
l=imresize(l,[64 64]);
figure(1)
imshow(l);
[m,m]=size(l);
n=m^2;
M=double(l)./255;
v=reshape(M,n,1);
ID=intendiff(v);
ED=pdistance(m);
ID=abs(ID);
scale1=100;
scale2=1;
sigmai=scale1*var(v);
sigmaxy=scale2*min(var(ED));
W=exp(-ID./sigmai).*exp(-ED./sigmaxy);
k=30;
for i=1:n
    rw=sort(W(i,:),'descend');
    threshold=rw(k+1);
    W(i,:)=W(i,:).*(W(i,:)>=threshold);
end
W=W-diag(ones(n,1));
W=max(W,W');
D=diag(sum(W,2));
L=D-W;
L=sparse(L);
r=20;
[V, E]=eigs(L,r,'sm');
%reorder the eigenvectors and eigenvalues in ascending order
ev=diag(E);
ev=ev(r:-1:1);
U=zeros(size(V));
U=V(:,r:-1:1);
%remove the zeros eigenvalue and its corresponding eigenvector
ev=ev(2:r);
U=U(:,2:r);