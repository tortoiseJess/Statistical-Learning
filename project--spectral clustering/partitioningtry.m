
k=49;
IDX=kmeans(U,k);
g1=reshape(IDX,size(l));
figure(3),
imshow(uint8(g1),[]);
title(sprintf('m=%g k=%g',r,k))
hold off

    

