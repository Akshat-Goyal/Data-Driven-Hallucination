function E=imgraphen(IDX,CO,CM_h,CM_v)
% Compute energy of the graph based on image
if length(size(IDX))==2
    [nh,nw]=size(IDX);
    IDX=reshape(IDX,[1 nh nw]);
end
E=CO(IDX);
E=sum(E(:));
IDX=squeeze(IDX);
[nh,nw]=size(IDX);
for i=1:nh
    for j=1:nw
        p=IDX(i,j);
        if j<nw
            E=E+CM_h(p,IDX(i,j+1),i,j);
        end
        if i<nh
            E=E+CM_v(p,IDX(i+1,j),i,j);
        end
    end
end