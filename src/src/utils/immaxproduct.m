function [IDX,En]=immaxproduct(CO,CM_h,CM_v,nIterations,alpha)
% Max-product belief propagation on image

[nstates,height,width]=size(CO);
if size(CM_h)~=[nstates nstates height width-1]
    error('The dimension of CM_h is incorrect!');
end
if size(CM_v)~=[nstates nstates height-1 width]
    error('The dimension of CM_v is incorrect!');
end

CMtb=reshape(CM_v,nstates,nstates*(height-1)*width);
CMbt=reshape(permute(CM_v,[2,1,3,4]),nstates,nstates*(height-1)*width);
CMlr=reshape(CM_h,nstates,nstates*height*(width-1));
CMrl=reshape(permute(CM_h,[2,1,3,4]),nstates,nstates*height*(width-1));
Mtb=zeros(nstates,(height-1),width);
Mbt=Mtb;
Mlr=zeros(nstates,height,width-1);
Mrl=Mlr;
[foo,IDX]=min(CO,[],1);
En(1)=imgraphen(IDX,CO,CM_h,CM_v);

for i=1:nIterations
    % update message from top to bottom
    Mtb1=zeros(nstates,height-1,width);
    Mtb1(:,2:end,:)=Mtb1(:,2:end,:)+Mtb(:,1:end-1,:);
    Mtb1(:,:,1:end-1)=Mtb1(:,:,1:width-1)+Mrl(:,1:end-1,:);
    Mtb1(:,:,2:end)=Mtb1(:,:,2:end)+Mlr(:,1:end-1,:);
    Mtb1=Mtb1+CO(:,1:end-1,:);
    Mtb1=kron(reshape(Mtb1,nstates,(height-1)*width),ones(1,nstates))+CMtb;
    Mtb1=reshape(min(Mtb1,[],1),[nstates,height-1,width]);
    
    % update message from bottom to top
    Mbt1=zeros(nstates,height-1,width);
    Mbt1(:,1:end-1,:)=Mbt1(:,1:end-1,:)+Mbt(:,2:end,:);
    Mbt1(:,:,1:end-1)=Mbt1(:,:,1:end-1)+Mrl(:,2:end,:);
    Mbt1(:,:,2:end)=Mbt1(:,:,2:end)+Mlr(:,2:end,:);
    Mbt1=Mbt1+CO(:,2:end,:);
    Mbt1=kron(reshape(Mbt1,nstates,(height-1)*width),ones(1,nstates))+CMbt;
    Mbt1=reshape(min(Mbt1,[],1),[nstates,height-1,width]);
    
    % update message from left to right
    Mlr1=zeros(nstates,height,width-1);
    Mlr1(:,:,2:end)=Mlr1(:,:,2:end)+Mlr(:,:,1:end-1);
    Mlr1(:,1:end-1,:)=Mlr1(:,1:end-1,:)+Mbt(:,:,1:end-1);
    Mlr1(:,2:end,:)=Mlr1(:,2:end,:)+Mtb(:,:,1:end-1);
    Mlr1=Mlr1+CO(:,:,1:end-1);
    Mlr1=kron(reshape(Mlr1,nstates,height*(width-1)),ones(1,nstates))+CMlr;
    Mlr1=reshape(min(Mlr1,[],1),[nstates,height,width-1]);
    
    % update message from right to left
    Mrl1=zeros(nstates,height,width-1);
    Mrl1(:,:,1:end-1)=Mrl1(:,:,1:end-1)+Mrl(:,:,2:end);
    Mrl1(:,1:end-1,:)=Mrl1(:,1:end-1,:)+Mbt(:,:,2:end);
    Mrl1(:,2:end,:)=Mrl1(:,2:end,:)+Mtb(:,:,2:end);
    Mrl1=Mrl1+CO(:,:,2:end);
    Mrl1=kron(reshape(Mrl1,nstates,height*(width-1)),ones(1,nstates))+CMrl;
    Mrl1=reshape(min(Mrl1,[],1),[nstates,height,width-1]);
    
    % reassign message
    Mtb=Mtb1*alpha+Mtb*(1-alpha);
    Mbt=Mbt1*alpha+Mbt*(1-alpha);
    Mlr=Mlr1*alpha+Mlr*(1-alpha);
    Mrl=Mrl1*alpha+Mrl*(1-alpha);
    
    % Bayesian MAP inference
    M=zeros(nstates,height,width);
    M(:,2:end,:)=M(:,2:end,:)+Mtb;
    M(:,1:end-1,:)=M(:,1:end-1,:)+Mbt;
    M(:,:,2:end)=M(:,:,2:end)+Mlr;
    M(:,:,1:end-1)=M(:,:,1:end-1)+Mrl;
    M=M+CO;
    [foo,IDX]=min(M,[],1);
    En(i+1)=imgraphen(IDX,CO,CM_h,CM_v);
end

M=zeros(nstates,height,width);
M(:,2:end,:)=M(:,2:end,:)+Mtb;
M(:,1:end-1,:)=M(:,1:end-1,:)+Mbt;
M(:,:,2:end)=M(:,:,2:end)+Mlr;
M(:,:,1:end-1)=M(:,:,1:end-1)+Mrl;
M=M+CO;
[foo,IDX]=min(M,[],1);
IDX=squeeze(IDX);