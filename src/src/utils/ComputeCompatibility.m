function [CM_h, CM_v]=ComputeCompatibility(all_patches, candidates,  flags)
NN=flags.NN;
overlapSize=flags.overlap_size;
patchDim=flags.patchDimL;
num_norm = flags.num_norm;

[h w]=size(candidates);
for i=1:h
    fprintf('Sub Iteration 1: %d/%d \r',i,h);
    for j=1:w
        idx = candidates(i,j).idx;
        candidates(i,j).patchesFull = double(all_patches(idx,:));
    end
end
CM_h = zeros([NN, NN, h, w-1]);
for i=1:h
    fprintf('Sub Iteration 2: %d/%d \r',i,h);
    for j=1:w-1
        foo1 = reshape(candidates(i,j).patchesFull',[patchDim,patchDim,3,NN]);
        foo2 = reshape(candidates(i,j+1).patchesFull',[patchDim,patchDim,3,NN]);
        foo1 = reshape(foo1(:,end-overlapSize+1:end,:,:),[3*patchDim*overlapSize,NN]);
        foo2 = reshape(foo2(:,1:overlapSize,:,:),[3*patchDim*overlapSize,NN]);
        foo1 = repmat(foo1(:),[1,NN]);
        foo2 = repmat(foo2,[NN,1]);
        if num_norm == 1 || num_norm == Inf
            CM_h(:,:,i,j) = reshape(sum(reshape(abs(foo1-foo2),[3*patchDim*overlapSize,NN^2]),1),[NN,NN]);
        else
            CM_h(:,:,i,j) = reshape(sum(reshape((foo1-foo2).^num_norm,[3*patchDim*overlapSize,NN^2]),1),[NN,NN]);
        end
    end
end

CM_v = zeros([NN, NN, h-1, w]);
for i=1:h-1
    fprintf('Sub Iteration 3: %d/%d \r',i,h);
    for j=1:w
        foo1 = reshape(candidates(i,j).patchesFull',[patchDim,patchDim,3,NN]);
        foo2 = reshape(candidates(i+1,j).patchesFull',[patchDim,patchDim,3,NN]);
        foo1 = reshape(foo1(end-overlapSize+1:end,:,:),[3*patchDim*overlapSize,NN]);
        foo2 = reshape(foo2(1:overlapSize,:,:),[3*patchDim*overlapSize,NN]);
        foo1 = repmat(foo1(:),[1,NN]);
        foo2 = repmat(foo2,[NN,1]);
        if num_norm == 1 || num_norm == Inf
            CM_v(:,:,i,j) = reshape(sum(reshape(abs(foo1-foo2),[3*patchDim*overlapSize,NN^2]),1),[NN,NN]);
        else
            CM_v(:,:,i,j) = reshape(sum(reshape((foo1-foo2).^num_norm,[3*patchDim*overlapSize,NN^2]),1),[NN,NN]);
        end
    end
end