function CO=ComputeEvidence(Patches, candidates, all_lowres, flags)
% Compute compatibility function
fprintf('Constructing data term function...');
NN=flags.NN;
overlapSize=flags.overlap_size;
patchDim=flags.patchDimL;
[h w]=size(candidates);
CO = zeros(NN,h,w);
for i=1:h
    for j=1:w
        lowres = all_lowres(candidates(i,j).idx,:);
        CO(:,i,j) = sum((repmat(Patches(:,i,j)',[NN,1])-lowres).^2,2);
    end
end
fprintf('done!\n');