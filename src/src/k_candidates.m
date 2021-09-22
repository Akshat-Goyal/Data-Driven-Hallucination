function [correspondence energy type image_descriptor best_frame_descriptor] = k_candidates(image, best_frame, flags)
%% Find  k-nearest neighbors.

addpath(fullfile('utils',  'TimeLapse'));
image=double(image);
best_frame=double(best_frame);
patch_size=flags.k_candidates.patch_size;
overlap_size=flags.k_candidates.overlap_size;

best_frame_patches = im2patchesN(best_frame, patch_size);

% Search for candidates
[Patches,Mask]=extractPatches(image, patch_size, overlap_size, patch_size);
[dim,h,w]=size(Patches);
candidates = SearchCandidates(Patches, Mask, best_frame_patches,best_frame, flags.k_candidates);
for i=1:h
    for j=1:w
        candidates(i,j).optimal_idx=1;
    end
end
correspondence=candidates;
type='patch_candidates';
energy=[];
