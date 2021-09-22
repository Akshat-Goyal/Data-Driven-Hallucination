function [warpI2]=warpImg(reference_frame, warp_field, input_image, flags)
warpI2 = merge_patches(input_image, warp_field,reference_frame, flags);

function warped_reference_frame=merge_patches(input_image, correspondence, reference_frame, flags)
  candidates=correspondence;
  patch_size=flags.mrf.patch_size;
  overlap_size=flags.mrf.overlap_size;
  reference_patches = im2patchesN(reference_frame, patch_size);
  [Patches,Mask] = extractPatches(input_image, patch_size, overlap_size, patch_size);
  [dim,h,w]=size(Patches);
  for i=1:h
    for j=1:w
      idx = candidates(i,j).idx;
      candidates(i,j).patches = reference_patches(idx,:);
    end
  end
  for i=1:h
    for j=1:w
      optimal_idx=candidates(i,j).optimal_idx;
      patches_from_candidates(:,i,j) = candidates(i,j).patches(optimal_idx,:)';
    end
  end
  [height,width,num_channel]=size(input_image);
  warped_reference_frame = mergePatches(patches_from_candidates, overlap_size, width, height);