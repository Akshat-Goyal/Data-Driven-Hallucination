function patches=im2patchesN(image, patch_size)
[h w num_dims]=size(image);
interval_size = 1;
boundary_size = patch_size;
foo = im2patches(image(:,:,1), patch_size, 1, boundary_size);
[nh num_samples]=size(foo);
patches = zeros(nh, num_samples*num_dims);
for i = 1 : num_dims
    patch = im2patches(image(:,:,i), patch_size, 1, boundary_size);
    patches(:, 1 + (i-1)*num_samples:i*num_samples) = patch;
end

function patches = im2patches(im,patchSize,intervalSize,boundarySize)
[p_xx,p_yy]=meshgrid(-patchSize:patchSize,-patchSize:patchSize);
nDim = numel(p_xx);
[height,width]=size(im);
[grid_xx,grid_yy]=meshgrid(boundarySize+1:intervalSize:width-boundarySize,boundarySize+1:intervalSize:height-boundarySize);
grid_xx = grid_xx(:); grid_yy = grid_yy(:);
nPatches = numel(grid_xx);
xx = repmat(p_xx(:)',[nPatches,1]) + repmat(grid_xx(:),[1,nDim]);
yy = repmat(p_yy(:)',[nPatches,1]) + repmat(grid_yy(:),[1,nDim]);
index = sub2ind([height,width],yy(:),xx(:));
patches = reshape(im(index),[nPatches,nDim]);