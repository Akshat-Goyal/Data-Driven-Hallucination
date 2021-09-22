function output=LocalRegression(input_image, example_input_image, example_output, flags)

patch_size=flags.patch_size;
epsilon=flags.epsilon;
gamma=flags.gamma;
radius=(patch_size-1)/2;
[height width num_channels]=size(input_image);
a_global = getGlobalRegression(example_input_image, example_output);
fprintf('Global matrix:\n');
a_global

fprintf('Compute local variance...');tic;
j_interval = [1:width-patch_size+1];
compact_table = lookup_table(patch_size);
sigmas=getScalingFactors(input_image, example_input_image, radius, patch_size, epsilon, gamma);
affinity_matrix_compact=zeros((2*patch_size-1)^2, height*(length(j_interval)+patch_size-1));
guidance = zeros(height*(length(j_interval)+patch_size-1), num_channels);
global_inds = global_indexes(patch_size, height, width);
I_3=eye(3);
I_k=eye(patch_size^2);

for j = j_interval
    ul_corners = [1 : (height-patch_size+1)] + (j-j_interval(1))*(height-patch_size+1);
    iis        = global_inds(:,ul_corners);
    iiis = zeros(patch_size^4, height - patch_size + 1);
    for m = 1 : patch_size^2
        iiis(1+(m-1)*patch_size^2:m*patch_size^2, :) = repmat(iis(m,:),[patch_size^2 1]);
    end
    linear_ids = repmat(compact_table(:), [1 height-patch_size+1]) + (2*patch_size-1)^2*(iiis-1);

    for i = 1 : height - patch_size + 1
        linear_index = i + (j-1)*(height-patch_size+1);
        sigma = sigmas(:,:,linear_index);
        x=input_image(i:i+patch_size-1,j:j+patch_size-1,:);
        x=reshape(x,patch_size^2, 3);
        x0=example_input_image(i:i+patch_size-1,j:j+patch_size-1,:);
        x0=reshape(x0,patch_size^2, 3);
        y0=example_output(i:i+patch_size-1,j:j+patch_size-1,:);
        y0=reshape(y0,patch_size^2, 3);

        M = inv(sigma);
        local_laplacian = I_k - x*M*x';
        local_guidance  = x*M*(epsilon*x0'*y0 + gamma*a_global);
        affinity_matrix_compact(linear_ids(:,i))= affinity_matrix_compact(linear_ids(:,i)) + local_laplacian(:);
        guidance(iis(:,i),1) = guidance(iis(:,i),1) +  local_guidance(:,1);
        guidance(iis(:,i),2) = guidance(iis(:,i),2) +  local_guidance(:,2);
        guidance(iis(:,i),3) = guidance(iis(:,i),3) +  local_guidance(:,3);
    end
end
t=toc; fprintf('Done. t=%3.3f\n', t);
affinity_matrix_gather = zeros((2*patch_size-1)^2, height*width);
guidance_gather = zeros(height*width, num_channels);

actual_interval=[1+height*(j_interval(1)-1): height*(j_interval(end)+patch_size-1)];
guidance_gather(actual_interval',:)=guidance_gather(actual_interval',:) + guidance;
affinity_matrix_gather(:,actual_interval) = affinity_matrix_gather(:,actual_interval) + affinity_matrix_compact;
affinity_matrix_index_i = global_inds_from_local(patch_size, height, width);
affinity_matrix_index_j = repmat([1:height*width], [(2*patch_size-1)^2 1]);

iis = affinity_matrix_index_i(affinity_matrix_index_i~=0);
jjs = affinity_matrix_index_j(affinity_matrix_index_i~=0);
vals= affinity_matrix_gather(affinity_matrix_index_i~=0);
laplacian=sparse(iis(:), jjs(:), vals(:), height*width, height*width);
output = laplacian\ guidance;

function a_global=getGlobalRegression(input, output)
	[h_e w_e num_channel_e] = size(output);
	[h_i w_i num_channel_i] = size(input);
	example_input_vec    = reshape(input, h_i*w_i, num_channel_i);
	example_output_vec   = reshape(output, h_e*w_e, num_channel_e);
	% Global regres prior
	sigma = example_input_vec'*example_input_vec;
	a_global= inv(sigma)*(example_input_vec'*example_output_vec);

function global_inds=global_indexes(patch_size, height, width)

% Elements in the same column as the patch origin (upper left)
same_col = repmat([0:patch_size-1]', [patch_size 1]);

diff_col = repmat([0:patch_size-1], [patch_size 1]);
% Index offset relative to the patch origin.
relative_gain =  same_col(:) + height*diff_col(:);
foo = reshape([1:height*width], [height width]);
origins = foo(1:height-patch_size+1, 1: width-patch_size+1);
global_inds = bsxfun(@plus, origins(:)', relative_gain);

function image_vector=getReshapeImage(input_image, global_inds)
input_r = input_image(:,:,1);
input_g = input_image(:,:,2);
input_b = input_image(:,:,3);
num_pix      = size(global_inds,1);
num_patches  = size(global_inds,2);
r_vec   = reshape(input_r(global_inds), [num_pix 1 num_patches]);
g_vec   = reshape(input_g(global_inds), [num_pix 1 num_patches]);
b_vec   = reshape(input_b(global_inds), [num_pix 1 num_patches]);
image_vector = cat(2, r_vec, g_vec, b_vec);

function [rr, rg, rb, gg, gb, bb]=variance_filter(I, radius)
% r is the radius, so the patch size is 2*r+1
[height width num_channels]=size(I);
if num_channels~=3
    error('Only support RGB images (num_channels=3) now');
end
rr=boxfilter(I(:,:,1).*I(:,:,1), radius);
rg=boxfilter(I(:,:,1).*I(:,:,2), radius);
rb=boxfilter(I(:,:,1).*I(:,:,3), radius);
gg=boxfilter(I(:,:,2).*I(:,:,2), radius);
gb=boxfilter(I(:,:,2).*I(:,:,3), radius);
bb=boxfilter(I(:,:,3).*I(:,:,3), radius);

function sigmas=getScalingFactors(input_image, example_input_image,radius, patch_size, epsilon, gamma)
[rr_i rg_i rb_i gg_i gb_i bb_i]=variance_filter(input_image, radius);
[rr_e rg_e rb_e gg_e gb_e bb_e]=variance_filter(example_input_image, radius);
rr = rr_i + epsilon*rr_e + gamma;
gg = gg_i + epsilon*gg_e + gamma;
bb = bb_i + epsilon*bb_e + gamma;
rg = rg_i + epsilon*rg_e;
rb = rb_i + epsilon*rb_e;
gb = gb_i + epsilon*gb_e;
[height width num_channels]=size(input_image);
num_elements=(height-patch_size+1)*(width-patch_size+1);
i_interval = [1+radius:height-patch_size+1+radius];
j_interval = [1+radius:width-patch_size+1+radius];

rr_sub_vec = reshape(rr(i_interval, j_interval), [1 1 num_elements]);
gg_sub_vec = reshape(gg(i_interval, j_interval), [1 1 num_elements]);
bb_sub_vec = reshape(bb(i_interval, j_interval), [1 1 num_elements]);
rg_sub_vec = reshape(rg(i_interval, j_interval), [1 1 num_elements]);
rb_sub_vec = reshape(rb(i_interval, j_interval), [1 1 num_elements]);
gb_sub_vec = reshape(gb(i_interval, j_interval), [1 1 num_elements]);
sigmas = [rr_sub_vec rg_sub_vec rb_sub_vec; rg_sub_vec gg_sub_vec gb_sub_vec; rb_sub_vec gb_sub_vec bb_sub_vec];

function table = lookup_table(patch_size)
table=zeros(patch_size^2, patch_size^2);
global_index = [1:(2*patch_size-1)^2];
global_index = reshape(global_index, [2*patch_size-1  2*patch_size-1]);
for i = 1 : patch_size
    for j = 1 : patch_size
        ids = global_index( patch_size - i + 1 : 2*patch_size - i, ...
            patch_size - j + 1 : 2*patch_size - j);
        table(:, (j-1)*patch_size + i ) = ids(:);
    end
end

function global_ids = global_inds_from_local(patch_size, height, width)
[oxx oyy] = meshgrid(1:width, 1:height);
[dxx dyy] = meshgrid(-patch_size+1:patch_size-1, -patch_size+1:patch_size-1);
xx = bsxfun(@plus, oxx(:)' , dxx(:));
yy = bsxfun(@plus, oyy(:)' , dyy(:));
global_ids = (xx-1)*height + yy;
global_ids(xx<=0)=0;
global_ids(yy<=0)=0;
global_ids(xx>width)=0;
global_ids(yy>height)=0;

function imDst = boxfilter(imSrc, r)
%   BOXFILTER   O(1) time box filtering using cumulative sum
[hei, wid] = size(imSrc);
imDst = zeros(size(imSrc));
imCum = cumsum(imSrc, 1);
imDst(1:r+1, :) = imCum(1+r:2*r+1, :);
imDst(r+2:hei-r, :) = imCum(2*r+2:hei, :) - imCum(1:hei-2*r-1, :);
imDst(hei-r+1:hei, :) = repmat(imCum(hei, :), [r, 1]) - imCum(hei-2*r:hei-r-1, :);
%cumulative sum over X axis
imCum = cumsum(imDst, 2);
%difference over Y axis
imDst(:, 1:r+1) = imCum(:, 1+r:2*r+1);
imDst(:, r+2:wid-r) = imCum(:, 2*r+2:wid) - imCum(:, 1:wid-2*r-1);
imDst(:, wid-r+1:wid) = repmat(imCum(:, wid), [1, r]) - imCum(:, wid-2*r:wid-r-1);
