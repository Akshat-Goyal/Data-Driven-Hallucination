function candidates=SearchCandidates(Patches, Mask, dictionary_lowres, im_best_manifolds, flags)
% Setup your opencv mex here
addpath('~/cv/mexopencv');
addpath('~/cv/mexopencv/opencv_contrib');
import('cv.*');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

NN = 30;
patchDimL=11;
sigma=flags.sigma;
[~,~,num_channel,~]=size(im_best_manifolds);
[dim,h,w]=size(Patches);
nSamples=size(dictionary_lowres,1);
for i=1:h
    for j=1:w
        candidates(i,j).idx=zeros(NN,1);
        candidates(i,j).patches=zeros(num_channel*patchDimL^2,1);
    end
end
tree=[];
dictionary_mean=[];
dictionary_S=[];

fprintf('Constructing Markov Network:\n');
ta=tic;
for i=1:h
    idx=[];
    for j=1:w
        Dist=[];
        patch     = Patches(:,i,j);
        patch     = reshape(patch,[patchDimL, patchDimL, num_channel]);
        cost_map = cv.matchTemplate(im_best_manifolds(:,:,:,1), patch, 'Method', 'SqDiff');
        
        query_y_coordinate = i/h;
        lower_pass_bound = query_y_coordinate - flags.half_band;
        upper_pass_bound = query_y_coordinate + flags.half_band;
        cost_map_height=size(cost_map,1);
        penalty = cost_map*0;
        if lower_pass_bound > 0
            penalty(1:1+ceil(lower_pass_bound*cost_map_height),:) = Inf;
        end
        if upper_pass_bound < 1
            penalty(ceil(upper_pass_bound*cost_map_height):end,:) = Inf;
        end
        cost_map = cost_map + penalty;
        
        Dist(:,:,1) = cost_map;        
        Dist=Dist(:);
        sampled_idx=randsample(length(Dist), NN, true, exp(-Dist/2/patchDimL^2/sigma^2));
        candidate_dist=Dist(sampled_idx);
        [~, candidate_order] = sort(candidate_dist);
        idx=sampled_idx(candidate_order);
        candidates(i,j).idx     = idx;
    end
    fprintf('Iteration: %f\n',i);
end
fprintf('Finished NN search. t=%f\n', toc(ta));