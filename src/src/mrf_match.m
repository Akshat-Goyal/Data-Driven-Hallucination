function [correspondence energy type image_decriptor best_frame_descriptor]...
                            =mrf_match(image, best_frame, flags) 
  % Compute the correspondence by volume-based MRF. 
  % The working flow is described as below 
  %   -First search for k-nearest patches
  %   -Then compute energy terms, including evidence terms and compatibility 
  %    terms
  %   -Finally return the correspondence by belief propagation 


  %addpath(fullfile(flags.helper_root, 'mrf_matlab', 'TimeLapse'));
  addpath(fullfile('libs',  'TimeLapse'));
  image=double(image);
  best_frame=double(best_frame);

  %%Add details as feature
  %image_laplacian = GetDetails(rgb2gray(image/255))*255 + 128;
  %best_frame_laplacian = GetDetails(rgb2gray(best_frame/255))*255 + 128;
  %image = cat(3, image, 5*image_laplacian);
  %best_frame = cat(3, best_frame, 5*best_frame_laplacian);

  % Get k candidate
  [candidates energy type] = k_candidates(image, best_frame, flags); 
  patch_size=flags.mrf.patch_size;
  overlap_size=flags.mrf.overlap_size;
  interval_size=flags.mrf.interval_size;

  %% Compute Energy terms
  [Patches,Mask] = extractPatches(image, patch_size,...
                                        overlap_size, patch_size);
  CM_h=[];
  CM_v=[];
  best_frame_patches = im2patchesN(best_frame, patch_size, interval_size);
  %% Evidence terms
  CO = ComputeEvidence(Patches, candidates, best_frame_patches, flags.mrf);
  total_frame_num=flags.mrf_match.total_frame_num;
  examplar_video=VideoReader([flags.tlv_db flags.tlv_name]);
  sample_rate=examplar_video.NumberOfFrames/total_frame_num;

  %% Compatibility terms. 
  sequential=false;
  if sequential
    for k=1:total_frame_num
      frame_id=max(floor(sample_rate*k),1);
      frame=read(examplar_video, frame_id);
      if flags.resize_example_image
        height_width_ratio=size(frame, 1)/size(frame, 2);
        frame=imresize(frame,...
          flags.example_image_width*[height_width_ratio 1]);
      end
      all_patches = im2patches3(frame, patch_size, interval_size);
      [CM_h_t CM_v_t] = ComputeCompatibility(all_patches, candidates, flags.mrf);
      if isempty(CM_h)
        CM_h = CM_h_t;
        CM_v = CM_v_t;
      else
        CM_h = CM_h + CM_h_t;
        CM_v = CM_v + CM_v_t;
      end
      fprintf('Constructing compatibility. Progress: %d/%d \r', k, total_frame_num);
    end
    CM_h=CM_h/total_frame_num;
    CM_v=CM_v/total_frame_num;
  else
    frames=[];
    for k=1:total_frame_num
      frame_id=max(floor(sample_rate*k),1);
      frame=read(examplar_video, frame_id);
      if flags.resize_example_image
        height_width_ratio=size(frame, 1)/size(frame, 2);
        frame=imresize(frame,...
          flags.example_image_width*[height_width_ratio 1]);
      end
      frames(:,:,:,k)=frame;
      fprintf('Read frames.... Progress: %d/%d  \r', k, total_frame_num);
    end

    serial=parfor_progress('init', total_frame_num);
    parfor k=1:total_frame_num
      parfor_progress('count', serial, flags.verbose);
      frame=frames(:,:,:,k);
      all_patches = im2patches3(frame, patch_size, interval_size);
      [CM_h_t(:,:,:,:,k) CM_v_t(:,:,:,:,k)] = ...
                ComputeCompatibility(all_patches, candidates, flags.mrf);
    end
    parfor_progress('delete', serial);
    num_norm=flags.mrf.num_norm;
    if num_norm == Inf 
      CM_h = max(CM_h_t, [], 5).^2;
      CM_v = max(CM_v_t, [], 5).^2;
    else
      CM_h = sum(CM_h_t,5).^(2/num_norm)/total_frame_num;
      CM_v = sum(CM_v_t,5).^(2/num_norm)/total_frame_num;
    end
  end

  %% Run belief propagation 
  alpha = 3;
  num_iterations=flags.mrf.belief_propagation.num_iterations;
  [IDX,En] = immaxproduct(CO,CM_h*alpha,CM_v*alpha,num_iterations,0.5);

  [dim,h,w]=size(Patches);
  for i=1:h
    for j=1:w
      candidates(i,j).optimal_idx=IDX(i,j);
    end
  end
  correspondence=candidates;

% Utilities
function im_laplacian = GetDetails(im)
[height,width,nchannels]=size(im);

    % downsample 
    im_Low = imresize(imfilter(im,fspecial('gaussian',25,1),'same','replicate'),0.25,'bicubic');

    % upsample
    im_low = imresize(im_Low,[height,width],'bicubic');

    % obtain laplacian
    im_laplacian = im - im_low;
