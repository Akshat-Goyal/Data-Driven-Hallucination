function output = VirtualTimeLapse(flags)
% Controls the working flow of photo relighting
video_path=flags.video_path;
end_frame=flags.end_frame;
output_folder=flags.output_folder;
input_image=imread(flags.input_image_name);

if flags.resize_input_image
    height_width_ratio=size(input_image,1)/size(input_image,2);
    input_image=imresize(input_image, ...
        flags.input_image_width*[height_width_ratio  1]);
end

tic; fprintf('Reading example video: %s ...', video_path);
try
    examplar_video=VideoReader(video_path);
    % examplar_video
catch err
    fprintf('Fatal Error zzz: cannot read %s\n', video_path);
    display(err);
end
t=toc; fprintf('Done.  t=%3.3f\n', t);

[correspondence best_manifold type example_match_box]=MatchImageToVideo(input_image, examplar_video,flags);
best_manifold=double(best_manifold);
input_image=double(input_image);

input_image_unfilter=input_image;
input_image(:,:,1) = bilateralFilter(input_image(:,:,1));
input_image(:,:,2) = bilateralFilter(input_image(:,:,2));
input_image(:,:,3) = bilateralFilter(input_image(:,:,3));
detail = input_image_unfilter - input_image;

if end_frame > examplar_video.NumberOfFrames
    error('flags.end_frame (%d) > examplar_video.frame_num (%d)', end_frame, examplar_video.NumberOfFrames);
end

i=flags.output_frame_sample;
    if i == -1
        reference_image=best_manifold; %zero transfer
    else
        reference_image=double(read(examplar_video, i));
    end
    if flags.resize_example_image
        height_width_ratio=size(reference_image, 1)/size(reference_image, 2);
        reference_image=imresize(reference_image,flags.example_image_width*[height_width_ratio 1]);
    end
    reference_image=CropImage(reference_image, example_match_box);

    %% Color appearance transfer kernel
    transfer_field=regularized_linear_regression(input_image, correspondence, type, reference_image, best_manifold, flags );
    fprintf('Transfer_field at frame %d is rendered. \n', i);

    transfer_field = transfer_field + detail;
    if ~exist(output_folder, 'dir')
        mkdir(output_folder)
    end
    imwrite(uint8(reference_image), strcat(output_folder,'reference_image.jpg'),'jpg', 'Quality',100);
    imwrite(uint8(warpImg(reference_image, correspondence, input_image, flags)), strcat(output_folder,'warpReference_image.jpg'), 'jpg', 'Quality',100);
    %% Global affine
    a_global=noise_robust_regres(reshape(best_manifold, [],3), reshape(reference_image, [],3));
    global_vec= reshape(input_image, [],3)*a_global;
    o_global = reshape(global_vec, size(input_image));
    imwrite(uint8(transfer_field),strcat(output_folder,'output.jpg'), 'jpg', 'Quality',100);

imwrite(uint8(input_image), strcat(output_folder,'input_image.jpg'),  'jpg', 'Quality', 100);
imwrite(uint8(best_manifold),strcat(output_folder,'best_frame.jpg'),'jpg', 'Quality', 100);
output = true;