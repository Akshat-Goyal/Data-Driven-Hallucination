function [correspondence best_frame type match_box image_descriptor...p
    best_frame_descriptor] = MatchImageToVideo(image,video,flags)
%  Finds the dense correspondence between the imageand the target


[best_frame match_box]=best_distribution_frame(image, video, flags);

fprintf('Doing domain adaptation...');
image=TransferColorByExample(double(image), double(best_frame),'LAB');
image = uint8(image);
fprintf('done.\n');
[correspondence energy type]=mrf_match(image, best_frame, flags);

function [best_frame, match_box] = best_distribution_frame(image, video, flags)
fprintf('Total Frames in reference video %d\n',video.NumberOfFrames);
tic; fprintf('Searching for the best frame...');
% Searching along time domain
sampling_rate=25;
image_descriptor=[imhist(image(:,:,1)); imhist(image(:,:,2)); imhist(image(:,:,3))]/numel(image);

total_samples = floor(video.NumberOfFrames/sampling_rate);
for i = 1 : total_samples
    fprintf('Checking sample %d/%d \r',i,total_samples)
    frame = read( video , i*sampling_rate);
    if flags.resize_example_image
        height_width_ratio=size(frame, 1)/size(frame, 2);
        frame=imresize(frame,...
            flags.example_image_width*[height_width_ratio 1]);
    end
    frame_descriptor=[imhist(frame(:,:,1)); imhist(frame(:,:,2)); imhist(frame(:,:,3))]/numel(image);
    dist(i) = norm(image_descriptor - frame_descriptor);
end

[minDist best_n] = find(dist==min(dist));
best_n = best_n*sampling_rate;

t=toc; fprintf('Done. The best matching: %d th frame. t=%2.2f\n', best_n, t);
best_frame=read(video, best_n);
if flags.resize_example_image
    height_width_ratio=size(best_frame, 1)/size(best_frame, 2);
    best_frame=imresize(best_frame,...
        flags.example_image_width*[height_width_ratio 1]);
end
match_box.ul_corner=[1 1];
match_box.width=size(best_frame, 2);
match_box.height=size(best_frame, 1);
