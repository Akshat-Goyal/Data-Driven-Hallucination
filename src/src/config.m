function flags = exp_001_002(default_flags)

% flags.input_image_name = 'input_images/singapore.jpg';
% flags.video_path = '../../videos_h264/RHYTHM_OF_THE_CITY___SEATTLE_AT_NIGHT.mp4';
% flags.end_frame = 179;
% flags.output_frame_sample=178;
% flags.input_image_width=480;
% flags.output_folder='./results/blue_singapore/';

flags.input_image_name = '../input_images/singapore_25.jpg';
flags.video_path = '../videos/dubai.avi';
flags.end_frame = 1512;
flags.output_frame_sample=387;
flags.input_image_width=480;
flags.output_folder='../results/demo_singapore/';

% flags.input_image_name = 'input_images/yash_original.jpg';
% flags.video_path = '../../videos_h264/24_hour_time_lapse__2_.mp4';
% flags.end_frame = 8160;
% flags.output_frame_sample=1960;
% flags.input_image_width=420;
% flags.output_folder='./results/black_yash/';

% flags.input_image_name = 'input_images/iiit3.jpeg';
% flags.video_path = '../../videos_h264/Miraflores_Timelapse___Lima__Peru__01_2012__1080p_HD.mp4';
% flags.end_frame = 2601;
% flags.output_frame_sample=2000;
% flags.input_image_width=300;
% flags.output_folder='./results/demo_iiit/';

% flags.input_image_name = 'input_images/felicity.jpeg';
% flags.video_path = '../../videos_h264/1170691376.mp4';
% flags.end_frame = 9335;
% flags.output_frame_sample=7859;
% flags.input_image_width=300;
% flags.output_folder='./results/demo_feicity/';

% flags.input_image_name = 'input_images/paris.jpg';
% flags.video_path = '../../videos_h264/911_2012_Timelapse.mp4';
% flags.end_frame = 4349;
% flags.output_frame_sample=5;
% flags.input_image_width=350;
% flags.output_folder='./results/demo_paris/';

% flags.input_image_name = 'input_images/hampi_4.jpg';
% flags.video_path = '../../videos_h264/dubai.avi';
% flags.end_frame = 1889;
% flags.output_frame_sample=200;
% flags.input_image_width=320;
% flags.output_folder='./results/demo_hampi/';

%% Hard-coded default flags
flags.resize_input_image=true;
flags.resize_example_image=true;
flags.example_image_width=flags.input_image_width;
flags.make_video=false;
flags.mrf_match.total_frame_num=50;
flags.k_best_patches.k=1;
flags.k_candidates.patch_size=5; % this is half patchsize
flags.k_candidates.overlap_size=4;
flags.k_candidates.NN=30;
flags.k_candidates.patchDimL=11; % this is the full patch size  (2*patch_size+1)
flags.k_candidates.sigma=8;
flags.k_candidates.half_band=2;

% ------------------------
flags.mrf.patch_size=5;
flags.mrf.overlap_size=4;
flags.mrf.patchDimL=11;
flags.mrf.NN=30;
flags.mrf.num_norm=Inf;
% ------------------------
flags.mrf.solver='belief_propagation';
flags.mrf.belief_propagation.num_iterations=20;
flags.mrf.belief_propagation.alpha=0.5;
flags.mrf.visualization='merge';

flags.regularized_linear_regression.patch_size=5;
flags.regularized_linear_regression.epsilon=0.008;
flags.regularized_linear_regression.gamma=1;
