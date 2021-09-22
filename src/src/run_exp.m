function run_exp(config)
addpath('utils');
flags=[];
fprintf('================================\n');
%% Read base config. files from configs/base/
ReadConfiguration=str2func(config);
flags=ReadConfiguration(flags);
[~,input_name,~]=fileparts(flags.input_image_name);
output=VirtualTimeLapse(flags);
fprintf('Done\n');