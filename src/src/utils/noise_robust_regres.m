function a = noise_robust_regres(input_vec, output_vec)
rgb2yuv_mat =   [0.299    0.587    0.114
    -0.14713 -0.28886 0.436
    0.615    -0.51498 -0.10001];
yuv2rgb_mat =   [1      0      1.13983
    1   -0.39465  -0.58060
    1    2.03211     0    ];
input_vec  = input_vec*(rgb2yuv_mat');
output_vec = output_vec*(rgb2yuv_mat');
sigma_y  = input_vec(:,1)'*input_vec(:,1);
a_y=[(inv(sigma_y))*(input_vec(:,1)'*output_vec(:,1)) ;0 ;0];
sigma = input_vec'*input_vec;
a_uv=[(inv(sigma))*(input_vec'*output_vec(:,2:3))];
a = (rgb2yuv_mat')*[a_y a_uv]*(yuv2rgb_mat');
