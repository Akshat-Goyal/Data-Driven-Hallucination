function transfer_field=regularized_linear_regression(input_image, correspondence,type, reference_frame, best_frame, flags)
% Transforms the input image by finding an affine transformation function between the best_frame and reference_frame onto input image.
example_field=warpImg(reference_frame,correspondence,input_image,flags);
warped_best_frame=warpImg(best_frame,correspondence,input_image,flags);
transfer_field=LocalRegression(input_image, warped_best_frame, example_field, flags.regularized_linear_regression);
transfer_field = reshape(transfer_field, size(example_field));
end
