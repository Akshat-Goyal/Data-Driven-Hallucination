function output_image=CropImage(input_image, bounding_box)
if isempty(bounding_box)
    error('bounding_box cannot be empty');
end
x=bounding_box.ul_corner(1);
y=bounding_box.ul_corner(2);
width=bounding_box.width;
height=bounding_box.height;
output_image=input_image(y : y + height - 1, x : x + width - 1, :);