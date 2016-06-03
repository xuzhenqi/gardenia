function [new_shape, a, b] = align_to_shape(shape, mean_shape)
% shape: matrix of size [2, 68], centered at origin.
% mean_shape: matrix of size [2, 68], centered at origin.
% new_shape: the aligned shape.
% a, b: parameter to perform similarity transformation.
% reference: Statistical Models of Appearance for Computer Vision. Appendix
% B.

shape = reshape(shape, [2, 68]);
mean_shape = reshape(mean_shape, [2, 68]);
xlen = dot(shape(:), shape(:));
a = dot(shape(:), mean_shape(:)) / xlen;
b = (dot(shape(1,:), mean_shape(2,:)) - dot(shape(2,:), ...
    mean_shape(1,:))) / xlen;
new_shape = [a, -b; b, a] * shape;
end