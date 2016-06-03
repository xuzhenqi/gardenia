function [new_set, mean_shape] = align_shapes(train_set, verbose)
% train_set: matrix of size [M, 136];
% verbose: 1. draw shapes. 0. Not to draw shapes;
% new_set: matrix of size [M, 136];
% mean_shape: vector of size 136;

if nargin < 2
    verbose = 0;
end

M = size(train_set, 1);
train_set = reshape(train_set, [M, 2, 68]);
%% Make examples centered at origin.
for i = 1:M
    train_set(i, :, :) = train_set(i, :, :) - ...
        repmat(mean(train_set(i, :, :), 3), [1, 1, 68]);
end

%% Alignment the shapes.
mean_shape = mean(train_set, 1);
mean_shape = mean_shape / norm(mean_shape(:));
old_shape = ones(size(mean_shape));

delta = norm(old_shape(:) - mean_shape(:));
while delta > 1e-5
    fprintf('delta: %f\n', delta);
    draw_shape(mean_shape);
    for i = 1:M
        %draw_shape(train_set(i, :, :));
        train_set(i, :, :) = align_to_shape(train_set(i, :, :), ...
            mean_shape);
        %draw_shape(train_set(i, :, :));
    end
    old_shape = mean_shape;
    mean_shape = mean(train_set, 1);
    mean_shape = mean_shape / norm(mean_shape(:));
    delta = norm(old_shape(:) - mean_shape(:));
end

new_set = train_set;

end