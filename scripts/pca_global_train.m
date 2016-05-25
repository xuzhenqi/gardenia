function [W, mean_shape] = pca_global_train(label, path)
%% label shape is a textfile without filenames.

train_data = load(label);
train_data = train_data + 1; % index start from 1
N = size(train_data, 1);

[train_data, mean_shape] = align_shapes(train_data);
[W, ~, ~, ~, ~, ~] = pca(reshape(train_data, [N, 136]));
if nargin > 1
    save(path, 'W', 'mean_shape');
end

end