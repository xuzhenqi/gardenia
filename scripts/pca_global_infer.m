function new_shape = pca_global_infer(pred, W, mean_shape, t, path)
%%% pred is a text file without filenames
test_data = load(pred);
test_data = test_data + 1;
K = size(test_data, 1);
test_data_align = reshape(test_data, [K, 2, 68]);
a = zeros(K, 1);
b = zeros(K, 1);
test_data_mean = zeros(2, K);
for i = 1:K
    %draw_shape(test_data_align(i, :, :));
    test_data_mean(:, i) = mean(squeeze(test_data_align(i, :, :)), 2);
    [test_data_align(i, :, :), a(i), b(i)] = align_to_shape(...
        test_data_align(i, :, :) - reshape(repmat(test_data_mean(:, i), ...
        [1, 68]), [1, 2, 68]), mean_shape); 
    %draw_shape(test_data_align(i, :, :));
end

mu = mean_shape;
W_r = W(:, 1:t);
test_data_align = reshape(test_data_align, [K, 136]);
test_align_r = (test_data_align - repmat(mu, [K, 1])) * (W_r * W_r') + ...
    repmat(mu, [K, 1]);
test_align_r = reshape(test_align_r, [K, 2, 68]);

for i = 1 : K
    % translate directly
    ra = a(i) / (a(i)*a(i) + b(i)*b(i));
    rb = -b(i) / (a(i)*a(i) + b(i)*b(i));
    rm = test_data_mean(:, i);
    test_align_r(i, :, :) = [ra, -rb; rb ra] * ...
        squeeze(test_align_r(i, :, :)) + repmat(rm, [1, 68]);
    % draw_shape(test_align_r(i, :, :));
    % draw_shape(test_data(i, :, :));
    % draw_shape(test_label(i, :, :));
end
test_align_r = min(test_align_r, 224);
test_align_r = max(test_align_r, 0);
new_shape = reshape(test_align_r, [K, 136]);

if nargin > 4
    f = fopen(path, 'w');
    for i = 1:K
        for j = 1:136
            fprintf(f, '%f ', new_shape(i,j));
        end
        fprintf(f, '\n');
    end
end

end
