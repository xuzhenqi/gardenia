function test_dataset_list(root, filelist)
f = fopen(filelist);
data = textscan(f, ['%s', repmat(' %f', 1, 136)]);
fclose(f);
filenames = data{1};
shapes = cell2mat(data(2:end));
rng = randi([1, length(filenames)], [1, 10]);

for i = rng
    imshow([root filenames{i}]);
    shape = reshape(shapes(i, :), 2, []);
    hold on;
    plot(shape(1,:), shape(2,:), '.g');
    hold off;
    pause;
end

end