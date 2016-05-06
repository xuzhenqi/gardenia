% crop the trainset
% input: ../data/
% output: ../data/test.txt ../data/trans_param.txt ../data/crop/
clear;
clc;
input = '../data/';
dst = '../data/crop/';
output_file = '../data/test.txt';
output_param = '../data/trans_param.txt';
source = fopen(output_file, 'w');
param = fopen(output_param, 'w');

scale = 1.15; % scale > 1
sz = [224 224];

subsets = {'helen/testset/', 'lfpw/testset/', 'ibug/'};
subsets_bbx = {'helen_testset', 'lfpw_testset', 'ibug'};

for is = 1:length(subsets)
    load([input 'bounding_boxes/bounding_boxes_' subsets_bbx{is} '.mat']);
    % bounding_boxes
    root = [input subsets{is}];
    
    for i = 1:length(bounding_boxes)
        imgName = bounding_boxes{i}.imgName;
%         bbx_g = bounding_boxes{i}.bb_ground_truth;
        bbx_g = bounding_boxes{i}.bb_detector;
        if ~exist([root imgName], 'file');
            fprintf('%s%s not exists\n', root, imgName);
            continue;
        end
        fprintf('%s%s\n', root, imgName);
        im = imread([root imgName]);

        % new bbx
        width = bbx_g(3) - bbx_g(1);
        height = bbx_g(4) - bbx_g(2);
        x_new = round(max(bbx_g(1) - (scale - 1) * width, 1));
        y_new = round(max(bbx_g(2) - (scale - 1) * height, 1));
        xe_new = round(min(bbx_g(3) + (scale - 1) * width, size(im, 2)));
        ye_new = round(min(bbx_g(4) + (scale - 1) * height, size(im, 1)));

        im_new = im(y_new:ye_new, x_new:xe_new, :);
        im_new = imresize(im_new, sz);
        scale_height = (ye_new - y_new)/size(im_new, 1);
        scale_width = (xe_new - x_new)/size(im_new, 2);

        imwrite(im_new, [dst subsets{is} imgName]);

        % label
        labelFile = [root imgName(1:end-4) '.pts'];
        f = fopen(labelFile);
        fgetl(f); fgetl(f); fgetl(f);
        label = cell2mat(textscan(f, '%f %f', 68));
        fclose(f);

        label(:,2) = (label(:,2) - y_new) / scale_height;
        label(:,1) = (label(:,1) - x_new) / scale_width;
        fprintf(source, '%s', [subsets{is} imgName]);
        for j = 1:size(label, 1)
            fprintf(source, ' %f %f', label(j, 1), label(j, 2));
        end
        fprintf(source, '\n');
        fprintf(param, '%s %f %f %f %f\n', [subsets{is} imgName], x_new, y_new, ...
            scale_width, scale_height);
    end
end
fclose(source);
fclose(param);
