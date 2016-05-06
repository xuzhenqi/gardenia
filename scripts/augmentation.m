% augmentation the trainset
% input: ../data/
% output: ../data/train.txt ../data/augment/
clear;
clc;
input = '../data/';
dst = '../data/augment/';
output_file = '../data/train.txt';
source = fopen(output_file, 'w');


scales = [1.05, 1.15, 1.25];
angles = -30:5:30;
sz = [224 224];
subsets = {'afw/', 'lfpw/trainset/', 'helen/trainset/'};
subsets_bbx = {'afw', 'lfpw_trainset', 'helen_train_set'};
for is = 1:length(subsets)
    load([input 'bounding_boxes/bounding_boxes_' subsets_bbx{is} '.mat']); 
    % bounding_boxes
    root = [input subsets{is}];

    for id = 1:length(bounding_boxes)
        imgName = bounding_boxes{id}.imgName;
        bbx_g = bounding_boxes{id}.bb_ground_truth;
        if ~exist([root imgName], 'file');
            fprintf('%s%s not exists\n', root, imgName);
            continue;
        end
        im = imread([root imgName]);
        % h = imshow(im);
        % rectangle('Position', [bbx_g(1), bbx_g(2), bbx_g(3)-bbx_g(1), ...
        %     bbx_g(4)-bbx_g(2)], 'EdgeColor', 'red');
        labelFile = [root imgName(1:end-4) '.pts'];
        f = fopen(labelFile);
        fgetl(f);fgetl(f);fgetl(f);
        label = cell2mat(textscan(f, '%f %f', 68));
        fclose(f);

    %     hold on;
    %     plot(new_label(:,1), new_label(:,2), '*', 'MarkerEdgeColor','cyan','MarkerSize',8);
    %     hold off;

        % new bbx
        new_bbx = bbx_scale(bbx_g, size(im), 1.8);
        im_crop = im(new_bbx(2):new_bbx(4), new_bbx(1):new_bbx(3), :);
        new_label = label - repmat([new_bbx(1), new_bbx(2)], [size(label, 1), 1]) + 1;
        for an = angles
            im_rotate = imrotate(im_crop, an, 'bilinear');
            center = (1+[size(im_crop, 2), size(im_crop, 1)])/2;
            center_rot = (1+[size(im_rotate, 2), size(im_rotate, 1)])/2;
            rot_label = rotate_label(new_label, an, center, center_rot);

            bbx = [min(rot_label) max(rot_label)];
            % rectangle('Position', [bbx(1), bbx(2), bbx(3)-bbx(1), ...
            %     bbx(4)-bbx(2)], 'EdgeColor', 'red');
            for i = 1:length(scales)
                new_bbx = bbx_scale(bbx, size(im_rotate), scales(i));
                % rectangle('Position', [new_bbx(1), new_bbx(2), new_bbx(3)-new_bbx(1), ...
                %     new_bbx(4)-new_bbx(2)], 'EdgeColor', 'red');
                im_new = imresize(im_rotate(new_bbx(2):new_bbx(4), new_bbx(1):new_bbx(3), :), sz);
                scale_height = (new_bbx(4) - new_bbx(2) + 1)/size(im_new, 1);
                scale_width = (new_bbx(3) - new_bbx(1) + 1)/size(im_new, 2);
                im_new_label(:,2) = (rot_label(:,2) - new_bbx(2) + 1) / scale_height;
                im_new_label(:,1) = (rot_label(:,1) - new_bbx(1) + 1) / scale_width;
                filename = [dst subsets{is} imgName(1:end-4) '_' sprintf('%.2f', scales(i)) '_' int2str(an) '.png'];
                imwrite(im_new, filename);
                fprintf(source, '%s', [subsets{i} filename]);
                for j = 1:size(im_new_label, 1)
                    fprintf(source, ' %f %f', im_new_label(j, 1), im_new_label(j, 2));
                end
                fprintf(source, '\n');
            end
        end
    end
end
fclose(source);