function new_bbx = bbx_scale(bbx, img_size, scale)
% scale > 1
    if (scale <= 1)
        error('scale must be greater than 1');
    end
    width = bbx(3) - bbx(1);
    height = bbx(4) - bbx(2);
    x_new = max(round(bbx(1) - (scale-1)*width), 1);
    y_new = max(round(bbx(2) - (scale-1)*height), 1);
    xe_new = min(round(bbx(3) + (scale-1)*width), img_size(2));
    ye_new = min(round(bbx(4) + (scale-1)*height), img_size(1));
    new_bbx = [x_new, y_new, xe_new, ye_new];
end