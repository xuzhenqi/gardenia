function new_label = rotate_label(label, angle, center, center_rot)
    dis = bsxfun(@minus, label, center);
%     [rho, leng] = cart2pol(dis(:,1), dis(:,2));
%     rho = rho - angle * pi / 180;
%     [X, Y] = pol2cart(rho, leng);
    rot = [cos(angle*pi/180), sin(angle*pi/180); -sin(angle*pi/180), cos(angle*pi/180)];
    dis = dis * rot';
    new_label = bsxfun(@plus, dis, center_rot);
    save('rot2', 'dis', 'rot', 'center', 'center_rot', 'new_label'); 
end