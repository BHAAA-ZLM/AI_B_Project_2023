function h_target = GetHist(target)
    h1_target = imhist(target(:,:,1));
    h2_target = imhist(target(:,:,2));
    h3_target = imhist(target(:,:,3));
    h1_target = h1_target / sum(h1_target);
    h2_target = h2_target / sum(h2_target);
    h3_target = h3_target / sum(h3_target);
    h_target = [h1_target, h2_target, h3_target];
end