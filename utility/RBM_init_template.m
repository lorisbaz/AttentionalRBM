function [hsT,target_T,padW] = RBM_init_template(img,bbox,W,b,gazes_pos,gaze)
% [hsT,target_T,padW] = RBM_init_template(img,bbox,W,b,gazes_pos,gaze)
%
% The function extracts the gaze-based template from the first frame and 
% computes the hidden representation (i.e., hs of the bin. RBM) of each
% gaze. 
%
% - img: first frame
% - bbox: target bounding box
% - W,b: RBM learned parameters (weights and biases)
% - gazes_pos: relative position of the gazes
% - gaze: structure of the gaze
%
% - hsT: hidden representation of the gaze
% - target_T: image of the padded template (only for visualization)
% - padW: padding for the gazes
%
% Loris Bazzani
% loris.bazzani@univr.it


target_T = double(img(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1));
target_T = medfilt3(target_T); % remove noise
padW = (gaze.lowR.ptc_dim-gaze.highR.ptc_dim)/2;
target_T = padarray(target_T,[padW padW],0,'both');

hsT = zeros(size(gazes_pos,1),size(W,2));
for ns = 1:size(gazes_pos,1)  % create the gazes fot the template
    gaze_vec    = image2linGaze(target_T,gazes_pos(ns,1),gazes_pos(ns,2),gaze);
    hsT(ns,:)   = binRBM_vis2hid(double(gaze_vec),W,b)';
end