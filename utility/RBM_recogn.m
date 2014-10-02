function [count_class,probH,pred_cH,act_filt,im_gaze] = RBM_recogn(img,estimate,count_class,weight,bias,W,b,gazes_pos,gaze,act_gaze,padW)
% [count_class,probH,pred_cH,act_filt,im_gaze] = RBM_recogn(img,estimate,count_class,weight,bias,W,b,gazes_pos,gaze,act_gaze,padW)
%
% Multi-logit classification using the hidden representation (hs of the RBM).
% This function classifies the estimation of the tracker using just the
% selected gaze.
% 
% - img: current image
% - estimate: current estimation of the target (bounding box)
% - count_class: cumulative classification result distribution
% - weight,bias: multi-logit class. learned parameters
% - W,b: RBM learned parameters
% - gazes_pos,gaze: gazes structure
% - act_gaze: current selected gaze
% - padW: paddin for the template
%
% - count_class: cumulative classification result distribution
% - probH,pred_cH,act_filt,im_gaze are stuff for visualization
%
% Loris Bazzani
% loris.bazzani@univr.it

obs = double(img(estimate(2)-padW:estimate(2)+estimate(4)-1+padW,estimate(1)-padW:estimate(1)+estimate(3)-1+padW));
obs = image2linGaze(obs,gazes_pos(act_gaze,1),gazes_pos(act_gaze,2),gaze); % only the active gaze
[hsH,act_filt] = binRBM_vis2hid(obs,W,b); % hypothesis
clVec = hsH;
probH = softmax(weight*clVec' + bias); % Recognition
[~,pred_cH] = max(probH);
count_class(pred_cH) = count_class(pred_cH)+1;
pred_cH = pred_cH-1; % because it start from 0
im_gaze =  linGaze2image(obs,gaze);
