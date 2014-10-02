function display_GAZE_mostActFilters_binRBM(act_filt,Nvis,W,gaze)


[~,inds_ord] = sort(act_filt);
inds_ord = inds_ord(1:Nvis); % choose the most activated ones

W = W(:,inds_ord);

display_network_G(W,gaze);
title('Most Active RBM filters','fontsize',15)
