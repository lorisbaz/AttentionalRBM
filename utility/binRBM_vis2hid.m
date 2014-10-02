function [feature,act_filt] = binRBM_vis2hid(data,W,b)

numcases = size(data,1);

data = data/255;
act_filt = data*W + repmat(b,numcases,1);
feature = logistic(act_filt);
