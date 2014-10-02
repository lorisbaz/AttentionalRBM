function display_mostActFilters_binRBM(act_filt,Nvis,W)


[~,inds_ord] = sort(act_filt);
inds_ord = inds_ord(1:Nvis); % choose the most activated ones

W = W(:,inds_ord);

display_network(W);