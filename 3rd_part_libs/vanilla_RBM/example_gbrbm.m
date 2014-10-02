%Example script for how to run the RBM code
load cifar10bw_small.mat
data = data - repmat(mean(data),size(data,1),1);
options.method = 'CD';
options.eta = 0.01;
options.sigma = 10;
options.momentum = 0.5;
options.maxepoch = 500;
options.avgstart = 450;
%penalty == weight decay
options.penalty = 2e-4;
options.numhid = 100;
options
[W,c,b] = fitgbrbm(data,options);