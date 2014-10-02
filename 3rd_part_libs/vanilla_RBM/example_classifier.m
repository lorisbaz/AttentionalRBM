%Example script for how to run the RBM code
load mnist_classification_small.mat
options.method = 'CD';
options.eta = 0.1;
options.momentum = 0.5;
options.maxepoch = 50;
options.avgstart = 45;
%penalty == weight decay
options.penalty = 2e-4;
options.numhid = 100;
options
[W,c,b] = fitclassifier(data,targets,testdata,testtargets,options);