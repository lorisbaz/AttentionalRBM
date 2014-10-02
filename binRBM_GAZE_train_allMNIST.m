% Train RBM on all mnist dataset using gaze data
clear,clc,close all;
addpath('utility/')
addpath('3rd_part_libs/vanilla_RBM/')

% load dataset
load('MAT/mnist_all.mat')
HT      = 28; WT      = 28;

%% Parameters
% gaze setup
gaze.highR.ptc_dim  = 8;
gaze.highR.px       = 1; % pixel for each vis. unit
gaze.mediumR.ptc_dim= 16;
gaze.mediumR.px     = 2; % pixel for each vis. unit
gaze.lowR.ptc_dim   = 24;
gaze.lowR.px        = 4; % pixel for each vis. unit
H = 28+gaze.lowR.px*2; W = 28+gaze.lowR.px*2; % padding
NS = 2; % num. gazes for each instance
padW = (gaze.lowR.ptc_dim-gaze.highR.ptc_dim)/2; % padding

% RBM params
options.method = 'SML';
options.eta = 0.1;
options.momentum = 0.3;
options.maxepoch = 100;
options.avgstart = 50;
options.penalty = 1e-3;
options.numhid = 500;

%% Build gazes for each data
fprintf('Loading data... ');
datanow_gaze = []; labels = [];
HTMP = HT+padW*2-gaze.lowR.ptc_dim;
WTMP = WT+padW*2-gaze.lowR.ptc_dim;
for n = 1:10 % for each digit    
    NSEL = n-1; % select the digit you want to train
    datanow = double(eval(['train' num2str(NSEL)]));
    
    for i = 1:size(datanow,1)
        datanow_img = permute(reshape(datanow(i,:),[28,28]),[2,1]);
        datanow_img = padarray(datanow_img,[padW padW],'replicate','both');

        for ns = 1:NS
            x = randi(WTMP); y = randi(HTMP);
            
            gaze_vec = image2linGaze(datanow_img,x,y,gaze);
            
            datanow_gaze = [datanow_gaze;gaze_vec];
            labels = [labels;NSEL];

        end
    end
    fprintf('%d ',NSEL);
end
fprintf('Randomize data... ');
inds_rand   = randperm(size(datanow_gaze,1))';
datanow_gaze= datanow_gaze(inds_rand,:)/255;
labels  = labels(inds_rand);

%% bin RBM on soft data
fprintf('\nTraining... ');
options
[W,c,b] = fit_G(datanow_gaze,options,gaze); % fit binary RBM

fprintf('\nSaving... ');
save('MAT/binRBM_GAZE_MNIST_all.mat','W','c','b','gaze');
