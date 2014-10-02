clear,clc,close all;
addpath('utility/')

load('MAT/mnist_all.mat')
load('MAT/binRBM_GAZE_MNIST_all.mat') % load binRBM

% User setup
NOISE   = 0; % 1=on, 2=off
nrcl    = 10;
nr      = 1;
Nch     = 1;
HT      = 28;
WT      = 28;
padW = (gaze.lowR.ptc_dim-gaze.highR.ptc_dim)/2; % padding

%% Parameters (for logistic classifier)
eps = 0.001;
wd = 0.04;
nr_epochs = 10;
batch_size = 200;
NS = 1; % # gazes for each data

%% Load gaze data and compute RBM responses
HTMP = HT+padW*2-gaze.lowR.ptc_dim;
WTMP = WT+padW*2-gaze.lowR.ptc_dim;

% load training data
fprintf('Loading training data... ');
datanow_gaze = []; tr_labels = [];
for n = 1:nrcl % for each digit
    NSEL = n-1; % select the digit you want to train
    datanow = double(eval(['train' num2str(NSEL)]));
    
    for i = 1:size(datanow,1)
        datanow_img = permute(reshape(datanow(i,:),[HT,WT]),[2,1]);
        datanow_img = padarray(datanow_img,[padW padW],'replicate','both');
        
        for ns = 1:NS
            x = randi(WTMP); y = randi(HTMP);
            
            gaze_vec = image2linGaze(datanow_img,x,y,gaze);
            
            datanow_gaze = [datanow_gaze;gaze_vec];
            tr_labels = [tr_labels;NSEL];
        end
    end
    
    fprintf('%d ',NSEL);
end
tr_feature  = binRBM_vis2hid(double(datanow_gaze),W,b);


if NOISE
    % Add noisy training data
    fprintf('\nLoading noisy training data... ');
    datanow_gaze = []; tr_labels2 = [];
    for n = 1:nrcl % for each digit
        NSEL = n-1; % select the digit you want to train
        datanow = double(eval(['train' num2str(NSEL)]));
        for i = 1:size(datanow,1)
            datanow_img = permute(reshape(datanow(i,:),[HT,WT]),[2,1]);
            datanow_img = padarray(datanow_img,[padW padW],0,'both');
            im_noisy 	= imnoise(zeros(size(datanow_img)),'salt & pepper',NoiseLev);
            im_noisy    = im_noisy.*randi([1,255],size(im_noisy));
            datanow_img(datanow_img==0) = im_noisy(datanow_img==0);
            
            for ns = 1:NS
                x = randi(WTMP); y = randi(HTMP);
            
                gaze_vec = image2linGaze(datanow_img,x,y,gaze);
                
                datanow_gaze = [datanow_gaze;gaze_vec];
                tr_labels2 = [tr_labels2;NSEL];
            end
        end
        fprintf('%d ',NSEL);
    end
    % White noisy training set + feature compute
    tr_feature2 = binRBM_vis2hid(double(datanow_gaze),W,b);
    tr_feature = [tr_feature;tr_feature2]; % merge noisy data
    tr_labels  = [tr_labels;tr_labels2];
end
fprintf('\nRandomize data... ');
inds_rand   = randperm(size(tr_feature,1))';
tr_feature= tr_feature(inds_rand,:); % soft data
tr_labels = tr_labels(inds_rand);

% Load testing set
fprintf('\nLoading testing data... ');
datanow_gaze = []; ts_labels = [];
for n = 1:nrcl % for each digit
    NSEL = n-1; % select the digit you want to train
    datanow = double(eval(['test' num2str(NSEL)]));
    for i = 1:size(datanow,1)
        datanow_img = permute(reshape(datanow(i,:),[HT,WT]),[2,1]);
        datanow_img = padarray(datanow_img,[padW padW],'replicate','both');
        
        for ns = 1:NS
            x = randi(WTMP); y = randi(HTMP);
            
            gaze_vec = image2linGaze(datanow_img,x,y,gaze);
            
            datanow_gaze = [datanow_gaze;gaze_vec];
            ts_labels = [ts_labels;NSEL];
        end
    end
    fprintf('%d ',NSEL);
end
fprintf('\n');
% White testing set + feature compute
ts_feature = binRBM_vis2hid(double(datanow_gaze),W,b);

if NOISE
    % Add noisy testing data
    fprintf('\nLoading noisy testing data... ');
    datanow_gaze = []; ts_labels2 = [];
    for n = 1:nrcl % for each digit
        NSEL = n-1; % select the digit you want to train
        datanow = double(eval(['test' num2str(NSEL)]));
        for i = 1:size(datanow,1)
            datanow_img = permute(reshape(datanow(i,:),[HT,WT]),[2,1]);
            datanow_img = padarray(datanow_img,[padW padW],0,'both');
            im_noisy 	= imnoise(zeros(size(datanow_img)),'salt & pepper',NoiseLev);
            im_noisy    = im_noisy.*randi([1,255],size(im_noisy));
            datanow_img(datanow_img==0) = im_noisy(datanow_img==0);
            
            for ns = 1:NS
                x = randi(WTMP); y = randi(HTMP);
                
                gaze_vec = image2linGaze(datanow_img,x,y,gaze);
                
                datanow_gaze = [datanow_gaze;gaze_vec];
                ts_labels2 = [ts_labels2;NSEL];
            end
        end
        fprintf('%d ',NSEL);
    end
    % White noisy training set + feature compute
    ts_feature2 = binRBM_vis2hid(double(datanow_gaze),W,b); % template
    ts_feature = [ts_feature;ts_feature2]; % merge noisy data
    ts_labels  = [ts_labels;ts_labels2];
end

ts_feature = ts_feature';
tr_feature = tr_feature';


%% Logistic Regression [Ranzato's code]
targetmat = eye(nrcl);
dim = size(tr_feature,1);
weight = 0.02*randn(nrcl,dim);
bias = zeros(nrcl,1);
epsw = eps;
epsb = eps/10;
nr_batches = floor(size(tr_feature,2)/batch_size);
for ee = 1 : nr_epochs
    fprintf('Epoch %d: ' , ee)
    epswc = epsw./max(1,floor(max(1,ee/20))-2);
    epsbc = epsb./max(1,floor(max(1,ee/20))-2);
    for bb = 1 : nr_batches
        in = tr_feature(:,1+batch_size*(bb-1) : batch_size*bb);
        target = targetmat(tr_labels(1+batch_size*(bb-1) : batch_size*bb)+1,:);
        out = softmax(weight*in + repmat(bias,1,batch_size));
        temp = (out' - target);
        dEdBias = sum(temp)';
        dEdWeight = (in*temp)';
        % update
        weight = weight - (epswc/batch_size) * (dEdWeight + wd*sign(weight));
        bias = bias - (epsbc/batch_size) * dEdBias;
    end
    wtrrate(ee) = test_logistic_regression(tr_feature,tr_labels+1,weight,bias);
    fprintf(' - Training Accuracy rate %f',wtrrate(ee))
    tsrate(ee) = test_logistic_regression(ts_feature,ts_labels+1,weight,bias);
    fprintf(' - Test Accuracy rate %f\n',tsrate(ee))
end

% Compute the testing error
[rate tsprediction] = test_logistic_regression(ts_feature,ts_labels+1,weight,bias);
cm = zeros(nrcl);
for tt = 1 : length(ts_labels)
    cm(ts_labels(tt)+1,tsprediction(tt)) = cm(ts_labels(tt)+1,tsprediction(tt)) + 1;
end

% Save classifier
if ~NOISE
    save('MAT/logclass_GAZE_binRBM_MNIST_all.mat','weight','bias','rate','tsprediction','wd','nr_epochs','eps','wtrrate','tsrate','cm')
else
    save('MAT/logclass_GAZE_binRBM_noisyMNIST_all.mat','weight','bias','rate','tsprediction','wd','nr_epochs','eps','wtrrate','tsrate','cm')
end