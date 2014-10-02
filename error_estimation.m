clc,clear;
addpath('synth_RES/')

% Tracking Error computation

for num = 0:9
    namedataset = ['exp1_num' num2str(num) '_Noise1_BG1'];
    load(['../datasets/synth_single/' namedataset '.mat']); % load experiment
    
    fprintf('\n ====== NUMBER %d ======\n',num);
    for c = 1:3 % control policy
        load(['synth_RES/binRBM_ctr' num2str(c) '_GAZE_' namedataset]);
        error = zeros(Nframe,1);
        for t = 2:Nframe
            error(t) = pdist([estimate(t,1:2);double(synth(t).gt(1:2))]);
        end
        error = error(2:Nframe);
        fprintf('Tracking error binRBM policy %d: %4.2f %4.2f \n',c,mean(error),var(error));
        strored_val_mean(c,num+1) = mean(error);
        strored_val_var(c,num+1) = var(error);
    end    
end
for c = 1:3 % control policy
    fprintf('\nMean Tracking error binRBM policy %d: %4.2f %4.2f \n',c,mean(strored_val_mean(c,:)),mean(strored_val_var(c,:)));
end
