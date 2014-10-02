%% RBM tracking on gazes with control system: synthetic version with single target
%
% Loris Bazzani
% loris.bazzani@univr.it
% www.lorisbazzani.info

clear all; clc; close all; % vacum cleaner


%% USER SELECTIONS
control = 1; % active learning control = 1, circular control = 2, random control = 3
noise_on= 1; % load dataset with (=1) or without noise (=0)
NSEL    = 1; % select the experiment
plot_y  = 1; % plot stuff


%% LOAD STUFF
% load experiment dataset
load(['../datasets/synth_single/exp1_num' num2str(NSEL) '_Noise' num2str(noise_on) '_BG1.mat']); 

% Load utilities
addpath('3rd_part_libs/vanilla_RBM/')
addpath('utility/')

% load learned models
load('MAT/binRBM_GAZE_MNIST_all.mat') % BINARY RBM weights
load('MAT/logclass_GAZE_binRBM_MNIST_all.mat') % MULTI-LOGIT CLASS. weights


%% Parameters
% tracking
N       = 100;  % # particles
sigma   = 5;    % dynamical model position
sigma_v = 0.2;  % dynamical model velocity
sigmaobs= 0.05; % observation model

% active learning (hedging)
eta     = 0.1;

% motion detection
tstart_track= 2;


%% Targets INIT by Motion detection
bbox= init_from_motion(synth(1).img,synth(tstart_track).img,[HT,WT],plot_y);
K   = size(bbox,1); % # of targets (should be = 1)
cT  = NSEL; % class ground truth


%% INIT variables, template and gaze structure

% set the gazes in RELATIVE coordinates
shift1 = 4; % useful to center the gazes where the number is
shift2 = 2;
% gazes_pos = [shift1,shift1;...  % 4-gaze uniform template
%     gaze.highR.ptc_dim+1+shift1+shift2,shift1;...
%     gaze.highR.ptc_dim+1+shift1+shift2,gaze.highR.ptc_dim+1+shift1+shift2;...
%     shift1,gaze.highR.ptc_dim+1+shift1+shift2]; 
gazes_pos = [shift1+6,shift1+6;... % 4-gaze biased template
    gaze.highR.ptc_dim+1+shift1+shift2+6,shift1;...
    gaze.highR.ptc_dim+1+shift1+shift2+6,gaze.highR.ptc_dim+1+shift1+shift2+6;...
    shift1,gaze.highR.ptc_dim+1+shift1+shift2]; 


% INIT the gaze template
[hsT,target_T,padW] = RBM_init_template(synth(1).img,bbox,W,b,gazes_pos,gaze);

% init useful variables
x_prec      = double(repmat(bbox,N,1)); 
x_prec(:,1) = x_prec(:,1) + normrnd(0,sigma,N,1); % add noise to init
x_prec(:,2) = x_prec(:,2) + normrnd(0,sigma,N,1);
vel_prec    = normrnd(0,sigma_v,N,2);
pdf_prec    = ones(N,1)/N;
x_curr      = zeros(N,4);
vel_curr    = ones(N,2);
pdf_curr    = ones(N,1)/N;

% memorize things for visualization
Nvis        = 36; % # filters displayed
ACT_FLT     = zeros(Nframe,size(W,2));
estimate    = zeros(Nframe,4);
probH_time  = zeros(Nframe,10); 
CT_CLASS    = zeros(Nframe,10);
GAZE        = zeros(Nframe,1);
LABEL       = zeros(Nframe,1);
P_hedge     = zeros(size(gazes_pos,1),Nframe);
G           = zeros(size(gazes_pos,1),Nframe);
count_class = zeros(10,1);
pred_cH     = -1;
GAZE(1)     = 1; % starting gaze (for det. and rand.)
act_gaze    = 1; % ACTIVE GAZE DURING TRACKING

%% Start Tracking
labels_list = {'G1','G2','G3','G4'};
for t = tstart_track:Nframe
    
    %% Importance Sampling + Dynamics Model
    [x_curr,vel_curr] = IS_dynamics(x_prec,vel_prec,pdf_prec,sigma,sigma_v,N);
    
    
    %% Observation Model 
    if control==1 % ACTIVE LEARNING with Hedgeing (Max 1/ESS)
        
        pdf_curr_G  = zeros(size(gazes_pos,1),N);
        for g = 1:size(gazes_pos,1)
            pdf_curr_G(g,:) = RBM_obs_model(synth(t).img,x_curr,hsT,W,b,gazes_pos,gaze,g,N,sigmaobs,padW);
        end
        
        for g = 1:size(gazes_pos,1)
            % Reward = Effective Sample Size (ESS)
            xit = (sum(pdf_curr_G(g,:).^2)); % (sum(pdf_curr_G(g,:))^2
            G(g,t) = G(g,t-1) + xit;  % update total rewards
            % Rewarding probability
            P_hedge(g,t) = exp(eta*G(g,t-1))/sum(exp(eta*G(:,t-1)));
        end
        % Sample a action with probability P_hedge(:,t)
        cSum2    = cumsum(P_hedge(:,t));
        act_gaze = pick_base_sample(1,size(gazes_pos,1),cSum2);
        
        % select the current gaze
        pdf_curr = pdf_curr_G(act_gaze,:)';        
        GAZE(t)  = act_gaze; % for visualization
        
    else % CIRCULAR & RANDOM CONTROL
        
        pdf_curr = RBM_obs_model(synth(t).img,x_curr,hsT,W,b,gazes_pos,gaze,GAZE(t-1),N,sigmaobs,padW)'; % based on GAZE(t-1)
        
        % update gaze
        GAZE(t)  = act_gaze; % for visualization
        if control==2
            P_hedge(act_gaze,t) = 1; % det.
        elseif control ==3
            P_hedge(:,t) = 1/size(gazes_pos,1); % rand.
        end
    end

    
    %% SELECTION STEP: [N. de Freitas]
    outIndex = deterministicR(1:N,pdf_curr);   % Kitagawa resampling
    
    
    %% Update + Estimation
    x_prec      = x_curr(outIndex,:);
    vel_prec    = vel_curr(outIndex,:);
    pdf_prec    = ones(N,1)/N;   
    estimate(t,:) = round(pdf_prec'*x_prec); % Mean estimation
    
    
    %% Recognition   
    [CT_CLASS(t,:),probH_time(t,:),LABEL(t),ACT_FLT(t,:),IMGAZE{t}] = RBM_recogn(synth(t).img,estimate(t,:),CT_CLASS(t-1,:),weight,bias,W,b,gazes_pos,gaze,act_gaze,padW);
    % note: memorize stuff for visualization
    
    
    %% Display current frame results
    if plot_y
        display_final_res;
    end
    
    
    %% UPDATE CONTROL (only random and determinitic)
    if control==2 % update det control        
        act_gaze = act_gaze+1;
        if act_gaze/size(gazes_pos,1)>1 % circular gaze selection
            act_gaze = 1;
        end
    elseif control == 3 % update random control
        act_gaze = randi(size(gazes_pos,1));
    end

end


%% Save and Watch
save(['synth_RES/binRBM_ctr' num2str(control) '_GAZE_exp1' ...
     '_num' num2str(NSEL) '_Noise' num2str(noise_on) '_BG1.mat'],'estimate')
 
 
%% Let's watch the video
for t = tstart_track:Nframe
    display_final_res;
    pause(0.01)
end