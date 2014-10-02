function pdf_curr = RBM_obs_model(img,x_curr,hsT,W,b,gazes_pos,gaze,sel_gaze,N,sigmaobs,padW)
% pdf_curr = RBM_obs_model(img,x_curr,hsT,W,b,gazes_pos,gaze,sel_gaze,N,sigmaobs,padW)
%
% Observation model of tracking algorithm computes the importance weights
% comparing the template against each hypothesis (particle) using the hidden
% representation (hs of the RBM).
%
% - img: current frame
% - x_curr: particles being tested
% - hsT: hidden representation of the template
% - W,b: RBM learned params
% - gazes_pos,gaze: gazes' structure
% - sel_gaze: current selected gaze
% - N: number of particles
% - sigmaobs: variance of the obs. model
% - padW: padding of the template
%
% - pdf_curr: current importance weights
%
% Loris Bazzani
% loris.bazzani@univr.it

[HIMG,WIMG,~] = size(img);

pdf_curr = zeros(1,N);
for n = 1:N
    bbox= round(x_curr(n,:));
    if bbox(2)+bbox(4)-1+padW<=HIMG && bbox(1)+bbox(3)-1+padW<=WIMG && ...
            bbox(2)-padW>=1 && bbox(1)-padW>=1
        
        obs = double(img(bbox(2)-padW:bbox(2)+bbox(4)-1+padW,bbox(1)-padW:bbox(1)+bbox(3)-1+padW));
        obs = image2linGaze(obs,gazes_pos(sel_gaze,1),gazes_pos(sel_gaze,2),gaze); % only the active gaze
        hsH = binRBM_vis2hid(obs,W,b); % hypothesis
        
        pdf_curr(n) =  exp(-1/(2*sigmaobs^2)*(bhattacharyya(hsT(sel_gaze,:),hsH))^2);
        
    else
        pdf_curr(n) = 0;
    end
end

pdf_curr = pdf_curr/sum(pdf_curr);
