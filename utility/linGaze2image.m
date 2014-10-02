function gaze_img = linGaze2image(gaze_vec,gaze)
% Convert a gaze vector in a image for visualization purposes of the
% filters


imgH = gaze_vec(1:(gaze.highR.ptc_dim)^2);
imgM = gaze_vec((gaze.highR.ptc_dim)^2+1:(gaze.mediumR.ptc_dim^2-gaze.highR.ptc_dim^2)/(gaze.mediumR.px)^2+(gaze.highR.ptc_dim)^2);
imgL = gaze_vec((gaze.mediumR.ptc_dim^2-gaze.highR.ptc_dim^2)/(gaze.mediumR.px)^2+(gaze.highR.ptc_dim)^2+1:end);

mask = ones(gaze.mediumR.ptc_dim/gaze.mediumR.px);
mask((gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2/gaze.mediumR.px+1:((gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+gaze.highR.ptc_dim)/gaze.mediumR.px,...
    (gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2/gaze.mediumR.px+1:((gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+gaze.highR.ptc_dim)/gaze.mediumR.px) = Inf;
mask2 = ones(gaze.lowR.ptc_dim/gaze.lowR.px);
mask2((gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2/gaze.lowR.px+1:((gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2+gaze.mediumR.ptc_dim)/gaze.lowR.px,...
    (gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2/gaze.lowR.px+1:((gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2+gaze.mediumR.ptc_dim)/gaze.lowR.px) = Inf;

imgH = reshape(imgH,[gaze.highR.ptc_dim,gaze.highR.ptc_dim]);
mask(mask==1)   = imgM;
mask2(mask2==1) = imgL;

imgH = permute(imgH,[2,1]);
imgM = permute(mask,[2,1]);
imgL = permute(mask2,[2,1]);

imgM = imresize(imgM,gaze.mediumR.px,'nearest');
imgL = imresize(imgL,gaze.lowR.px,'nearest');


gaze_img = zeros(gaze.highR.ptc_dim);
gaze_img(imgH~=Inf) = imgH(imgH~=Inf); % there shouldn't be overlapping!
gaze_img = padarray(gaze_img,[(gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2 (gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2],0,'both');
gaze_img(imgM~=Inf) = imgM(imgM~=Inf); 
gaze_img = padarray(gaze_img,[(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2 (gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2],0,'both');
gaze_img(imgL~=Inf) = imgL(imgL~=Inf);

% % just plots from here
% subplot(141), imagesc(imgL),axis equal
% subplot(142), imagesc(imgM),axis equal
% subplot(143), imagesc(imgH),axis equal
% subplot(144), imagesc(gaze_img),axis equal, colormap gray
% x = 1; y = 1;
% boxL = [x,y,gaze.lowR.ptc_dim,gaze.lowR.ptc_dim];
% boxM = [x+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
%     y+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.mediumR.ptc_dim,gaze.mediumR.ptc_dim];
% boxH = [x+(gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
%     y+(gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.highR.ptc_dim,gaze.highR.ptc_dim];
% 
% rectangle('Position',boxL,'EdgeColor',[0 0 1]);
% rectangle('Position',boxM,'EdgeColor',[0 1 0]);
% rectangle('Position',boxH,'EdgeColor',[1 0 0]);