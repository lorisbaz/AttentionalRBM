function gaze_vec = image2linGaze(img,x,y,gaze)
% Convert a image in a gaze vector as in the training protocol, where the 
% left-upper corner of the gaze is given by (x,y)


boxL = [x,y,gaze.lowR.ptc_dim,gaze.lowR.ptc_dim];
boxM = [x+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
    y+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.mediumR.ptc_dim,gaze.mediumR.ptc_dim];
boxH = [x+(gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
    y+(gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.highR.ptc_dim,gaze.highR.ptc_dim];
 
% subplot(141),imagesc(img),axis equal
% rectangle('Position',boxL,'EdgeColor',[0 0 1]);
% rectangle('Position',boxM,'EdgeColor',[0 1 0]);
% rectangle('Position',boxH,'EdgeColor',[1 0 0]);

imgL = imresize(img(boxL(2):boxL(2)+boxL(4)-1,boxL(1):boxL(1)+boxL(3)-1),1/gaze.lowR.px,'bilinear');
imgM = imresize(img(boxM(2):boxM(2)+boxM(4)-1,boxM(1):boxM(1)+boxM(3)-1),1/gaze.mediumR.px,'bilinear');
imgH = img(boxH(2):boxH(2)+boxH(4)-1,boxH(1):boxH(1)+boxH(3)-1);
imgL = permute(imgL,[2,1]);
imgM = permute(imgM,[2,1]);
imgH = permute(imgH,[2,1]);
mask = ones(size(imgM));
mask((gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2/gaze.mediumR.px+1:((gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+gaze.highR.ptc_dim)/gaze.mediumR.px,...
    (gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2/gaze.mediumR.px+1:((gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+gaze.highR.ptc_dim)/gaze.mediumR.px) = 0;
mask2 = ones(size(imgL));
mask2((gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2/gaze.lowR.px+1:((gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2+gaze.mediumR.ptc_dim)/gaze.lowR.px,...
    (gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2/gaze.lowR.px+1:((gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2+gaze.mediumR.ptc_dim)/gaze.lowR.px) = 0;

gaze_vec = [imgH(:)',imgM(logical(mask))',imgL(logical(mask2))'];


% subplot(142), imagesc(imgL),axis equal
% subplot(143), imagesc(imgM),axis equal
% subplot(144), imagesc(imgH),axis equal
% pause