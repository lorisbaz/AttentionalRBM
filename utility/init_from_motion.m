function bbox_init = init_from_motion(imgt1,imgt2,dimT,plot_y)
% bbox_init = init_from_motion(imgt1,imgt2,dimT,,plot_y)
%
% This function initializes the bounding box of the target from the motion
% flow, computed by subtracting successive frames.
%
% - imgt1: first image
% - imgt2: next image
% - dimT: FIXED dimension of the template
% - plot_y: visualize init (default = 0)
%
% - bbox_init: bounding box containing the target
%
% Loris Bazzani
% loris.bazzani@univr.it


if nargin<4
   plot_y = 0; 
end

SG = imgt1-imgt2;
SG_th = SG>0 | SG<0;
SG_th = medfilt3(SG_th)>0;
se = strel('disk',2);
SG_th = imopen(SG_th,se);
SG_th = imdilate(SG_th,se);
Rprops = regionprops(bwlabel(SG_th),'BoundingBox');
bbox = cat(1, Rprops.BoundingBox);
[~,ind] = max(bbox(:,3).*bbox(:,4));
bbox_init = bbox(ind,:); % keep only one target
% "resize" the template
bbox_init(:,1:2) = round(bbox_init(:,1:2)-(dimT(1:2)-bbox_init(:,3:4))/2);
bbox_init(:,3:4) = dimT;
K = size(bbox_init,1); % # of targets (should be = 1)


%% visualize the motion-based initialization
if plot_y
    figure(321);
    subplot(111),imagesc(imgt1),colormap gray, axis equal, axis off;
    for k = 1:K
        rectangle('Position',bbox_init(k,:),'EdgeColor',[1 1 0],'LineWidth',3);
    end
    title('Phase 1: Detect the moving object (no identity)')
    hold off
    pause(0.01)
end