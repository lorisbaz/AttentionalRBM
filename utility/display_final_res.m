

% Plot
figure(321);
x = estimate(t,1)-padW+gazes_pos(GAZE(t),1); y = estimate(t,2)-padW+gazes_pos(GAZE(t),2);
boxL = [x,y,gaze.lowR.ptc_dim,gaze.lowR.ptc_dim];
subplot(2,3,1),imagesc(synth(t).img), colormap gray,axis image; axis off;
hold on
h1 = rectangle('Position',synth(t).gt,'EdgeColor',[0 1 0],'LineWidth',3);
p1 = plot(nan,nan,'s','markeredgecolor',get(h1,'edgecolor'),...
    'markerfacecolor',get(h1,'facecolor'));
h2 = rectangle('Position',estimate(t,:),'EdgeColor',[1 0 0],'LineWidth',3);
p2 = plot(nan,nan,'s','markeredgecolor',get(h2,'edgecolor'),...
    'markerfacecolor',get(h2,'facecolor'));
scatter(synth(t).gt(1)+synth(t).gt(3)/2,synth(t).gt(2)+synth(t).gt(4)/2,'og','LineWidth',3)
scatter(estimate(t,1)+estimate(t,3)/2,estimate(t,2)+estimate(t,4)/2,'+r','LineWidth',3)
title('Tracking (green = ground truth, red = estimate)','fontsize',15)
hold off;

subplot(232),
imagesc(target_T), axis off, axis equal, hold on
for ns = 1:size(gazes_pos,1)  % create the gazes fot the template
    boxM = [gazes_pos(ns,1)+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
        gazes_pos(ns,2)+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.mediumR.ptc_dim,gaze.mediumR.ptc_dim];
    if ns==GAZE(t)
        rectangle('Position',boxM,'EdgeColor',[0 0 1],'LineWidth',2);
    else
        rectangle('Position',boxM,'EdgeColor',[1 0 0],'LineWidth',2);
    end
    text(boxM(1)+1,boxM(2)+2,labels_list{ns},'BackgroundColor',[1,1,1])
    
end
title('4-gaze template (blue = selected)','fontsize',15)
hold off;


subplot(2,3,3),imagesc(IMGAZE{t}),colormap gray,axis image; axis off;
boxM = [(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
    (gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.mediumR.ptc_dim,gaze.mediumR.ptc_dim];
boxH = [(gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,...
    (gaze.mediumR.ptc_dim-gaze.highR.ptc_dim)/2+(gaze.lowR.ptc_dim-gaze.mediumR.ptc_dim)/2,gaze.highR.ptc_dim,gaze.highR.ptc_dim];
rectangle('Position',[0,0,gaze.lowR.ptc_dim+1,gaze.lowR.ptc_dim+1],'EdgeColor',[0 0 1],'LineWidth',24);
rectangle('Position',boxM,'EdgeColor',[0 0 0.7],'LineWidth',2);
rectangle('Position',boxH,'EdgeColor',[0 0 0.5],'LineWidth',2);
title('Actual observation','fontsize',15)
    
subplot(234),
display_GAZE_mostActFilters_binRBM(ACT_FLT(t,:),Nvis,W,gaze);

subplot(236)
bar(0:9,CT_CLASS(t,:)/(t-tstart_track+1),'c'), hold on,bar(NSEL,CT_CLASS(t,NSEL+1)/(t-tstart_track+1),'m');
title('Cumulative class probability','fontsize',15)
xlabel('Digits','fontsize',14)
set(gca,'ylim',[0 1]), grid on, hold off;

subplot(235), bar(P_hedge(:,t),'r'); hold on,
bar(GAZE(t),P_hedge(GAZE(t),t),'b'); grid on;
set(gca,'ylim',[0 1]),
set(gca,'XTickLabel',labels_list),
xlabel('Gazes','fontsize',14),hold off;
switch control
    case 1
        title('Learned gaze policy','fontsize',15)
    case 2
        title('Deterministic gaze policy','fontsize',15)
    case 3
        title('Random gaze policy','fontsize',15)
end
drawnow;
