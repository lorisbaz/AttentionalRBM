clear,clc,close all;
addpath('utility')

%% Synth-video creation
Nexp    = 2;
NSEL    = 0; % selected number
saveon  = 1;
numbersBG   = 1;
Nnum        = 20;
Nframe  = 300;
HIMG    = 200;
WIMG    = 250;
HT      = 28;
WT      = 28;

load MAT/mnist_all.mat % select a target

target_or = eval(['test' num2str(NSEL)] );
ind = randi(size(target_or,1),1);
target_or = target_or(ind,:);

target  = permute(reshape(target_or,[28,28]),[2,1]);
alpha   = target>0; % fg/bg


img_BG = zeros(HIMG,WIMG);
if numbersBG == 1
    numbers = setdiff(0:9,NSEL);     
    for n = numbers
        numset = eval(['test' num2str(n)]);
        inds = randi([1,size(numset,1)],round(Nnum/length(numbers)),1);
        for i = inds'
            number = permute(reshape(numset(i,:),[28,28]),[2,1]);
            bbox = [randi(WIMG-WT),randi(HIMG-HT),WT,HT];
            tmp = img_BG(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1); % overlap the already bg
            alpha2 = number>0;
            tmp(alpha2) = number(alpha2);
            img_BG(bbox(2):bbox(2)+bbox(4)-1,bbox(1):bbox(1)+bbox(3)-1) = tmp;
        end
    end
    img_BG = single(img_BG); % binarize   
end

% Select the path
imagesc(img_BG); colormap gray,axis image
x = []; y = [];
but = 1;
while but == 1
    [xi, yi, but] = ginput(1);
    x = [x xi];
    y = [y yi];
end;

% spline building
n = length(x);
t = 1:n;
ts = linspace(1,n,Nframe);
xs = spline(t,x,ts);
ys = spline(t,y,ts);
pos= uint16([xs',ys']);

% Video
for i=1:2
    if i == 1
        randomNoise = 0;
        NoiseLev = 0.0;
    else
        randomNoise = 1;
        NoiseLev = 0.3;
    end
    synth = struct([]);
    RBMdata = [];
    for t = 1:Nframe
        if randomNoise
            im_noisy 	= imnoise(zeros(HIMG,WIMG),'salt & pepper',NoiseLev); % with noise here
            im_noisy    = im_noisy.*randi([1,255],size(im_noisy));
        else
            im_noisy 	= zeros(HIMG,WIMG);
        end
        im_noisy(img_BG>0) = img_BG(img_BG>0);
        synth(t).img = im_noisy;
        
        synth(t).gt     = uint16([pos(t,1)-WT/2 pos(t,2)-HT/2 WT HT]);
        
        tmp     = synth(t).img(synth(t).gt(2):synth(t).gt(2)+synth(t).gt(4)-1,synth(t).gt(1):synth(t).gt(1)+synth(t).gt(3)-1);
        tmpf1   = tmp;
        tmpf1(alpha) = target(alpha);
        synth(t).img(synth(t).gt(2):synth(t).gt(2)+synth(t).gt(4)-1,synth(t).gt(1):synth(t).gt(1)+synth(t).gt(3)-1) = tmpf1;
        
        imagesc(synth(t).img), colormap gray,axis image;
        hold on; rectangle('Position',synth(t).gt);
        scatter(pos(:,1),pos(:,2)), hold off;
        drawnow;
    end
    
    if saveon
        save(['synth_dataset/exp' num2str(Nexp) '_num' num2str(NSEL) '_Noise' num2str(randomNoise) '_BG' num2str(numbersBG)])
    end
end