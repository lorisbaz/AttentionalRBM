%Loads data, splits data into minibatches, initializes parameters, and calls code to train an RBM
function[W,c,b,Wc,cc,testerr] = fitclassifier(data,targets,testdata,testtargets,options,seed)
	%Set up hyperparameter values
	[init_params] = process_options(options);
    if (~exist('seed'))
        rand('state',1);
        randn('state',1);
    else
        rand('state',seed);
        randn('state',seed);
    end
    batchdata = {};
    %Split the data into mini batches
    nTrain = size(data, 1);
    num_batches = ceil(nTrain/init_params.batchsize);
    groups = repmat(1:num_batches,1,init_params.batchsize);
    groups = groups(1:nTrain);
    groups = groups(randperm(nTrain));
    for i=1:num_batches
        batchdata{i} = data(groups == i,:);
		batchtargets{i} = targets(groups == i,:);
    end
  	[W,c,b,Wc,cc,testerr] = rbmclassifier(batchdata,batchtargets,testdata,testtargets,init_params);
end

%From Mark Schmidt's minFunc package
function[init_params] = process_options(o)
o = toUpper(o);
init_params.batchsize = getOpt(o,'BATCHSIZE',100);
init_params.eta = getOpt(o,'ETA',0.1);
init_params.numhid = getOpt(o,'NUMHID',100);
init_params.maxepoch = getOpt(o,'MAXEPOCH',50);
init_params.avgstart = getOpt(o,'AVGSTART',Inf);
init_params.penalty = getOpt(o,'PENALTY',2e-4);
init_params.numon = getOpt(o,'NUMON',10);
init_params.momentum = getOpt(o,'MOMENTUM',0);
init_params.method = getOpt(o,'METHOD','CD');
end

function [v] = getOpt(options,opt,default)
if isfield(options,opt)
    if ~isempty(getfield(options,opt))
        v = getfield(options,opt);
    else
        v = default;
    end
else
    v = default;
end
end

function [o] = toUpper(o)
if ~isempty(o)
    fn = fieldnames(o);
    for i = 1:length(fn)
        o = setfield(o,upper(fn{i}),getfield(o,fn{i}));
    end
end
end