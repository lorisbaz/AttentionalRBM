%Trains an RBM using either CD or SML for classification. Based on code by Ruslan Salakhutdinov.
function[W,c,b,Wc,cc,testerr] = rbmclassifier(batchdata,batchclasses,testdata,testclasses,init_params)
eta = init_params.eta;
penalty = init_params.penalty;
numhid = init_params.numhid;
maxepoch = init_params.maxepoch;
avgstart = init_params.avgstart;
numbatches = length(batchdata);
momentum = init_params.momentum;
method = init_params.method;
numbatches = length(batchdata);
[numcases numdims]=size(batchdata{1});
numclasses = size(batchclasses{1},2);

W = 0.1*randn(numdims,numhid);
c = zeros(1,numdims);
b = zeros(1,numhid);

Wc = 0.1*randn(numclasses,numhid);
cc = zeros(1,numclasses);

ph = zeros(numcases,numhid);
nh = zeros(numcases,numhid);
phstates = zeros(numcases,numhid);
nhstates = zeros(numcases,numhid);
negdata = zeros(numcases,numdims);
negdatastates = zeros(numcases,numdims);

Winc  = zeros(numdims,numhid);
binc = zeros(1,numhid);
cinc = zeros(1,numdims);

Wcinc = zeros(numclasses,numhid);
ccinc = zeros(1,numclasses);

Wavg = W;
bavg = b;
cavg = c;

Wcavg = Wc;
ccavg = cc;
t = 1;
testerr = [];
for epoch = 1:maxepoch,
	fprintf(1,'epoch %d\r',epoch); 
	errsum=0;
	for batch = 1:numbatches,
		[numcases numdims]=size(batchdata{batch});
		numclasses = size(batchclasses{batch},2);
		if (mod(batch,100) == 0)
			fprintf(1,'epoch %d batch %d\r',epoch,batch); 
		end
		data = batchdata{batch};
		classes = batchclasses{batch};
		ph = logistic(data*W + classes*Wc + repmat(b,numcases,1));
		phstates = rand(numcases,numhid) < ph;
        if (isequal(method,'SML'))
            if (epoch == 1 && batch == 1)
                nhstates = phstates;
            end
        elseif (isequal(method,'CD'))
            nhstates = phstates;
        end
		
		negdata = logistic(nhstates*W' + repmat(c,numcases,1));
		negdatastates = negdata > rand(numcases,numdims);
		
		negclasses = softmax(nhstates*Wc' + repmat(cc,numcases,1));
		negclassesstates = softmax_sample(negclasses);
		
		nh = logistic(negdatastates*W + negclassesstates*Wc + repmat(b,numcases,1));
		nhstates = nh > rand(numcases,numhid);
		
        dW = (data'*ph - negdatastates'*nh);
        dc = sum(data) - sum(negdatastates);
        db = sum(ph) - sum(nh);

        dWc = (classes'*ph - negclassesstates'*nh);
        dcc = sum(classes) - sum(negclassesstates);
		
		err= sum(sum( (data-negdata).^2 ));
		errsum = err + errsum;
		
		decay = penalty*W;
		decayc = penalty*Wc;
        
		Winc = momentum*Winc + eta*(dW/numcases - decay);
		binc = momentum*binc + eta*(db/numcases);
		cinc = momentum*cinc + eta*(dc/numcases);
		
		Wcinc = momentum*Wcinc + eta*(dWc/numcases - decayc);
		ccinc = momentum*ccinc + eta*(dcc/numcases);
        
		
		W = W + Winc;
		b = b + binc;
		c = c + cinc;
		
		Wc = Wc + Wcinc;
		cc = cc + ccinc;
		
		%Trajectory averaging for faster convergence
		if (epoch > avgstart)
			Wavg = Wavg - (1/t)*(Wavg - W);
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
			
			Wcavg = Wcavg - (1/t)*(Wcavg - Wc);
			ccavg = ccavg - (1/t)*(ccavg - cc);
			
			t = t+1;
		else
			Wavg = W;
			bavg = b;
			cavg = c;
			
			Wcavg = Wc;
			ccavg = cc;
		end
    end
	testerr(end+1) = compute_test_error(testdata,testclasses,Wavg,bavg,cavg,Wcavg,ccavg);
	fprintf(1, 'epoch %4i\terror\t%6.1f\ttest error %6.4f\n', epoch, errsum, testerr(end));
	figure(10);
	subplot(2,2,1);
	display_network(Wavg(:,1:min(100,numhid)));
	subplot(2,2,2);
	imagesc(ph);
	colormap(gray);
	subplot(2,2,3);
	display_network(data');
	colormap(gray);
	subplot(2,2,4);
	display_network(negdata');
	colormap(gray);
	drawnow;
end;
W = Wavg;
b = bavg;
c = cavg;

Wc = Wcavg;
cc = ccavg;