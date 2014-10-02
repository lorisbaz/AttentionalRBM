%Trains an RBM using either CD or SML. Based on code by Ruslan Salakhutdinov.
function[W,c,b] = rbm(batchdata,init_params)
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

W = 0.1*randn(numdims,numhid);
c = zeros(1,numdims);
b = zeros(1,numhid);

ph = zeros(numcases,numhid);
nh = zeros(numcases,numhid);
phstates = zeros(numcases,numhid);
nhstates = zeros(numcases,numhid);
negdata = zeros(numcases,numdims);
negdatastates = zeros(numcases,numdims);

Winc  = zeros(numdims,numhid);
binc = zeros(1,numhid);
cinc = zeros(1,numdims);

Wavg = W;
bavg = b;
cavg = c;

t=1;
for epoch = 1:maxepoch,
% 	fprintf(1,'epoch %d\r',epoch); 
	errsum=0;
    for batch = 1:numbatches,
		[numcases numdims]=size(batchdata{batch});
		if (mod(batch,100) == 0)
			fprintf(1,'epoch %d batch %d\r',epoch,batch); 
		end
		data = batchdata{batch};

		ph = logistic(data*W + repmat(b,numcases,1));
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
		
		nh = logistic(negdatastates*W + repmat(b,numcases,1));
		nhstates = nh > rand(numcases,numhid);
		
		dW = (data'*ph - negdatastates'*nh);
		dc = sum(data) - sum(negdatastates);
		db = sum(ph) - sum(nh);
		
		err= sum(sum( (data-negdata).^2 ));
		errsum = err + errsum;
		
		decay = penalty*W; %% <= regularization term
		Winc = momentum*Winc + eta*(dW/numcases - decay); 
		binc = momentum*binc + eta*(db/numcases);
		cinc = momentum*cinc + eta*(dc/numcases);
		
		W = W + Winc;
		b = b + binc;
		c = c + cinc;
		
		%Trajectory averaging for faster convergence
		if (epoch > avgstart)
			Wavg = Wavg - (1/t)*(Wavg - W);
			cavg = cavg - (1/t)*(cavg - c);
			bavg = bavg - (1/t)*(bavg - b);
			t = t+1;
		else
			Wavg = W;
			cavg = c;
			bavg = b;
		end
    end
	fprintf(1, 'epoch %4i error %6.1f \n', epoch, errsum);
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