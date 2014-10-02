function[err] = compute_test_error(testdata,testclasses,W,b,c,Wc,cc)
[numdims,numhid] = size(W);
[numcases,numclasses] = size(testclasses);
F = zeros(size(testclasses));
for i=1:numclasses
	X = zeros(numcases,numclasses);
	X(:,i) = 1;
	%Is there a safer way to do this?
	F(:,i) = repmat(cc(i),numcases,1).*X(:,i)+sum(log(exp(testdata*W+X*Wc+repmat(b,numcases,1))+1),2);
end

[junk,prediction] = max(F,[],2);
[junk,classes] = max(testclasses,[],2);
err = sum(prediction~=classes)/numcases;