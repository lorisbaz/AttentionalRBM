function[y] = softmax2(x)
y = exp(x - repmat(logsum(x,2),1,size(x,2)));
y = y./repmat(sum(y,2),1,size(x,2));