%From Benjamin Marlin (or possibly Mark Schmidt, I forget which is commented and which is not): returns the log of sum of logs, summing over dimension dim
function ls = logsum(xx,dim)
% ls = logsum(x,dim)
%
% returns the log of sum of logs, summing over dimension dim
% computes ls = log(sum(exp(x),dim))
% but in a way that tries to avoid underflow/overflow
%
% basic idea: shift before exp and reshift back
% log(sum(exp(x))) = alpha + log(sum(exp(x-alpha)));
%

% if(size(xx,dim)<=1) ls=xx; return; end
% 
% xdims=size(xx);
% if(nargin<2) 
%   nonsingletons=find(xdims>1);
%   dim=nonsingletons(1);
% end
% 
% alpha = max(xx,[],dim)-log(realmax)/2+2*log(xdims(dim));
% repdims=ones(size(xdims)); repdims(dim)=xdims(dim);
% ls = alpha+log(sum(exp(xx-repmat(alpha,repdims)),dim));
if nargin < 2
  dim = 1;
end

% subtract the largest in each column
[y, i] = max(xx,[],dim);
dims = ones(1,ndims(xx));
dims(dim) = size(xx,dim);
xx = xx - repmat(y, dims);
ls = y + log(sum(exp(xx),dim));
i = find(~isfinite(y));
if ~isempty(i)
  ls(i) = y(i);
end

