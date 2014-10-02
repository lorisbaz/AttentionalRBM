function [rate tspred] = test_logistic_regression(tsdata,tslabel,weight,bias)
% compute accuracy of logistic regression classifier
%
% tsdata : DxP
% tslabel : Px1
% weight : CxD
% bias : Cx1
% tspred ; Px1
%
% P nr samples
% D dimensionality of data
% C nr of classes
%
% Marc'Aurelio Ranzato
% 11 Oct. 2009

out = softmax(weight*tsdata + repmat(bias,1,size(tsdata,2)));
% pick the max
[prob tspred] = max(out);
tspred = tspred';
rate = sum(tspred == tslabel)/length(tslabel);
