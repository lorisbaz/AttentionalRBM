digitdata=[]; 
targets=[]; 
load('../data/mnist/digit0'); digitdata = [digitdata; D]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];  
load('../data/mnist/digit1'); digitdata = [digitdata; D]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
load('../data/mnist/digit2'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)]; 
load('../data/mnist/digit3'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
load('../data/mnist/digit4'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)]; 
load('../data/mnist/digit5'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
load('../data/mnist/digit6'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
load('../data/mnist/digit7'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
load('../data/mnist/digit8'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
load('../data/mnist/digit9'); digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digitdata = digitdata/255;
data = digitdata;

digitdata=[];
testtargets=[];
load('../data/mnist/test0'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
load('../data/mnist/test1'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
load('../data/mnist/test2'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
load('../data/mnist/test3');digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
load('../data/mnist/test4'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
load('../data/mnist/test5'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
load('../data/mnist/test6'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
load('../data/mnist/test7'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
load('../data/mnist/test8'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
load('../data/mnist/test9'); digitdata = [digitdata; D]; testtargets = [testtargets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digitdata = digitdata/255;
testdata = digitdata;