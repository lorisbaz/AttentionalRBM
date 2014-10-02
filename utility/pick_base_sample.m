function[result]=pick_base_sample(data_largest_cumulative_prob,global_nsamples,data_cumul_prob_array);

%   This is binary search using cumulative probabilities to pick a base
%   sample. The use of this routine makes Condensation O(NlogN) where N
%   is the number of samples. It is probably better to pick base
%   samples deterministically, since then the algorithm is O(N) and
%   probably marginally more efficient, but this routine is kept here
%   for conceptual simplicity and because it maps better to the
%   published literature. 

  choice = unifrnd(0,1) * data_largest_cumulative_prob;
  low = 0;
  high = global_nsamples;
%fprintf('choice = %i ; \n',choice);
 
%fprintf('high = %i ; low = %i  \n',high,low);

while high>(low+1)
   %  fprintf('high = %i ; low = %i  \n',high,low);
     middle = ceil((high+low)/2);
      %fprintf('middle= %i ;\n', middle);
    if choice > data_cumul_prob_array(middle)
      low = middle;
    else high = middle;
    end
end
  result = high;
