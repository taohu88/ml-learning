function smallpot = marginalize_pot(bigpot, onto, maximize)

if nargin < 3, maximize = 0; end

ns = zeros(1, max(bigpot.domain));
ns(bigpot.domain) = bigpot.sizes;

if issparse(bigpot.T)
   smallT = marg_sparse_table(bigpot.T, bigpot.domain, bigpot.sizes, onto, maximize);
else 
   smallT = marg_table(bigpot.T, bigpot.domain, bigpot.sizes, onto, maximize);
end

smallpot = dpot(onto, ns(onto), smallT);
