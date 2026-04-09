function [pot, loglik] = normalize_pot(pot)

if isempty(pot.T)
   loglik = 0;
   return;
end
[pot.T, lik] = normalise(pot.T);
loglik = log(lik + (lik==0)*eps);

      
