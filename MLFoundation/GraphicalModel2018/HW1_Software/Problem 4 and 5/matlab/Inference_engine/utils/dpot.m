function pot = dpot(domain, sizes, T)

pot.domain = domain(:)'; % so we can see it when we display
if nargin < 3
  pot.T = myones(sizes);
  %pot.T = ones(1,prod(sizes)); % 1D vector
else 
   if isempty(T)
      pot.T = [];
   else
      if issparse(T)
         pot.T = T;   
      else
         pot.T = myreshape(T, sizes);  
      end
   end
end
pot.sizes = sizes(:)';
% pot = class(pot, 'dpot');
