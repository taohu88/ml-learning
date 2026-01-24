function pot = convert_to_pot(CPD, domain, evidence)
% CONVERT_TO_POT Convert a discrete CPD to a potential
% pot = convert_to_pot(CPD, pot_type, domain, evidence)
%
% pots = CPD evaluated using evidence(domain)

ncases = size(domain,2);
assert(ncases==1); % not yet vectorized

sz = CPD.sizes;
ns = zeros(1, max(domain));
ns(domain) = sz;

CPT1 = CPD.CPT;
spar = issparse(CPT1);
odom = domain(~isemptycell(evidence(domain)));
if spar
   T = convert_to_sparse_table(CPD, domain, evidence);
else 
   T = convert_to_table(CPD, domain, evidence);
end
ns(odom) = 1;
pot = dpot(domain, ns(domain), T); 
end
