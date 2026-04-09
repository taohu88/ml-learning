function m = marginal_family(engine, i, t)

if nargin < 3, t = 1; end

bnet = bnet_from_engine(engine);
if t==1
  m = marginal_nodes(engine, family(bnet.dag, i));
else
  ss = length(bnet.intra);
  fam = family(bnet.dag, i+ss);
  if any(fam<=ss)
    m = marginal_nodes(engine, fam, t-1);
  else
    m = marginal_nodes(engine, fam-ss, t);
  end
end     
