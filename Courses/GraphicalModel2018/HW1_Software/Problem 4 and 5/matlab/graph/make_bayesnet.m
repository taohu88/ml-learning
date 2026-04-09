function bnet = make_bayesnet(dag, node_sizes)

n = length(dag);

bnet.equiv_class = 1:n;
bnet.dnodes = 1:n; 
bnet.observed = [];
bnet.names = {};

bnet.dag = dag;
bnet.node_sizes = node_sizes(:)';

bnet.cnodes = mysetdiff(1:n, bnet.dnodes);

bnet.parents = cell(1,n);
for i=1:n
  bnet.parents{i} = find_parents(dag, i);
end

E = max(bnet.equiv_class);
mem = cell(1,E);
for i=1:n
  e = bnet.equiv_class(i);
  mem{e} = [mem{e} i];
end
bnet.members_of_equiv_class = mem;

bnet.CPD = cell(1, E);

bnet.rep_of_eclass = zeros(1,E);
for e=1:E
  mems = bnet.members_of_equiv_class{e};
  bnet.rep_of_eclass(e) = mems(1);
end

directed = 1;
if ~isacyclic(dag,directed)
  error('graph must be acyclic')
end

bnet.order = sort_topologically(bnet.dag);