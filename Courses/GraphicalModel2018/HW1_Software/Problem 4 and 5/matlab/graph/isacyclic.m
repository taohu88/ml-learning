function b = isacyclic(adj_mat, directed)
% retuns true if the graph has no (directed) cycles.


adj_mat = double(adj_mat);
if nargin < 2, directed = 1; end

if directed
  R = check_reachability_graph(adj_mat);
  b = ~any(diag(R)==1);
else
  [d, pre, post, cycle] = dfs(adj_mat,[],directed);
  b = ~cycle;    
end
