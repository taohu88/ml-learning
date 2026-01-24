function cs = find_children(adj_mat, i, t)
% Return the indices of a node's children in sorted order

if nargin < 3 
  cs = find(adj_mat(i,:));
else
  if t==1
    cs = find(adj_mat(i,:));
  else
    ss = length(adj_mat)/2;
    j = i+ss;
    cs = find(adj_mat(j,:)) + (t-2)*ss;
  end
end
