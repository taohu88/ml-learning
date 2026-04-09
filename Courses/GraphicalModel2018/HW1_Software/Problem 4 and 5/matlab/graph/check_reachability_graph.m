function C = check_reachability_graph(G)
M = expm(double(full(G))) - eye(length(G));
C = (M>0);
end