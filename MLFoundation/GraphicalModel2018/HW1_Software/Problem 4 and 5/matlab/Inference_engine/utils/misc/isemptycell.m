function E = isemptycell(C)
E = zeros(size(C));
for i=1:prod(size(C))
    E(i) = isempty(C{i});
end
E = logical(E);
end