function index = mk_multi_index(n, dims, vals)
if n==0
  index = { 1 };
  return;
end

index = cell(1,n);
for i=1:n
  index{i} = ':';
end
for i=1:length(dims)
  index{dims(i)} = vals(i);
end

