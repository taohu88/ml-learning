function T = myrepmat(T, sizes)

if length(sizes)==1
  T = repmat(T, [sizes 1]);
else
  T = repmat(T, sizes(:)');
end
