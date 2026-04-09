function T = myreshape(T, sizes)

if length(sizes)==0
  return;
elseif length(sizes)==1
  T = reshape(T, [sizes 1]);
else
  T = reshape(T, sizes(:)');
end