function T = myones(sizes)

if length(sizes)==0
  T = 1;
elseif length(sizes)==1
  T = ones(sizes, 1);
else
  T = ones(sizes(:)');
end
