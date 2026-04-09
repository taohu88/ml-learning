function p = find_equiv_posns(vsmall, vlarge)


if isempty(vsmall) | isempty(vlarge)
  p = [];
  return;
end
  
bitvec = sparse(1, max(vlarge)); 
bitvec(vsmall) = 1;
p = find(bitvec(vlarge));
