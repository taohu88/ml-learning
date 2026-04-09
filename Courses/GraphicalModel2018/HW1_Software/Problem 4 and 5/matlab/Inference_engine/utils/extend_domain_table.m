function B = extend_domain_table(A, smalldom, smallsz, bigdom, bigsz)

if isequal(size(A), [1 1]) % a scalar
  B = A; % * myones(bigsz);
  return;
end

map = find_equiv_posns(smalldom, bigdom);
sz = ones(1, length(bigdom));
sz(map) = smallsz;
B = myreshape(A, sz); % add dimensions for the stuff not in A
sz = bigsz;
sz(map) = 1; % don't replicate along A's dimensions
B = myrepmat(B, sz(:)');
                           
