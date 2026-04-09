function bigT = mult_by_table(bigT, bigdom, bigsz, smallT, smalldom, smallsz)

Ts = extend_domain_table(smallT, smalldom, smallsz, bigdom, bigsz);
bigT(:) = bigT(:) .* Ts(:);