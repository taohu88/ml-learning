function T = convert_to_table(CPD, domain, evidence)

domain = domain(:);
CPT = CPD.CPT;
odom = domain(~isemptycell(evidence(domain)));
vals = cat(1, evidence{odom});
map = find_equiv_posns(odom, domain);
index = mk_multi_index(length(domain), map, vals);
T = CPT(index{:});
T = T(:);