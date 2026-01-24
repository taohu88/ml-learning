function T = multiply_pots(T1, T2)

dom = myunion(T1.domain, T2.domain);
ns = zeros(1, max(dom));
ns(T1.domain) = T1.sizes;
ns(T2.domain) = T2.sizes;
T = dpot(dom, ns(dom));
T = multiply_by_pot(T, T1);
T = multiply_by_pot(T, T2);   
