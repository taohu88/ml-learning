function pot = mk_initial_pot(dom, ns, onodes)
ns(onodes) = 1;
pot = dpot(dom, ns(dom));
end