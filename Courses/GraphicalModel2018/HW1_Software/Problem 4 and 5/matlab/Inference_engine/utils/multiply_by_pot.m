function Tbig = multiply_by_pot(Tbig, Tsmall)
if issparse(Tbig.T) & issparse(Tsmall.T)
   Tbig.T = mult_by_sparse_table(Tbig.T, Tbig.domain, Tbig.sizes, Tsmall.T, Tsmall.domain, Tsmall.sizes);
else 
   Tbig.T = mult_by_table(Tbig.T, Tbig.domain, Tbig.sizes, Tsmall.T, Tsmall.domain, Tsmall.sizes);
end

