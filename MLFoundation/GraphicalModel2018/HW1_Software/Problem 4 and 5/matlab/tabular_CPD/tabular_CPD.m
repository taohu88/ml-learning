function CPD = tabular_CPD(bnet, self, varargin)
% Make a multinomial conditional prob. distrib. (CPT)
% e.g., tabular_CPD(bnet, i, 'CPT', T)

CPD = init_fields;

ns = bnet.node_sizes;
ps = find_parents(bnet.dag, self);
fam_sz = ns([ps self]);
CPD.sizes = fam_sz;

% extract optional args
args = varargin;

CPD.CPT = myreshape(args{1}, fam_sz);



%%%%%%%%%%%

function CPD = init_fields()
% This ensures we define the fields in the same order 
% no matter whether we load an object from a file,
% or create it from scratch. (Matlab requires this.)

CPD.CPT = [];
CPD.sizes = [];

