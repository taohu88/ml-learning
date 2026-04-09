function engine = var_elim_inf_engine(bnet, varargin)
engine.evidence = [];

engine = class(engine, 'var_elim_inf_engine', inf_engine(bnet));
