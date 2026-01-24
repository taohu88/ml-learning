function [engine, loglik] = enter_evidence(engine, evidence, varargin)

engine.evidence = evidence;

if nargout == 2
  [m, loglik] = marginal_nodes(engine, [1]);
end
