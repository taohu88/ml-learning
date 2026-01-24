function f = family(A,i,t)

if nargin < 3 
  f = [find_parents(A,i) i];
else
  if t == 1
    f = [find_parents(A,i) i];
  else
    ss = length(A)/2;
    j = i+ss;
    f = [find_parents(A,j) j] + (t-2)*ss;
  end
end
