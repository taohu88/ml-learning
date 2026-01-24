function [M, z] = normalise(A, dim)

if nargin < 2
  z = sum(A(:));
  s = z + (z==0);
  M = A / s;
elseif dim==1 % normalize each column
  z = sum(A);
  s = z + (z==0);
  M = A ./ repmatC(s, size(A,1), 1);
else
  z=sum(A,dim);
  s = z + (z==0);
  L=size(A,dim);
  d=length(size(A));
  v=ones(d,1);
  v(dim)=L;
  c=repmat(s,v');
  M=A./c;
end