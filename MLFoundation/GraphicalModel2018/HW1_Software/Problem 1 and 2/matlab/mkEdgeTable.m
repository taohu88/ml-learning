function [ EdgeTable ] = mkEdgeTable( edges )
%MKEDGETABLE makes a edge table for DIGRAPH
    EdgeTable = table(edges,'VariableNames',{'EndNodes'});
end

