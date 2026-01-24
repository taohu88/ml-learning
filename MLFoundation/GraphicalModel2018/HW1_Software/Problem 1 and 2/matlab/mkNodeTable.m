function [ NodeTable ] = mkNodeTable( names )
%MKNODETABLE makes a node table for DIGRAPH
    NodeTable = table(names,'VariableNames',{'Name'});
end

