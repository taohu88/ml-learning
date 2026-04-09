% =============================================================
% Problem 1 and 2, Homework 1
% Center for Informatics and Computational Science
% University of Notre Dame
% =============================================================
clear;
close all;
clc;

%% Solution 1
% Variable values:
%%
% Matlab code:
F = 1; % false
T = 2; % true
%%
% Joint distribution:
%%
% Matlab code:
P = zeros(2,2,2);
% x,y,z
P(F,F,F) = 0.192;
P(F,F,T) = 0.144;
P(F,T,F) = 0.048;
P(F,T,T) = 0.216;
P(T,F,F) = 0.192;
P(T,F,T) = 0.064;
P(T,T,F) = 0.048;
P(T,T,T) = 0.096;
%%
% a)
% 
% $$p(x) = \sum_{y=F}^T \sum_{z=F}^T p(x,y,z)$$
% 
%%
% Matlab code:
Px = [sum(sum(P(F,:,:))),sum(sum(P(T,:,:)))]';
disp(['Solution:',newline,'p(x = F) = ', num2str(Px(F)),...
                  newline, 'P(x = T) = ', num2str(Px(T))]);
%%
% 
% $$p(z|x) = \frac{p(x,z)}{p(x)} = \frac{\sum_{y=F}^T p(x,y,z)}{p(x)}$$
% 
%%
% Matlab code:
Pz_x = zeros(2,2);
%    z,x
Pz_x(F,F) = sum(P(F,:,F))/Px(F);
Pz_x(F,T) = sum(P(T,:,F))/Px(T);
Pz_x(T,F) = sum(P(F,:,T))/Px(F);
Pz_x(T,T) = sum(P(T,:,T))/Px(T);
disp(['Solution:',newline,'p(z=F|x=F) = ', num2str(Pz_x(F,F)),...
                  newline,'p(z=F|x=T) = ', num2str(Pz_x(F,T)),...
                  newline,'p(z=T|x=F) = ', num2str(Pz_x(T,F)),...
                  newline,'p(z=T|x=T) = ', num2str(Pz_x(T,T))]);
%%
% 
% $$p(y|z) = \frac{p(y,z)}{p(z)} = \frac{\sum_{x=F}^T p(x,y,z)}{\sum_{x=F}^T \sum_{y=F}^T p(x,y,z)}$$
% 
%%
% Matlab code:
Pz = [sum(sum(P(:,:,F))),sum(sum(P(:,:,T)))]';
Py_z = zeros(2,2);
%    y,z
Py_z(F,F) = sum(P(:,F,F))/Pz(F);
Py_z(F,T) = sum(P(:,F,T))/Pz(T);
Py_z(T,F) = sum(P(:,T,F))/Pz(F);
Py_z(T,T) = sum(P(:,T,T))/Pz(T);
disp(['Solution:',newline,'p(y=F|z=F) = ', num2str(Py_z(F,F)),...
                  newline,'p(y=F|z=T) = ', num2str(Py_z(F,T)),...
                  newline,'p(y=T|z=F) = ', num2str(Py_z(T,F)),...
                  newline,'p(y=T|z=T) = ', num2str(Py_z(T,T))]);
%%
% b) 
% 
% $$x \not\perp y \Leftrightarrow p(x,y) \neq
% p(x)p(y)$$
% 
% $$ p(x,y) = \sum_{z=F}^T p(x,y,z) \neq \Big(\sum_{y=F}^T \sum_{z=F}^T 
% p(x,y,z)\Big) \Big(\sum_{x=F}^T \sum_{z=F}^T p(x,y,z)\Big) = p(x)p(y)$$
%
%%
% Matlab code:
Py = [sum(sum(P(:,F,:))),sum(sum(P(:,T,:)))]';
Pxy = zeros(2,2);
%   x,y
Pxy(F,F) = sum(P(F,F,:));
Pxy(F,T) = sum(P(F,T,:));
Pxy(T,F) = sum(P(T,F,:));
Pxy(T,T) = sum(P(T,T,:));
disp(['Solution:',newline,'p(x=F,y=F) = ', num2str(Pxy(F,F)),...
                      ' ~= p(x=F)p(y=F) = ', num2str(Px(F)*Py(F)),...
                  newline,'p(x=F,y=T) = ', num2str(Pxy(F,T)),...
                      ' ~= p(x=F)p(y=F) = ', num2str(Px(F)*Py(T)),...
                  newline,'p(x=T,y=F) = ', num2str(Pxy(T,F)),...
                      ' ~= p(x=F)p(y=F) = ', num2str(Px(T)*Py(F)),...
                  newline,'p(x=T,y=T) = ', num2str(Pxy(T,T)),...
                      ' ~= p(x=F)p(y=F) = ', num2str(Px(T)*Py(T))]);
%% 
% Therefore, 
% 
% $$x \not\perp y$$
% 
%%
% c) Using the results from (a).
% 
%
% Matlab code:
v = 'FT';
disp('Solution:');
for x=[F,T]
    for y=[F,T]
        for z=[F,T]
            disp(['p(x=',v(x),',y=',v(y),',z=',v(z),') = ', ...
                                                    num2str(P(x,y,z)),...
               ' = p(x=',v(x),')p(y=',v(y),'|z=',v(z),')p(z=',v(z),'|',...
               'x=',v(x),') = ', num2str(Px(x)*Py_z(y,z)*Pz_x(z,x))]);
        end
    end
end
%% 
% Therefore, 
% 
% $$p(x,y,z)=p(x)p(y|z)p(z|x)$$,
% 
% and the resulting direct graph:
%
% Matlab code:
plotdigraph(digraph(mkEdgeTable([1 3; 3 2]),mkNodeTable({'x';'y';'z'})),...
    'NodeFont',24);

%% Solution 2
p = plotdigraph(digraph(mkEdgeTable([2 1; 3 1; 4 1; 5 3]),...
    mkNodeTable({'t_n'; '\bf{x}_n'; 'w_i'; '\beta';'\alpha_i'})),...
    'NodeFont',24,'PlateEdges',{[2 1],'N';[5 3],'M'});
%% Solution 3
%
% Considering a DAG shown in following figure:
plotdigraph(digraph(mkEdgeTable([1 2; 2 3]),...
    mkNodeTable({'x_1'; 'x_2'; 'x_3'})),'NodeFont',24);
%%
% such that $x_1$, $x_2$ and $x_3$ follow a Gaussian distribution. Since,
%
% $$p(x_i|Pa_i) = \mathcal{N}\Big(x_i;\sum_{j \in Pa_i}w_{ij}x_i +
% b_i,\nu_i \Big) = \sum_{j \in Pa_i}w_{ij}x_i + b_i + \sqrt{\nu_i}\varepsilon_i, $$
%
% where
%
% $$ \varepsilon_i \sim \mathcal{N}(0,1), E[\varepsilon_i] = 0,
% E[\varepsilon_i\varepsilon_j] = I_{i,j},$$
%
% and $I_{i,j}$ is the $(i,j)-$th element of an identity matrix.
%
% Hence,
%
% $E[x_i] = E[\sum_{j \in Pa_i}w_{ij}x_i + b_i +
% \sqrt{\nu_i}\varepsilon_i]$
%
% $= E[\sum_{j \in Pa_i}w_{ij}x_i] + E[b_i] +
% E[\sqrt{\nu_i}\varepsilon_i]$
%
% $= \sum_{j \in Pa_i}w_{ij}E[x_i] + b_i +
% \sqrt{\nu_i}E[\varepsilon_i]$
%
% $= \sum_{j \in Pa_i}w_{ij}E[x_i] + b_i,$
%
% and
%
% $cov[x_i,x_j] = E[(x_i-E[x_i])(x_j-E[x_j])]$
%
% $= E\left[(x_i-E[x_i])\Big(\sum_{k \in Pa_j}w_{jk}(x_k-E[x_k]) + \sqrt{\nu_j}\varepsilon_j\Big)\right]$
%
% $= \sum_{k \in Pa_j}w_{jk}E[(x_i-E[x_i])(x_k-E[x_k])] + \sqrt{\nu_j}E[(x_i-E[x_i])\varepsilon_j]$
%
% $= \sum_{k \in Pa_j}w_{jk}cov[x_i,x_k] + I_{ij}\nu_j$

