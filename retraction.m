function [X]= retraction(Y, B, type)
% Reference: https://arxiv.org/abs/1903.11576
% modified from AManPG-master
% Y: tangent point;
% B: M = BTB;
% M: generalized Stiefel manifold;
% type = 1;   using covariance matrix
% otherwise   data matrix preferred

M = B'*B;
m = size(B,1);
[u, ~, v] = svd(Y, 0);

if type == 1
    [q, ssquare] = eig(u'*(M*u));
else
    [q, ssquare] = eig(u'*(B'*(B*u))/(m-1));
end

qsinv = q/sparse(diag(sqrt(diag(ssquare))));
X = u*((qsinv*q')*v'); % X'*B*X is identity.
