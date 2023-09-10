function [X, Y, C] = simcca(V, W, rho, n, varargin)
% SIMCCA Simulate joint random normal deviates with the requested canonical
%   vectors (V, W) and coefficients rho. We model the free parameter matrices
%   Tx and Ty as the diagonal matrices noisex*I and noisey*I. Note that the
%   population covariance matrices will be singular unless noisex > 0 and
%   noisey > 0
%
% INPUT:
%   V: px-by-d left canonical vectors
%   W: py-by-d right canonical vectors
%   rho: d-by-1 canonical correlations between 0 and 1
%   n: number of replicates to be simulated
%
% OPTIONAL NAME-VALUE PAIRs:
%   mux: mean of X vectors
%   muy: mean of Y vectors
%   noisex: noise level in X vectors
%   noisey: noise level in Y vectors
%
% OUTPUT:
%   X: n-by-px simulated x variables
%   Y: n-by-py simulated y variables
%   C: px+py-by-px+py Covariance matrix
%
% Reference: Eun Jeong Min, Eric Chi, and Hua Zhou. Tensor Canonical
% Correlation Analysis. Stat. 2019+

%% parse inputs
argin = inputParser;
argin.addRequired('V', @isnumeric);
argin.addRequired('W', @isnumeric);
argin.addRequired('rho', @isnumeric);
argin.addRequired('n', @isnumeric);
argin.addParameter('mux', zeros(size(V, 1), 1), @isnumeric);
argin.addParameter('muy', zeros(size(W, 1), 1), @isnumeric);
argin.addParameter('noisex', 0, @(x) x>=0);
argin.addParameter('noisey', 0, @(x) x>=0);
argin.parse(V, W, rho, n, varargin{:});
mux = argin.Results.mux;
muy = argin.Results.muy;
noisex = argin.Results.noisex;
noisey = argin.Results.noisey;

% retrieve dimension
px = size(V, 1);
py = size(W, 1);
d = size(V, 2);

% "Economy Size" QR decomposition
[Qx, Rx] = qr(V, 0);
[Qy, Ry] = qr(W, 0);

% Construct the joint covariance matrix
Tx = randn(px, px);
TxQx = Tx * Qx;
Ty = randn(py, py);
TyQy = Ty * Qy;
Cxx = Qx / Rx';
Cyy = Qy / Ry';
Cxy = bsxfun(@times, Cxx, reshape(rho, 1, d)) * Cyy';
Cxx = Cxx * Cxx' + noisex * (Tx * Tx' - TxQx * TxQx');
Cyy = Cyy * Cyy' + noisey * (Ty * Ty' - TyQy * TyQy');
C = [Cxx Cxy; Cxy' Cyy];

% simulate random normal deviates
RV = mvnrnd([mux' muy'], C, n);
X = RV(:, 1:px);
Y = RV(:, px+1:end);
