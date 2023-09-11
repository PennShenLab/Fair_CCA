%% Synthetic data

% ground truth
U1 = zeros(100, 5);
U1([13 24 57 73 92], 1) = 1;
U1([17 33 49 75 86], 2) = 1;
U1([23 45 67 79 83], 3) = 1;
U1([8 37 48 53 60], 4) = 1;
U1([3 25 44 70 95], 5) = 1;

V1 = zeros(100, 1);
V1([4 19 33 67 83], 1) = 1;
V1([6 14 18 29 56], 2) = 1;
V1([9 18 25 44 88], 3) = 1;
V1([14 29 47 75 93], 4) = 1;
V1([18 24 45 84 97], 5) = 1;

U2 = zeros(100, 1);
U2([13 24 57 73 92], 1) = 1;
U2([17 33 49 75 86], 2) = 1;
U2([23 45 67 79 83], 3) = 1;
U2([8 37 48 53 60], 4) = 1;
U2([3 25 44 70 95], 5) = 1;
U2(57, 1) = 0.5;
U2(77, 1) = 0.7;
U2(49, 2) = 1.3;
U2(44, 3) = 0.8;
U2(58, 3) = 0.7;
U2(88, 4) = 1.1;
U2(70, 5) = 1.5;
U2(71, 5) = 0.7;

V2 = zeros(100, 1);
V2([4 19 33 67 83], 1) = 1;
V2([6 14 18 29 56], 2) = 1;
V2([9 18 25 44 88], 3) = 1;
V2([14 29 47 75 93], 4) = 1;
V2([18 24 45 84 97], 5) = 1;
V2(33, 1) = 0.9;
V2(99, 1) = 0.3;
V2(43, 2) = 1.2;
V2(58, 3) = 0.7;
V2(39, 4) = 1.3;
V2(48, 5) = 0.9;
V2(97, 5) = 1.6;
V2(44, 5) = 0.8;

U3 = zeros(100, 1);
U3([13 24 57 73 92],1) = 1;
U3([17 33 49 75 86], 2) = 1;
U3([23 45 67 79 83], 3) = 1;
U3([8 37 48 53 60], 4) = 1;
U3([3 25 44 70 95], 5) = 1;
U3(57, 1) = 0.8;
U3(49, 2) = 1.3;
U3(44, 3) = 0.8;
U3(58, 3) = 0.7;
U3(60, 4) = 0.8;
U3(71, 5) = 1.2;

V3 = zeros(100, 1);
V3([4 19 33 67 83], 1) = 1;
V3([6 14 18 29 56], 2) = 1;
V3([9 18 25 44 88], 3) = 1;
V3([14 29 47 75 93], 4) = 1;
V3([18 24 45 84 97], 5) = 1;
V3(33, 1) = 0.2;
V3(57, 1) = 1.2;
V3(57, 4) = 0.9;
V3(61, 5) = 0.3;

mux = zeros(size(U, 1), 1)
muy = zeros(size(V, 1), 1)

% simulate joint normal random deviates
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
g = 3;

n1 = 1000;
rho1 = [0.95; 0.9; 0.85; 0.7; 0.5];
[X1, Y1] = simcca(U1, V1, rho1, n1, 'noisex', 1, 'noisey', 1e-1, 'mux', mux, 'muy', muy);

n2 = 150;
rho2 = [0.8; 0.77; 0.75; 0.7; 0.6];
[X2, Y2] = simcca(U2, V2, rho1, n2, 'noisex', 1, 'noisey', 1e-1, 'mux', mux, 'muy', muy);

n3 = 200;
rho3 = [0.7; 0.68; 0.65; 0.62; 0.59];
[X3, Y3] = simcca(U3, V3, rho1, n3, 'noisex', 1, 'noisey', 1e-1, 'mux', mux, 'muy', muy);

X_lt = cell(g, 1);
Y_lt = cell(g, 1);
X_lt{1} = X1;
X_lt{2} = X2;
X_lt{3} = X3;
Y_lt{1} = Y1;
Y_lt{2} = Y2;
Y_lt{3} = Y3;

X = [X1; X2; X3];
Y = [Y1; Y2; Y3];


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
% Modified based on:
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
r = size(V, 2);

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
Cxy = bsxfun(@times, Cxx, reshape(rho, 1, r)) * Cyy';
Cxx = Cxx * Cxx' + noisex * (Tx * Tx' - TxQx * TxQx');
Cyy = Cyy * Cyy' + noisey * (Ty * Ty' - TyQy * TyQy');
C = [Cxx Cxy; Cxy' Cyy];

% simulate random normal deviates
RV = mvnrnd([mux' muy'], C, n);
X = RV(:, 1:px);
Y = RV(:, px+1:end);

end
