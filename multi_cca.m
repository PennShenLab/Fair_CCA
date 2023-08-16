function [U, V, r] = multi_cca(X_lt, Y_lt, X, Y, K, varargin)
    % MF-CCA
    % Input:
    
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2
    % K - dimension of canonical correlation
    % max_iter - maximum iteration for the optimization
    % alpha - regularization term for convex objective function
    % lr - learning rate
    % lr_control - 'on' --> decreasing learning rate
    % rho - regularization term for disparity error
    % tol1 - error tolerance of lower level optimization, default: 1e-6
    % tol2 - error tolerance of upper level optimization, default: 1e-6
    % type - retraction type
    % phi - penalty function, default: abs()

    % Output:
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk
    % r - canonical correlations
    
    %% Parse inputs
    argin = inputParser;
    argin.addRequired('X_lt', @iscell);
    argin.addRequired('Y_lt', @iscell);
    argin.addRequired('X', @ismatrix);
    argin.addRequired('Y', @ismatrix);
    argin.addRequired('K', @isnumeric);

    argin.addParameter('max_iter', 10, @isnumeric);
    argin.addParameter('alpha', 1e-6, @isnumeric);
    argin.addParameter('lr', 1e-1, @isnumeric);
    argin.addParameter('lr_control', 'off');
    argin.addParameter('rho', 1, @isnumeric);
    argin.addParameter('tol1', 1e-6, @isnumeric);
    argin.addParameter('tol2', 1e-6, @isnumeric);
    argin.addParameter('type', 'ret');
    argin.addParameter('phi', 'abs');

    argin.parse(X_lt, Y_lt, X, Y, K, varargin{:});
    max_iter = argin.Results.max_iter;
    alpha = argin.Results.alpha;
    lr = argin.Results.lr;
    lr_control = argin.Results.lr_control;
    type = argin.Results.type;
    phi = argin.Results.phi;

    rho = argin.Results.rho;
    tol1 = argin.Results.tol1;
    tol2 = argin.Results.tol2;
    
    count = 0;
    adjust = ones(K, 1);

    %% Normalization
    X = X - mean(X, 1);
    Y = Y - mean(Y, 1);

    group_num = size(X_lt, 1);
    for g = 1:group_num
        X_lt{g} = X_lt{g} - mean(X_lt{g}, 1);
        Y_lt{g} = Y_lt{g} - mean(Y_lt{g}, 1);
    end

    %% Initialize canonical weights

    [U, V, ~] = canoncorr(X, Y);
    U = U(:, 1:K);
    V = V(:, 1:K);
    
    U_lt = cell(group_num, 1);
    V_lt = cell(group_num, 1);
    for g = 1:group_num
        [Uk, Vk, ~] = canoncorr(X_lt{g}, Y_lt{g});
        U_lt{g} = Uk(:, 1:K);
        V_lt{g} = Vk(:, 1:K);
    end

    %% main loop
    for t = 1:max_iter
        if strcmpi(lr_control, 'on')
            lr = lr/sqrt(t);
        end
        options = optimoptions('fminimax', 'Display', 'off');

        % Optimize U
        l_init = rand(group_num + 1, 1);
        [l_u, ~] = fminimax(@(lambda) objective_function( ...
            X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, alpha, rho, lambda, 'phi', phi), ...
            l_init, [], [], [], [], [], [], @unitdisk, options);
        D_u = zeros(size(X, 2), K);
        grd_terms = multi_obj_grad(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, 'phi', phi);
        for i = 1:group_num+1
            D_u = D_u + l_u(i)*grd_terms{i};
        end

        % Optimize V
        l_init = rand(group_num + 1, 1);
        [l_v, ~] = fminimax(@(lambda) objective_function( ...
            Y_lt, X_lt, Y, X, V_lt, U_lt, V, U, alpha, rho, lambda, 'phi', phi), ...
            l_init, [], [], [], [], [], [], @unitdisk, options);
        D_v = zeros(size(Y, 2), K);
        grd_terms = multi_obj_grad(Y_lt, X_lt, Y, X, V_lt, U_lt, V, U, 'phi', phi);
        for i = 1:group_num+1
            D_v = D_v + l_v(i)*grd_terms{i};
        end
        
        % Update canonical weights
        for i = 1:K
            U(:,i) = U(:,i) - lr*adjust(i)*D_u(:,i);
            V(:,i) = V(:,i) - lr*adjust(i)*D_v(:,i);
        end
        if strcmpi(type, 'ret')
            U = retraction(U, X, 1);
            V = retraction(V, Y, 1);
        else
            U = U / sqrtm(U'*(X'*X)*U);
            V = V / sqrtm(V'*(Y'*Y)*V);
        end
        
        % Stop criterion
        if norm(D_u,'fro') <= tol1 && norm(D_v,'fro') <= tol2
            count = count + 1;
        end
        if count > 5
            break
        else
            count = 0;
        end

    end
    r = diag(corr(X*U, Y*V));
end