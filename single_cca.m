function [U, V, r] = single_cca(X_lt, Y_lt, X, Y, K, varargin)
    % SF-CCA
    % Input:
    
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2
    % K - dimension of canonical correlation
    
    % Optional:
    % lr1 - learning rate of U in lower level, dfault: 1e-4~1e-5
    % lr2 - learning rate of V in lower level, dfault: 1e-4~1e-5
    % lr3 - learning rate of U in upper level, dfault: 1e-4~1e-5
    % lr4 - learning rate of V in upper level, dfault: 1e-4~1e-5

    % lr - learning rate
    % lr_control - 'on' --> decreasing learning rate
    % rho - regularization term for disparity error
    
    % type - retraction type
    % typeG - 'fmincon' --> use fmincon to find descent direction
    % max_iter - maximum iteration for the optimization
    % tol1 - error tolerance of lower level optimization, default: 1e-6
    % tol2 - error tolerance of upper level optimization, default: 1e-6
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

    argin.addParameter('lr', 1e-2, @isnumeric);
    argin.addParameter('lr_control', 'off'); % 'on' or 'off'
    argin.addParameter('rho', 1, @isnumeric);
    argin.addParameter('type', 'ret');
    argin.addParameter('typeG', 'off');
    argin.addParameter('phi', 'abs');

    argin.addParameter('max_iter', 1e2, @isnumeric);
    argin.addParameter('tol1', 1e-6, @isnumeric);
    argin.addParameter('tol2', 1e-6, @isnumeric);
    
    argin.parse(X_lt, Y_lt, X, Y, K, varargin{:});

    lr = argin.Results.lr;
    lr_control = argin.Results.lr_control;
    rho = argin.Results.rho;
    type = argin.Results.type;
    typeG = argin.Results.typeG;
    phi = argin.Results.phi;
    
    max_iter = argin.Results.max_iter;
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
    
    %% Main loop
    options = optimoptions('fmincon', 'Display', 'off');
    for t = 1:max_iter
        if strcmpi(lr_control, 'on')
            lr = lr/sqrt(t);
        end
 
        grad1 = multi_obj_grad(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, 'phi', phi);
        dU = grad1{1};
        for i = 1:group_num
            dU = dU + rho*grad1{i};
        end
        if strcmpi(typeG, 'fmincon')
            objective1 = @(G) trace(G' * dU) + 0.5 * norm(G)^2;
            G0 = eye(size(dU));
            [G_opt, ~] = fmincon(objective1, G0, [], [], [], [], [], [], [], options);
            dU = - G_opt;
        end

        grad2 = multi_obj_grad(Y_lt, X_lt, Y, X, V_lt, U_lt, V, U, 'phi', phi);
        dV = grad2{1};
        for i = 1:group_num
            dV = dV + rho*grad2{i};
        end
        if strcmpi(typeG, 'fmincon')
            objective2 = @(G) trace(G' * dV) + 0.5 * norm(G)^2;
            G0 = eye(size(dV));
            [G_opt, ~] = fmincon(objective2, G0, [], [], [], [], [], [], [], options);
            dV = - G_opt;
        end
        
        for i = 1:K
            U(:,i) = U(:,i) - lr*adjust(i)*dU(:,i);
            V(:,i) = V(:,i) - lr*adjust(i)*dV(:,i);
        end
        
        if strcmpi(type, 'ret')
            U = retraction(U, X, 1);
            V = retraction(V, Y, 1);
        else
            U = U / sqrtm(U'*(X'*X)*U);
            V = V / sqrtm(V'*(Y'*Y)*V);
        end
        
        % Stop  criterion
        if norm(dU,'fro') <= tol1 && norm(DV,'fro') <= tol2
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