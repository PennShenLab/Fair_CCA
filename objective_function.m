function max_terms = objective_function(X_lt, Y_lt, X, Y, ...
    U_lt, V_lt, U, V, alpha, rho, lambda, varargin)
    % Input:
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2
    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk
    
    % lambda - weights for gradients of multi-objective function
    % alpha - regularization term for convex objective function
    % rho - balance between fairness and correlation

    % Output:
    % The objective function of the min-max optimization finding the
    % descent direction
    
    %% Parse inputs
    argin = inputParser;
    argin.addRequired('X_lt');
    argin.addRequired('Y_lt');
    argin.addRequired('X');
    argin.addRequired('Y');
    argin.addRequired('U_lt');
    argin.addRequired('V_lt');
    argin.addRequired('U');
    argin.addRequired('V');
    argin.addRequired('alpha');
    argin.addRequired('rho');
    argin.addRequired('lambda');

    argin.addParameter('phi', 'abs');
    argin.parse(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, alpha, rho, lambda, varargin{:});
    phi = argin.Results.phi;

    %%
    group_num = size(X_lt, 1);
    max_terms = zeros(1, group_num+1);
    
    grd_terms = multi_obj_grad(X_lt,Y_lt,X,Y,U_lt,V_lt,U,V,'phi',phi);
    for i = 1:size(lambda)
        grd_terms{i} = grd_terms{i} * diag(1./sqrt(sum(grd_terms{i}.^2))) + alpha*U;
    end
    for i = 2:size(lambda)
        grd_terms{i} = rho * grd_terms{i};
    end

    descen_dr = zeros(size(U, 1), size(U, 2));
    for i = 1:size(lambda)
        descen_dr = descen_dr + lambda(i)*grd_terms{i};
    end
    
    for i = 1:group_num+1
        max_terms(i) = trace(-descen_dr'*grd_terms{i}) + 1/2*norm(descen_dr,"fro");
    end
end