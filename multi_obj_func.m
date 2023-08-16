function obj_terms = multi_obj_func(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, varargin)
    % Input: 
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2
    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk

    % Output:
    % The multi-objective function to be minimized
    
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

    argin.addParameter('phi', 'abs');
    argin.parse(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, varargin{:});
    phi = argin.Results.phi;

    %%
    group_num = size(X_lt, 1);
    obj_terms = cell(1, group_num+1);
    obj_terms{1} = - diag(corr(X*U, Y*V));
    for i = 1:group_num
        if strcmpi(phi, 'abs')
            obj_terms{i+1} = abs(disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i}));
        end
        if strcmpi(phi, 'exp')
            obj_terms{i+1} = exp(disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i}));
        end
        if strcmpi(phi, 'square')
            obj_terms{i+1} = (disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i})).^2;
        end
    end
end