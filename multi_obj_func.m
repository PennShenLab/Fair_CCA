function obj_terms = multi_obj_func(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V)
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

    group_num = size(X_lt, 1);
    obj_terms = cell(1, group_num+1);
    obj_terms{1} = - diag(corr(X*U, Y*V));
    for i = 1:group_num
        obj_terms{i+1} = abs(disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i}));
    end
end