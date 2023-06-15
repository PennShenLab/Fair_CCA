function error = pair_disparity(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, Uk, Vk)
    % Input:
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2

    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk
    % Uk - canonical weight of group k of dimensions M1xk
    % Vk - canonical weight of group k of dimensions M2xk
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk

    % Output:
    % The pairwise disparity error of k-th group

    error = 0;
    for i = 1:size(X_lt, 1)
        error = error + (disparity(X, Y, U, V, Uk, Vk) - ...
            disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i})).^2;
    end
end