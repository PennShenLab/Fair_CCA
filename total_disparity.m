function error = total_disparity(X_lt, Y_lt, U_lt, V_lt, U, V)
    % Input:
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2

    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk

    % Output:
    % The pairwise disparity error of k-th group

    error = 0;
    for i = 1:size(X_lt, 1)
        for j = 1:size(X_lt, 1)
            error = error + (disparity(X_lt{j}, Y_lt{j}, U, V, U_lt{j}, V_lt{j}) - disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i})).^2;
        end
    end
end