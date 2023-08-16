function error = total_disparity(X_lt, Y_lt, U_lt, V_lt, U, V, varargin)
    % Input:
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2

    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk

    % Output:
    % The pairwise disparity error of k-th group
    
    %% Parse inputs
    argin = inputParser;
    argin.addRequired('X_lt');
    argin.addRequired('Y_lt');
    argin.addRequired('U_lt');
    argin.addRequired('V_lt');
    argin.addRequired('U');
    argin.addRequired('V');

    argin.addParameter('phi', 'abs');
    argin.parse(X_lt, Y_lt, U_lt, V_lt, U, V, varargin{:});
    phi = argin.Results.phi;

    %%
    error = 0;
    for i = 1:size(X_lt, 1)
        for j = 1:size(X_lt, 1)
            if strcmpi(phi, 'abs')
                error = error + abs(disparity(X_lt{j}, Y_lt{j}, U, V, U_lt{j}, V_lt{j}) - disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i}));
            end
            if strcmpi(phi, 'exp')
                error = error + exp(disparity(X_lt{j}, Y_lt{j}, U, V, U_lt{j}, V_lt{j}) - disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i}));
            end
            if strcmpi(phi, 'square')
                error = error + (disparity(X_lt{j}, Y_lt{j}, U, V, U_lt{j}, V_lt{j}) - disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i})).^2;
            end
        end
    end
end