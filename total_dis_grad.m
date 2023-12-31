function grad = total_dis_grad(X_lt, Y_lt, U_lt, V_lt, U, V, varargin)
    % Input:
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk

    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk

    % Output:
    % The gradient of the total pairwise disparity error wrt U
    
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
    [n, K] = size(U);
    g = size(X_lt, 1);
    grad = zeros(n, K);
    
    for i = 1:K
        for k = 1:g
            disp1 = disparity(X_lt{k}, Y_lt{k}, U, V, U_lt{k}, V_lt{k});
            for s = 1:g
                disp2 = disparity(X_lt{s}, Y_lt{s}, U, V, U_lt{s}, V_lt{s});
                disp_grad1 = disp_grad(X_lt{k}, Y_lt{k}, U(:,i), V(:,i), U_lt{k}(:,i), V_lt{k}(:,i));
                disp_grad2 = disp_grad(X_lt{s}, Y_lt{s}, U(:,i), V(:,i), U_lt{s}(:,i), V_lt{s}(:,i));
                if strcmpi(phi, 'abs')
                    grad(:, i) = grad(:, i) + sign(disp1(i) - disp2(i))*(disp_grad1 - disp_grad2);
                end
                if strcmpi(phi, 'exp')
                    grad(:, i) = grad(:, i) + exp(disp1(i) - disp2(i))*(disp_grad1 - disp_grad2);
                end
                if strcmpi(phi, 'square')
                    grad(:, i) = grad(:, i) + 2*(disp1(i) - disp2(i))*(disp_grad1 - disp_grad2);
                end
            end
        end
    end
end