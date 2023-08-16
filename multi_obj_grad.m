function grd_terms = multi_obj_grad(X_lt, Y_lt, X, Y, U_lt, V_lt, U, V, varargin)
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
    % The gradients of multi-objective function
    
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
    [n, K] = size(U);
    group_num = size(X_lt, 1);
    grd_terms = cell(group_num+1, 1);
    for i = 1:group_num+1
        grd_terms{i} = zeros(n,K);
    end

%     grd_terms{1} = loss_grad(X, Y, V);
%     for k = 1:group_num
%         grd_terms{k+1} = disp_grad(X_lt{k}, Y_lt{k}, U, V, U_lt{k}, V_lt{k});
%     end
    grd_terms{1} = loss_grad(X, Y, V);
    for i = 1:K
        for k = 1:group_num
            disp1 = disparity(X_lt{k}, Y_lt{k}, U, V, U_lt{k}, V_lt{k});
            for s = 1:group_num
                disp2 = disparity(X_lt{s}, Y_lt{s}, U, V, U_lt{s}, V_lt{s});
                disp_grad1 = disp_grad(X_lt{k}, Y_lt{k}, ...
                    U(:,i), V(:,i), U_lt{k}(:,i), V_lt{k}(:,i));
                disp_grad2 = disp_grad(X_lt{s}, Y_lt{s}, ...
                    U(:,i), V(:,i), U_lt{s}(:,i), V_lt{s}(:,i));
                if strcmpi(phi, 'abs')
                    grd_terms{k+1}(:, i) = grd_terms{k+1}(:, i) + ...
                        sign(disp1(i) - disp2(i))*(disp_grad1 - disp_grad2);
                end
                if strcmpi(phi, 'exp')
                    grd_terms{k+1}(:, i) = grd_terms{k+1}(:, i) + ...
                        exp(disp1(i) - disp2(i))*(disp_grad1 - disp_grad2);
                end
                if strcmpi(phi, 'square')
                    grd_terms{k+1}(:, i) = grd_terms{k+1}(:, i) + ...
                        2*(disp1(i) - disp2(i))*(disp_grad1 - disp_grad2);
                end
            end
        end
    end
end
