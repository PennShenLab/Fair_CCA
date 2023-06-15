function grad = pair_dis_grad(X_lt, Y_lt, U_lt, V_lt, U, V, k)
    % Input:
    % X_lt - list of matrices of dimensions NxM1
    % Y_lt - list of matrices of dimensions NxM2
    % U_lt - list of canonical weights of dimensions M1xk
    % V_lt - list of canonical weights of dimensions M2xk

    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk
    % k - index of group

    % Output:
    % The gradient of the pairwise disparity error of group k wrt Uk

    [n, K] = size(U);
    g = size(X_lt, 1);
    grad = zeros(n, K);
    disp1 = disparity(X_lt{k}, Y_lt{k}, U, V, U_lt{k}, V_lt{k});

    for j = 1:K
        for i = 1:g
            Uk = U(:,j);
            Vk = V(:,j);
            disp2 = disparity(X_lt{i}, Y_lt{i}, U, V, U_lt{i}, V_lt{i});
            dispg = disp_grad(X_lt{k}, Y_lt{k}, U_lt{k}(:,j), V_lt{k}(:,j), Uk, Vk);
            grad(:,j) = grad(:,j) + 2*(disp1(j) - disp2(j))*dispg;
        end
    end
end