function grad = disp_grad(Xk, Yk, U, V, Uk, Vk)
    % Input:
    % Xk - matrix of dimensions NxM1
    % Yk - matrix of dimensions NxM2
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk
    % Uk - canonical weight of group k of dimensions M1xk
    % Vk - canonical weight of group k of dimensions M2xk

    % Output:
    % The gradient of the disparity error wrt U
    
    [n, K] = size(U);
    grad = zeros(n, K);
    disp = disparity(Xk, Yk, U, V, Uk, Vk);

    for i = 1:K
        grad(:,i) = loss_grad(Xk, Yk, V(:,i))*sign(disp(i));
    end
end