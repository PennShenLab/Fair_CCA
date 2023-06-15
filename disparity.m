function error =  disparity(X, Y, U, V, Uk, Vk)
    % Input: 
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk
    % Uk - canonical weight of group k of dimensions M1xk
    % Vk - canonical weight of group k of dimensions M2xk

    % Output:
    % The disparity error of k-th group
    
    error = abs(diag(- corr(X*U, Y*V) + corr(X*Uk, Y*Vk)));
end