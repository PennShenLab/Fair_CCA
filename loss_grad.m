function grad = loss_grad(X, Y, V)
    % Input:
    % X - matrix of dimensions NxM1
    % Y - matrix of dimensions NxM2
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk

    % Output:
    % The gradient of the loss function wrt U
    
    n = size(X, 1);
    grad = - X'*Y*V / (n-1);

end