function grad = disp_grad(Xk, Yk, U, V, Uk, Vk, varargin)
    % Input:
    % Xk - matrix of dimensions NxM1
    % Yk - matrix of dimensions NxM2
    % U - canonical weight of dimensions M1xk
    % V - canonical weight of dimensions M2xk
    % Uk - canonical weight of group k of dimensions M1xk
    % Vk - canonical weight of group k of dimensions M2xk

    % Output:
    % The gradient of the disparity error wrt U

    %% Parse inputs
    argin = inputParser;
    argin.addRequired('Xk');
    argin.addRequired('Yk');
    argin.addRequired('U');
    argin.addRequired('V');
    argin.addRequired('Uk');
    argin.addRequired('Vk');

    argin.addParameter('phi', 'abs');
    argin.parse(Xk, Yk, U, V, Uk, Vk, varargin{:});
    phi = argin.Results.phi;

    %%
    [n, K] = size(U);
    grad = zeros(n, K);
    disp = disparity(Xk, Yk, U, V, Uk, Vk);

    for i = 1:K
        if strcmpi(phi, 'abs')
            grad(:,i) = loss_grad(Xk, Yk, V(:,i))*sign(disp(i));
        end
        if strcmpi(phi, 'exp')
            grad(:,i) = loss_grad(Xk, Yk, V(:,i))*exp(disp(i));
        end
        if strcmpi(phi, 'square')
            grad(:,i) = loss_grad(Xk, Yk, V(:,i))*2*(disp(i));
        end
    end
end