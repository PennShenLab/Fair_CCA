close all
clear
clc

format long

K = 7; G = 5; id = 0; alpha = 1e-5;
lr_b = 1e-2; lr_control_b = 'on';
lr_f = 2e-2; lr_control_f = 'on';
max_iter_s = 10; max_iter_m = 10;

%% Synthetic data

U = zeros(100, K);
U([13 24 57 73 92], 1) = 1;
U([17 33 49 75 86], 2) = 1;
U([23 45 67 79 83], 3) = 1;
U([8 37 48 53 60], 4) = 1;
U([3 25 44 70 95], 5) = 1;
U([9 17 23 45 67], 6) = 1;
U([23 45 64 83 92], 7) = 1;

V = zeros(100, K);
V([4 19 33 67 83], 1) = 1;
V([6 14 18 29 56], 2) = 1;
V([9 18 25 44 88], 3) = 1;
V([14 29 47 75 93], 4) = 1;
V([18 24 45 84 97], 5) = 1;
V([22 37 42 82 95], 6) = 1;
V([18 42 56 74 93], 7) = 1;

X = []; Y = [];
X_lt = cell(G, 1); Y_lt = cell(G, 1);

n = 500;
rho_temp = [0.95; 0.9; 0.85; 0.7; 0.55; 0.35; 0.1];
for i = 1:G
    U_temp = U + 1e-3*randn(100, K);
    V_temp = V + 1e-3*randn(100, K);
    [X_temp, Y_temp] = simcca(U_temp, V_temp, rho_temp-0.1*(i-1), n-(i-1)*50, ...
        'noisex', 1, 'noisey', 1e-1);
    X_lt{i} = X_temp; Y_lt{i} = Y_temp;
    X = [X; X_temp]; Y = [Y; Y_temp];
end

%% Initialize canonical weights
X = X - mean(X, 1);
Y = Y - mean(Y, 1);
X = normalize(X, 1);
Y = normalize(Y, 1);

for g = 1:G
    X_lt{g} = X_lt{g} - mean(X_lt{g}, 1);
    Y_lt{g} = Y_lt{g} - mean(Y_lt{g}, 1);
    X_lt{g} = normalize(X_lt{g}, 1);
    Y_lt{g} = normalize(Y_lt{g}, 1);
end

U_lt = cell(G, 1);
V_lt = cell(G, 1);
for g = 1:G
    [Uk, Vk, ~] = canoncorr(X_lt{g}, Y_lt{g});
    U_lt{g} = Uk(:, 1:K);
    V_lt{g} = Vk(:, 1:K);
end

disp(['ID: ', num2str(id)])

dim = 1:K;
time = zeros(3, 1);
disparities = zeros(G*3, K);
correlation = zeros((G+1)*3, K);
correlation2 = zeros(G, K);
pairdisparity = zeros((G+1)*3, K);

%% Experiment of CCA

t1 = tic;
[U_pred, V_pred, r_pred] = canoncorr(X, Y);
time(1) = toc(t1);
U_pred = U_pred(:, 1:K);
V_pred = V_pred(:, 1:K);
disp('Classical CCA:')
correlation(1, :) = diag(corr(X*U_pred,Y*V_pred));
pairdisparity(1, :) = total_disparity(X_lt, Y_lt, U_lt, V_lt, U_pred, V_pred);
for i = 1:G
    disparities(i, :) = disparity(X_lt{i}, Y_lt{i}, U_pred, V_pred, U_lt{i}, V_lt{i});
    correlation2(i, :) = diag(corr(X_lt{i}*U_lt{i},Y_lt{i}*V_lt{i}));
    correlation(1+i, :) = diag(corr(X_lt{i}*U_pred,Y_lt{i}*V_pred));
    pairdisparity(1+i, :) = pair_disparity(X_lt, Y_lt, X_lt{i}, Y_lt{i}, ...
        U_lt, V_lt, U_pred, V_pred, U_lt{i}, V_lt{i});
end

% Plotting of canonical correlations
figure;
plot(dim, r_pred(1:K), '-o', 'LineWidth', 2);
for i = 1:length(dim)
    text(dim(i), r_pred(i), num2str(r_pred(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
xlabel('dimension');
ylabel('correlation');
ylim([0., 1])
hold on
for g = 1:G
    plot(dim, diag(corr(X_lt{g}*U_pred,Y_lt{g}*V_pred)), 'LineWidth', 2)
end
legend('overall', 'g1', 'g2', 'g3', 'g4', 'g5');
title('Classical CCA')
% saveas(gcf, ['Experiment_' num2str(id) '_Classical_CCA.png']);
hold off

%% SF-CCA

disp('SF-CCA')
disp(['learning rate: ', num2str(lr_b)])
disp(['learning rate control: ', lr_control_b])
disp(['maximum iteration: ', num2str(max_iter_s)])

t2 = tic;
[A1, B1, ~] = single_cca(X_lt, Y_lt, X, Y, K, ...
    'lr', lr_b, 'lr_control', lr_control_b, 'max_iter', max_iter_s);
time(2) = toc(t2);

correlation(2+G, :) = diag(corr(X*A1,Y*B1));
pairdisparity(2+G, :) = total_disparity(X_lt, Y_lt, U_lt, V_lt, A1, B1);
for i = 1:G
    disparities(G+i, :) = disparity(X_lt{i}, Y_lt{i}, A1, B1, U_lt{i}, V_lt{i});
    correlation(2+G+i, :) = diag(corr(X_lt{i}*A1,Y_lt{i}*B1));
    pairdisparity(2+G+i, :) = pair_disparity(X_lt, Y_lt, X_lt{i}, Y_lt{i}, ...
        U_lt, V_lt, A1, B1, U_lt{i}, V_lt{i});
end

% Plotting of canonical correlations
figure;
plot(dim, diag(corr(X*A1,Y*B1)), '-o', 'LineWidth', 2)
temp = diag(corr(X*A1,Y*B1));
for i = 1:length(dim)
    text(dim(i), temp(i), num2str(temp(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
xlabel('dimension');
ylabel('correlation');
ylim([0., 1])
hold on
for g = 1:G
    plot(dim, diag(corr(X_lt{g}*A1,Y_lt{g}*B1)), 'LineWidth', 2)
end
legend('overall', 'g1', 'g2', 'g3', 'g4', 'g5');
title('SF-CCA')
% saveas(gcf, ['Experiment_' num2str(id) '_SF_CCA.png']);
hold off


%% MF-CCA

disp('MF-CCA')
disp(['alpha: ', num2str(alpha)])
disp(['learning rate: ', num2str(lr_f)])
disp(['learning rate control: ', lr_control_f])
disp(['maximum iteration: ', num2str(max_iter_m)])

t3 = tic;
[A2, B2, ~] = multi_cca(X_lt, Y_lt, X, Y, K, ...
    'lr', lr_f, 'lr_control', lr_control_f, 'alpha', alpha, 'max_iter', max_iter_m);
time(3) = toc(t3);

correlation(3+2*G, :) = diag(corr(X*A2,Y*B2));
pairdisparity(3+2*G, :) = total_disparity(X_lt, Y_lt, U_lt, V_lt, A2, B2);
for i = 1:G
    disparities(2*G+i, :) = disparity(X_lt{i}, Y_lt{i}, A2, B2, U_lt{i}, V_lt{i});
    correlation(3+2*G+i, :) = diag(corr(X_lt{i}*A2,Y_lt{i}*B2));
    pairdisparity(3+2*G+i, :) = pair_disparity(X_lt, Y_lt, X_lt{i}, Y_lt{i}, ...
        U_lt, V_lt, A2, B2, U_lt{i}, V_lt{i});
end

% Plotting of canonical correlations
figure;
plot(dim, diag(corr(X*A2,Y*B2)), '-o', 'LineWidth', 2)
temp = diag(corr(X*A2,Y*B2));
for i = 1:length(dim)
    text(dim(i), temp(i), num2str(temp(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end
xlabel('dimension');
ylabel('correlation');
ylim([0., 1])
hold on
for g = 1:G
    plot(dim, diag(corr(X_lt{g}*A2,Y_lt{g}*B2)), 'LineWidth', 2)
end
legend('overall', 'g1', 'g2', 'g3', 'g4', 'g5');
title('MF-CCA')
% saveas(gcf, ['Experiment_' num2str(id) '_MF_CCA.png']);
hold off

%% Plotting of disparity error
figure;
plot(dim, abs(total_disparity(X_lt, Y_lt, U_lt, V_lt, A2, B2)), 'LineWidth', 2)
xlabel('dimension');
ylabel('total disparity error');
hold on
plot(dim, abs(total_disparity(X_lt, Y_lt, U_lt, V_lt, U_pred, V_pred)), 'LineWidth', 2)
plot(dim, abs(total_disparity(X_lt, Y_lt, U_lt, V_lt, A1, B1)), 'LineWidth', 2)
legend('MF-CCA', 'Classical CCA', 'SF-CCA');
title('Disparity Error of CCA')
% saveas(gcf, ['Experiment_' num2str(id) '_Disparity_CCA.png']);
hold off
