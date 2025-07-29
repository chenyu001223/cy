clear; clc; close all;

%RP 预测结果
%-------------------- 1. 数据预处理 --------------------
load sdata.mat
input_window = 1;
predict_horizon = 1;
TYPE=0;
X = []; Y = [];
for i = 1:(length(data) - input_window - predict_horizon + 1)
    X = [X; data(i:i+input_window-1)'];
    Y = [Y; data(i+input_window+predict_horizon-1)];
end
[X_norm, x_mu, x_sigma] = zscore(X);
[Y_norm, y_mu, y_sigma] = zscore(Y);

n_train = floor(0.8 * size(X, 1));
X_train = X_norm(1:n_train, :);
Y_train = Y_norm(1:n_train);
X_test = X_norm(n_train+1:end, :);
Y_test = Y_norm(n_train+1:end);

%-------------------- 显示数据集信息 --------------------
fprintf('=== 数据集划分信息 ===\n');
fprintf('原始数据总量: %d 个样本\n', size(X, 1));
fprintf('训练集数量: %d 个样本\n', size(X_train, 1));
fprintf('测试集数量: %d 个样本\n', size(X_test, 1));
fprintf('训练集占比: %.1f%%\n', size(X_train, 1)/size(X, 1)*100);
fprintf('测试集占比: %.1f%%\n', size(X_test, 1)/size(X, 1)*100);
fprintf('\n=== 数据集形状信息 ===\n');
fprintf('X_train 形状: [%d, %d] (样本数 × 特征数)\n', size(X_train, 1), size(X_train, 2));
fprintf('Y_train 形状: [%d, %d]\n', size(Y_train, 1), size(Y_train, 2));
fprintf('X_test 形状: [%d, %d] (样本数 × 特征数)\n', size(X_test, 1), size(X_test, 2));
fprintf('Y_test 形状: [%d, %d]\n', size(Y_test, 1), size(Y_test, 2));
fprintf('输入窗口长度: %d\n', input_window);
fprintf('预测时间跨度: %d\n', predict_horizon);
fprintf('================================\n\n');

params.hidden_size = 20;           % 隐层单元数为20
params.output_size = 1;            % 输出维度为1
params.tau = 2.0;                  % 时间常数tau控制动态响应
input_dim = input_window;          % 输入维度等于滑动窗口长度

% 初始化权重矩阵和偏置项
params.W_in = randn(params.hidden_size, 1) * 0.1;           % 输入权重 (每个时间步共享)
params.W_rec = randn(params.hidden_size, params.hidden_size) * 0.1; % 隐层循环连接权重
params.W_out = randn(params.output_size, params.hidden_size) * 0.1; % 输出层权重
params.b = zeros(params.hidden_size, 1);                    % 偏置初始化为0

%-------------------- 2. IALA 参数设置 --------------------
SearchAgents_no = 100;
Max_iter = 100;
dim = 3;
lb = [50, 8, 0.001];
ub = [300, 64, 0.05];
X0 = zeros(SearchAgents_no, dim);
for i = 1:SearchAgents_no
    X0(i,1) = randi([lb(1), ub(1)]);
    X0(i,2) = randi([lb(2), ub(2)]);
    X0(i,3) = lb(3) + rand() * (ub(3)-lb(3));
end

fobj = @(x) LNN_RMSE_Eval(round(x(1)), round(x(2)), x(3), X_train, Y_train);
if TYPE==1
    [~, best_params, ~] = IALA(X0, SearchAgents_no, Max_iter, lb, ub, dim, fobj);

    num_epochs = round(best_params(1));
    batch_size = round(best_params(2));
    learning_rate = best_params(3);

    % ✅ 添加：显示IALA优化结果
    fprintf('\n=== IALA优化完成 ===\n');
    fprintf('最优超参数组合:\n');
    fprintf('  训练轮数: %d\n', num_epochs);
    fprintf('  批次大小: %d\n', batch_size);
    fprintf('  学习率: %.6f\n', learning_rate);
    fprintf('==================\n\n');

else
    num_epochs = 300;
    batch_size = 16;
    learning_rate =0.001578;
end

momentum = 0.9;           % 动量因子
patience = 10;            % 早停容忍轮数

% 验证集划分（20%的训练集用于验证）
val_split = 0.2;
val_idx = randperm(size(X_train, 1), floor(val_split * size(X_train, 1)));
train_idx = setdiff(1:size(X_train, 1), val_idx);
X_tr = X_train(train_idx, :);
Y_tr = Y_train(train_idx);
X_val = X_train(val_idx, :);
Y_val = Y_train(val_idx);

%-------------------- 显示训练/验证集划分信息 --------------------
fprintf('=== 训练/验证集划分信息 ===\n');
fprintf('实际训练集数量: %d 个样本 (占原训练集的 %.1f%%)\n', size(X_tr, 1), size(X_tr, 1)/size(X_train, 1)*100);
fprintf('验证集数量: %d 个样本 (占原训练集的 %.1f%%)\n', size(X_val, 1), size(X_val, 1)/size(X_train, 1)*100);
fprintf('X_tr 形状: [%d, %d]\n', size(X_tr, 1), size(X_tr, 2));
fprintf('Y_tr 形状: [%d, %d]\n', size(Y_tr, 1), size(Y_tr, 2));
fprintf('X_val 形状: [%d, %d]\n', size(X_val, 1), size(X_val, 2));
fprintf('Y_val 形状: [%d, %d]\n', size(Y_val, 1), size(Y_val, 2));
fprintf('================================\n\n');

% 初始化动量变量
v_W_in = zeros(size(params.W_in));
v_W_rec = zeros(size(params.W_rec));
v_W_out = zeros(size(params.W_out));
v_b = zeros(size(params.b));

best_val_loss = inf;         % 最优验证损失初始化为无穷
no_improve_count = 0;        % 连续未提升计数
loss_history = zeros(num_epochs, 1);  % 存储每轮验证损失
if TYPE==1
    for epoch = 1:num_epochs
        idx = randperm(size(X_tr, 1));  % 打乱训练样本
        for i = 1:batch_size:size(X_tr, 1)
            batch_idx = idx(i:min(i+batch_size-1, end));       % 获取每个小批次索引
            X_batch = X_tr(batch_idx, :);                      % 当前批次输入
            Y_batch = Y_tr(batch_idx);                         % 当前批次输出

            % 初始化梯度
            grad_W_in = zeros(size(params.W_in));
            grad_W_rec = zeros(size(params.W_rec));
            grad_W_out = zeros(size(params.W_out));
            grad_b = zeros(size(params.b));

            % 批处理中每个样本的梯度累加
            for j = 1:length(batch_idx)
                x_seq = X_batch(j, :);
                y_true = Y_batch(j);

                % 前向传播 (保存中间状态)
                [outputs, h, net] = simpleLNN_train(x_seq, params);
                y_pred = outputs(end);

                % 反向传播 (BPTT)
                grads = bptt(x_seq, y_true, params, outputs, h, net);

                % 累加梯度
                grad_W_in = grad_W_in + grads.W_in;
                grad_W_rec = grad_W_rec + grads.W_rec;
                grad_W_out = grad_W_out + grads.W_out;
                grad_b = grad_b + grads.b;
            end

            % 平均梯度
            grad_W_in = grad_W_in / length(batch_idx);
            grad_W_rec = grad_W_rec / length(batch_idx);
            grad_W_out = grad_W_out / length(batch_idx);
            grad_b = grad_b / length(batch_idx);

            % 动量更新
            v_W_in = momentum * v_W_in - learning_rate * grad_W_in;
            v_W_rec = momentum * v_W_rec - learning_rate * grad_W_rec;
            v_W_out = momentum * v_W_out - learning_rate * grad_W_out;
            v_b = momentum * v_b - learning_rate * grad_b;

            % 更新权重
            params.W_in = params.W_in + v_W_in;
            params.W_rec = params.W_rec + v_W_rec;
            params.W_out = params.W_out + v_W_out;
            params.b = params.b + v_b;
        end

        % 计算验证损失
        val_loss = 0;
        for i = 1:size(X_val, 1)
            pred = simpleLNN(X_val(i,:), params);
            val_loss = val_loss + (pred(end) - Y_val(i))^2;
        end
        val_loss = val_loss / size(X_val, 1);
        loss_history(epoch) = val_loss;
        fprintf("Epoch %d | Val Loss: %.6f\n", epoch, val_loss); % 打印验证损失

        % 早停检查
        if val_loss < best_val_loss
            best_val_loss = val_loss;
            best_params = params;
            no_improve_count = 0;
        else
            no_improve_count = no_improve_count + 1;
            if no_improve_count >= patience
                fprintf("Early stopping at epoch %d.\n", epoch);
                break;
            end
        end
    end
    params = best_params;

    % ✅ 添加：保存最佳参数到文件
    fprintf('\n=== 保存最佳模型 ===\n');
    fprintf('最佳验证损失: %.6f\n', best_val_loss);
    fprintf('最佳验证RMSE: %.6f\n', sqrt(best_val_loss));
    
    % 保存完整的训练结果
    save('best_model.mat', 'params', 'num_epochs', 'batch_size', 'learning_rate', ...
         'best_val_loss', 'x_mu', 'x_sigma', 'y_mu', 'y_sigma', 'loss_history');
    
    fprintf('最佳模型已保存到 best_model.mat\n');
    fprintf('包含内容:\n');
    fprintf('  - params: 模型权重和偏置\n');
    fprintf('  - 超参数: num_epochs, batch_size, learning_rate\n');
    fprintf('  - 预处理参数: x_mu, x_sigma, y_mu, y_sigma\n');
    fprintf('  - 训练历史: best_val_loss, loss_history\n');
    fprintf('===================\n\n');
else
    load snet.mat
end
Y_pred_norm = zeros(size(Y_test));
for i = 1:size(X_test, 1)
    out = simpleLNN(X_test(i, :), params);
    Y_pred_norm(i) = out(end);  % 取最后一个时刻的预测输出
end

% 反归一化处理
Y_pred = Y_pred_norm * y_sigma + y_mu;
Y_real = Y_test * y_sigma + y_mu;

% 计算评价指标
errors = Y_pred - Y_real;  % 预测误差
abs_errors = abs(errors);  % 绝对误差

% 1. 均方根误差 (RMSE)
rmse = sqrt(mean(errors.^2));

% 2. 平均绝对误差 (MAE)
mae = mean(abs_errors);

% 3. 决定系数 (R²)
SS_res = sum(errors.^2);  % 残差平方和
SS_tot = sum((Y_real - mean(Y_real)).^2);  % 总平方和
R2 = 1 - (SS_res / SS_tot);

% 4. Δt95指标 (95%误差范围)
sorted_errors = sort(abs_errors);  % 排序绝对误差
n = length(sorted_errors);
index_95 = ceil(0.95 * n);  % 95%位置
delta_t95 = sorted_errors(index_95);  % Δt95值

% 输出所有评价指标
fprintf('\n=== 测试结果评价指标 ===\n');
fprintf('RMSE = %.3f\n', rmse);
fprintf('MAE = %.3f\n', mae);
fprintf('R² = %.3f\n', R2);
fprintf('Δt95 = %.3f\n', delta_t95);
fprintf('=======================\n');

%% 在评价指标输出后添加以下代码

%-------------------- 散点图可视化 --------------------
fprintf('\n=== 生成预测结果散点图 ===\n');

% 创建图形窗口
figure('Position', [100, 100, 800, 700]);

% 绘制散点图 - 使用蓝色系
scatter(Y_real, Y_pred, 30, [0.3, 0.6, 0.9], 'filled', 'MarkerFaceAlpha', 0.8, 'MarkerEdgeColor', [0.2, 0.4, 0.7], 'MarkerEdgeAlpha', 0.6);
hold on;

% 计算坐标轴范围
min_val = min([Y_real; Y_pred]);
max_val = max([Y_real; Y_pred]);
range_val = max_val - min_val;
plot_min = min_val - 0.05 * range_val;
plot_max = max_val + 0.05 * range_val;

% 绘制理想线 y=x - 使用更细的线条
plot([plot_min, plot_max], [plot_min, plot_max], 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.5);

% 绘制Δt95误差虚线
plot([plot_min, plot_max], [plot_min + delta_t95, plot_max + delta_t95], '--', 'Color', [0.8, 0.5, 0.5], 'LineWidth', 1.2);
plot([plot_min, plot_max], [plot_min - delta_t95, plot_max - delta_t95], '--', 'Color', [0.8, 0.5, 0.5], 'LineWidth', 1.2);

% 设置坐标轴
xlim([plot_min, plot_max]);
ylim([plot_min, plot_max]);
xlabel('Observed RT (min)', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Predicted RT (min)', 'FontSize', 14, 'FontWeight', 'bold');

% 设置图形属性 - 无网格，淡色系
axis equal;
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
set(gca, 'Box', 'on');
set(gca, 'Color', 'white');  % 白色背景

% 添加主标题和副标题
title_str = sprintf('RP\nR² = %.3f     MAE = %.2f min     Δt95 = %.2f min', R2, mae, delta_t95);
title(title_str, 'FontSize', 16, 'FontWeight', 'bold', 'Color', [0.3, 0.3, 0.3]);

% 保存图形 - 高质量保存
save_dir = './results/';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = [save_dir, 'RT_scatter_', timestamp, '.png'];

% 设置高分辨率保存
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, filename, '-dpng', '-r300');
fprintf('高质量散点图已保存为: %s\n', filename);

fprintf('=========================\n');


function [outputs, h, net] = simpleLNN_train(x_seq, params)
T = length(x_seq);                             % 输入序列长度
h = zeros(params.hidden_size, T+1);            % 隐状态 (包括h0)
net = zeros(params.hidden_size, T);            % 存储net值
outputs = zeros(T, params.output_size);        % 存储每一步的输出

for t = 1:T
    % LNN核心状态更新
    net(:,t) = params.W_in * x_seq(t) + params.W_rec * h(:,t) + params.b;
    h(:,t+1) = h(:,t) + (-h(:,t) + tanh(net(:,t))) / params.tau;
    outputs(t,:) = (params.W_out * h(:,t+1))';
end
end

function outputs = simpleLNN(x_seq, params)
T = length(x_seq);                             % 输入序列长度
h = zeros(params.hidden_size, 1);              % 隐状态初始化为0
outputs = zeros(T, 1);                         % 存储输出

for t = 1:T
    % 状态更新
    net = params.W_in * x_seq(t) + params.W_rec * h + params.b;
    h = h + (-h + tanh(net)) / params.tau;
    outputs(t) = params.W_out * h;
end
end

% BPTT 反向传播函数
function grads = bptt(x_seq, y_true, params, outputs, h, net)
T = length(x_seq);  % 序列长度
grads = struct(...
    'W_in', zeros(size(params.W_in)), ...
    'W_rec', zeros(size(params.W_rec)), ...
    'W_out', zeros(size(params.W_out)), ...
    'b', zeros(size(params.b)));

% 输出层梯度 (只关心最后时间步)
dL_do = 2 * (outputs(end) - y_true);
grads.W_out = dL_do * h(:, end)';  % h(:,end)对应h_{T+1}

% 初始化反向传播变量
dh_next = params.W_out' * dL_do;  % 从输出层反向传播的梯度

% 沿时间步反向传播
for t = T:-1:1
    % 计算当前时间步的梯度
    dnet = (1/params.tau) * (1 - tanh(net(:,t)).^2) .* dh_next;

    % 参数梯度累积
    grads.W_in = grads.W_in + dnet * x_seq(t);
    grads.W_rec = grads.W_rec + dnet * h(:,t)';
    grads.b = grads.b + dnet;

    % 计算前一时间步的隐状态梯度
    dh_prev = (1 - 1/params.tau) * dh_next + params.W_rec' * dnet;

    % 更新下一时间步的梯度
    dh_next = dh_prev;
end
end

%-------------------- 4. 子函数定义 --------------------
function rmse = LNN_RMSE_Eval(num_epochs, batch_size, learning_rate, X_train, Y_train)

% ✅ 在这里添加计数器
persistent eval_count;
if isempty(eval_count)
    eval_count = 0;
end
eval_count = eval_count + 1;

fprintf('=== 第%d次超参数评估 ===\n', eval_count);
fprintf('参数: 轮数=%d, 批次=%d, 学习率=%.6f\n', num_epochs, batch_size, learning_rate);


val_split = 0.2;
val_idx = randperm(size(X_train, 1), floor(val_split * size(X_train, 1)));
train_idx = setdiff(1:size(X_train, 1), val_idx);
X_tr = X_train(train_idx, :);
Y_tr = Y_train(train_idx);
X_val = X_train(val_idx, :);
Y_val = Y_train(val_idx);

params.hidden_size = 20;
params.output_size = 1;
params.tau = 2.0;
params.W_in = randn(params.hidden_size, 1) * 0.1;
params.W_rec = randn(params.hidden_size, params.hidden_size) * 0.1;
params.W_out = randn(params.output_size, params.hidden_size) * 0.1;
params.b = zeros(params.hidden_size, 1);

momentum = 0.9;
v_W_in = zeros(size(params.W_in));
v_W_rec = zeros(size(params.W_rec));
v_W_out = zeros(size(params.W_out));
v_b = zeros(size(params.b));

best_val_loss = inf;
no_improve_count = 0;
patience = 5;

for epoch = 1:num_epochs
    idx = randperm(size(X_tr, 1));
    for i = 1:batch_size:size(X_tr, 1)
        batch_idx = idx(i:min(i+batch_size-1, end));
        X_batch = X_tr(batch_idx, :);
        Y_batch = Y_tr(batch_idx);

        grad_W_in = zeros(size(params.W_in));
        grad_W_rec = zeros(size(params.W_rec));
        grad_W_out = zeros(size(params.W_out));
        grad_b = zeros(size(params.b));

        for j = 1:length(batch_idx)
            x_seq = X_batch(j, :);
            y_true = Y_batch(j);
            [outputs, h, net] = simpleLNN_train(x_seq, params);
            grads = bptt(x_seq, y_true, params, outputs, h, net);
            grad_W_in = grad_W_in + grads.W_in;
            grad_W_rec = grad_W_rec + grads.W_rec;
            grad_W_out = grad_W_out + grads.W_out;
            grad_b = grad_b + grads.b;
        end

        grad_W_in = grad_W_in / length(batch_idx);
        grad_W_rec = grad_W_rec / length(batch_idx);
        grad_W_out = grad_W_out / length(batch_idx);
        grad_b = grad_b / length(batch_idx);

        v_W_in = momentum * v_W_in - learning_rate * grad_W_in;
        v_W_rec = momentum * v_W_rec - learning_rate * grad_W_rec;
        v_W_out = momentum * v_W_out - learning_rate * grad_W_out;
        v_b = momentum * v_b - learning_rate * grad_b;

        params.W_in = params.W_in + v_W_in;
        params.W_rec = params.W_rec + v_W_rec;
        params.W_out = params.W_out + v_W_out;
        params.b = params.b + v_b;
    end

    val_loss = 0;
    for i = 1:size(X_val, 1)
        pred = simpleLNN(X_val(i,:), params);
        val_loss = val_loss + (pred(end) - Y_val(i))^2;
    end
    val_loss = val_loss / size(X_val, 1);
    fprintf('Validation Loss: %.6f (评估%d-轮次%d)\n', val_loss, eval_count, epoch);

    if val_loss < best_val_loss
        best_val_loss = val_loss;
        no_improve_count = 0;
    else
        no_improve_count = no_improve_count + 1;
        if no_improve_count >= patience
            break;
        end
    end
end
rmse = sqrt(best_val_loss);
end