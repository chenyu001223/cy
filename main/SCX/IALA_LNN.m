clear; clc; close all;
%
%SCX 数据集预测结果
%
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
else
    num_epochs = 300;
    batch_size = 16;
    learning_rate =0.01578;
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
fprintf('测试结果评价指标:\n');
fprintf('RMSE = %.3f\n', rmse);
fprintf('MAE = %.3f\n', mae);
fprintf('R² = %.3f\n', R2);
fprintf('Δt95 = %.3f\n', delta_t95);


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
    fprintf('Validation Loss: %.6f\n', val_loss); % 打印验证损失

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