clear; clc; close all;
%
% SCX dataset prediction results
%
%-------------------- 1. Data Preprocessing --------------------

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

params.hidden_size = 20;           % Number of hidden units is 20
params.output_size = 1;            % Output dimension is 1
params.tau = 2.0;                  % Time constant tau controls dynamic response
input_dim = input_window;          % Input dimension equals sliding window length

% Initialize weight matrices and bias terms
params.W_in = randn(params.hidden_size, 1) * 0.1;          % Input weights (shared across time steps)
params.W_rec = randn(params.hidden_size, params.hidden_size) * 0.1; % Hidden layer recurrent connection weights
params.W_out = randn(params.output_size, params.hidden_size) * 0.1; % Output layer weights
params.b = zeros(params.hidden_size, 1);                    % Bias initialized to 0

%-------------------- 2. IALA Parameter Settings --------------------
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

momentum = 0.9;          % Momentum factor
patience = 10;           % Early stopping patience epochs

% Validation set split (20% of training set used for validation)
val_split = 0.2;
val_idx = randperm(size(X_train, 1), floor(val_split * size(X_train, 1)));
train_idx = setdiff(1:size(X_train, 1), val_idx);
X_tr = X_train(train_idx, :);
Y_tr = Y_train(train_idx);
X_val = X_train(val_idx, :);
Y_val = Y_train(val_idx);

% Initialize momentum variables
v_W_in = zeros(size(params.W_in));
v_W_rec = zeros(size(params.W_rec));
v_W_out = zeros(size(params.W_out));
v_b = zeros(size(params.b));

best_val_loss = inf;         % Best validation loss initialized to infinity
no_improve_count = 0;        % Counter for consecutive epochs without improvement
loss_history = zeros(num_epochs, 1);  % Store validation loss for each epoch
if TYPE==1
    for epoch = 1:num_epochs
        idx = randperm(size(X_tr, 1));  % Shuffle training samples
        for i = 1:batch_size:size(X_tr, 1)
            batch_idx = idx(i:min(i+batch_size-1, end));       % Get indices for each mini-batch
            X_batch = X_tr(batch_idx, :);                      % Current batch input
            Y_batch = Y_tr(batch_idx);                        % Current batch output

            % Initialize gradients
            grad_W_in = zeros(size(params.W_in));
            grad_W_rec = zeros(size(params.W_rec));
            grad_W_out = zeros(size(params.W_out));
            grad_b = zeros(size(params.b));

            % Accumulate gradients for each sample in the batch
            for j = 1:length(batch_idx)
                x_seq = X_batch(j, :);
                y_true = Y_batch(j);

                % Forward propagation (save intermediate states)
                [outputs, h, net] = simpleLNN_train(x_seq, params);
                y_pred = outputs(end);

                % Backpropagation (BPTT)
                grads = bptt(x_seq, y_true, params, outputs, h, net);

                % Accumulate gradients
                grad_W_in = grad_W_in + grads.W_in;
                grad_W_rec = grad_W_rec + grads.W_rec;
                grad_W_out = grad_W_out + grads.W_out;
                grad_b = grad_b + grads.b;
            end

            % Average gradients
            grad_W_in = grad_W_in / length(batch_idx);
            grad_W_rec = grad_W_rec / length(batch_idx);
            grad_W_out = grad_W_out / length(batch_idx);
            grad_b = grad_b / length(batch_idx);

            % Momentum update
            v_W_in = momentum * v_W_in - learning_rate * grad_W_in;
            v_W_rec = momentum * v_W_rec - learning_rate * grad_W_rec;
            v_W_out = momentum * v_W_out - learning_rate * grad_W_out;
            v_b = momentum * v_b - learning_rate * grad_b;

           % Update weights
            params.W_in = params.W_in + v_W_in;
            params.W_rec = params.W_rec + v_W_rec;
            params.W_out = params.W_out + v_W_out;
            params.b = params.b + v_b;
        end

        % Calculate validation loss
        val_loss = 0;
        for i = 1:size(X_val, 1)
            pred = simpleLNN(X_val(i,:), params);
            val_loss = val_loss + (pred(end) - Y_val(i))^2;
        end
        val_loss = val_loss / size(X_val, 1);
        loss_history(epoch) = val_loss;
        fprintf("Epoch %d | Val Loss: %.6f\n", epoch, val_loss); % Print validation loss

        % Early stopping check
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
    Y_pred_norm(i) = out(end);  
end

% Denormalization
Y_pred = Y_pred_norm * y_sigma + y_mu;
Y_real = Y_test * y_sigma + y_mu;

% Calculate evaluation metrics
errors = Y_pred - Y_real;  % Prediction errors
abs_errors = abs(errors);  % Absolute errors

% 1. Root Mean Square Error (RMSE)
rmse = sqrt(mean(errors.^2));

% 2. Mean Absolute Error (MAE)
mae = mean(abs_errors);

% 3. Coefficient of Determination (R²)
SS_res = sum(errors.^2);  % Residual sum of squares
SS_tot = sum((Y_real - mean(Y_real)).^2);  % Total sum of squares
R2 = 1 - (SS_res / SS_tot);

% 4. Δt95 metric (95% error range)
sorted_errors = sort(abs_errors);  % Sort absolute errors
n = length(sorted_errors);
index_95 = ceil(0.95 * n);  % 95% position
delta_t95 = sorted_errors(index_95);  % Δt95 value

% Output all evaluation metrics
fprintf('Test Results Evaluation Metrics:\n');
fprintf('RMSE = %.3f\n', rmse);
fprintf('MAE = %.3f\n', mae);
fprintf('R² = %.3f\n', R2);
fprintf('Δt95 = %.3f\n', delta_t95);


function [outputs, h, net] = simpleLNN_train(x_seq, params)
T = length(x_seq);                             % Input sequence length
h = zeros(params.hidden_size, T+1);            % Hidden states (including h0)
net = zeros(params.hidden_size, T);           % Store net values
outputs = zeros(T, params.output_size);         % Store outputs at each step

for t = 1:T
% LNN core state update
    net(:,t) = params.W_in * x_seq(t) + params.W_rec * h(:,t) + params.b;
    h(:,t+1) = h(:,t) + (-h(:,t) + tanh(net(:,t))) / params.tau;
    outputs(t,:) = (params.W_out * h(:,t+1))';
end
end

function outputs = simpleLNN(x_seq, params)
T = length(x_seq);                             % Input sequence length
h = zeros(params.hidden_size, 1);              % Hidden state initialized to 0
outputs = zeros(T, 1);                          % Store outputs

for t = 1:T
    % State update
    net = params.W_in * x_seq(t) + params.W_rec * h + params.b;
    h = h + (-h + tanh(net)) / params.tau;
    outputs(t) = params.W_out * h;
end
end

% BPTT backpropagation function
function grads = bptt(x_seq, y_true, params, outputs, h, net)
T = length(x_seq);  % Sequence length
grads = struct(...
    'W_in', zeros(size(params.W_in)), ...
    'W_rec', zeros(size(params.W_rec)), ...
    'W_out', zeros(size(params.W_out)), ...
    'b', zeros(size(params.b)));

% Output layer gradient (only care about the last time step)
dL_do = 2 * (outputs(end) - y_true);
grads.W_out = dL_do * h(:, end)';  % h(:,end) corresponds to h_{T+1}

% Initialize backpropagation variables
dh_next = params.W_out' * dL_do;  % Gradient backpropagated from output layer

% Backpropagate along time steps
for t = T:-1:1
    % Calculate gradient at current time step
    dnet = (1/params.tau) * (1 - tanh(net(:,t)).^2) .* dh_next;

    % Accumulate parameter gradients
    grads.W_in = grads.W_in + dnet * x_seq(t);
    grads.W_rec = grads.W_rec + dnet * h(:,t)';
    grads.b = grads.b + dnet;

    % Calculate hidden state gradient for previous time step
    dh_prev = (1 - 1/params.tau) * dh_next + params.W_rec' * dnet;

    % Update gradient for next time step
    dh_next = dh_prev;
end
end

%-------------------- 4. Subfunction Definitions --------------------
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
    fprintf('Validation Loss: %.6f\n', val_loss); % Print validation loss

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