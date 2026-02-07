import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt


start_time = time.perf_counter()

def fnn(input_size=602, hidden_sizes=(16, 16, 16), output_size=1, activation=nn.ReLU()):
    layers = []
    in_size = input_size
    for size in hidden_sizes:
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    model = nn.Sequential(*layers)
    return model
class signn_lw(nn.Module):
    def __init__(self, features=64):
        super(signn_lw, self).__init__()
        self.dense = fnn(input_size=features, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.contiguous().view(-1, features) # x: (batch_size, seq_len, features) -> permute to (batch_size * seq_len, features)
        x = self.dense(x)
        x = x.view(batch_size, seq_len, -1)
        return x
class signn_l(nn.Module):
    def __init__(self, features=64):
        super(signn_l, self).__init__()
        self.dense = fnn(input_size=features, hidden_sizes=(32, 32, 32, 32, 32), output_size=features-1, activation=nn.ReLU())
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.contiguous().view(-1, features) # x: (batch_size, seq_len, features) -> permute to (batch_size * seq_len, features)
        x = self.dense(x)
        x = x.view(batch_size, seq_len, -1)
        return x
def sig_representation_SABR(paths_dW, rho, beta, x_0, J, M):
    '''
    paths_dW:布朗运动增量:np.array((M, J))
    rho:SABR相关系数
    beta:SABR弹性指数
    x_0:初值
    T:时间长度
    J:时间网格数
    M:路径数
    return: paths_SABR:np.array((M, J+1))
    '''
    paths_SABR = np.zeros((M, J+1))
    paths_SABR[:, 0] = x_0

    for j in range(1, J+1):
        dB = np.sqrt(dt)*np.random.standard_normal(M)
        paths_SABR[:, j] = (paths_SABR[:, j-1]
                            + np.sqrt(1-rho**2)*paths_SABR[:, j-1]**beta*lw[:, j-1]*dB
                            + fpw(paths_SABR[:, j-1], j-1)*lw[:, j-1]*paths_dW[:, j-1])
    return paths_SABR
def simulate_I(paths_v, paths_dW, J, M):
    '''
    return: paths_I:np.array((M, J+1))
    '''
    paths_I = np.zeros((M, J+1))

    for j in range(1, J+1):
        paths_I[:, j] = paths_I[:, j-1] + paths_v[:, j-1]*paths_dW[:, j-1]
    return paths_I
def eur_put_option(paths_price, K):
    option_prices = np.maximum(K-paths_price[:,-1,:], 0)
    return option_prices

'通用常数系数'
name = "rHeston" # "rBergomi", "rHeston"
method = "nonlinear" # "linear", "nonlinear"
T = 1; J = 251; M = 10000; dt = T/J
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
order_feature = {1:4, 2:8, 3:16, 4:32, 5:64, 6:128, 7:256, 8:512, 9:1024}
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L; X_0 = np.linspace(x_low, x_high, L+1)
rho = -0.4; beta = 0.6
K = 110
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')

def fpw(x, j, rho=rho, beta=beta):
    '''
    return: np.array(M,)
    '''
    return rho*x**beta+rho**2*beta*x**(2*beta-1)*pw[:,j]

'计算签名重构系数'
for n in orders:
    '导入路径签名'
    sig_hatW = np.load(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy')
    timee = np.broadcast_to(np.expand_dims(time_grid, axis=(0, -1)), (sig_hatW.shape[0], sig_hatW.shape[1], 1))
    sig_hatW_t = np.concatenate((sig_hatW, timee), axis=-1)

    '模型加载及输出'
    if method == 'nonlinear':
        nn_model = signn_lw(features=order_feature[n])
        nn_model.load_state_dict(torch.load(f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth', map_location=device))
        nn_model.eval()
        lw = nn_model(torch.from_numpy(sig_hatW_t).float()).detach().squeeze().numpy()  # (M, J+1)
    if method == 'linear':
        nn_model = signn_l(features=order_feature[n])
        nn_model.load_state_dict(torch.load(f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth', map_location=device))
        nn_model.eval()
        lw = nn_model(torch.from_numpy(sig_hatW_t).float()).detach().numpy()  # (M, J+1, features)
        lw = np.sum(lw*sig_hatW, -1)  # (M, J+1)

    pw = simulate_I(lw, paths_dW, J, M)
    '计算SABR签名重构路径'
    sig_paths_SABR = np.zeros((M, J+1, L+1))
    for i, x_0 in enumerate(X_0):
        sig_paths_SABR[:, :, i] = sig_representation_SABR(paths_dW, rho, beta, x_0, J, M)
        print(f'进度[{n}/{orders}][{i}/{L}]...')

    '欧式期权定价'
    option_prices_nn_sig_MC = np.mean(eur_put_option(sig_paths_SABR, K), 0)
    np.save(f'data/{name}/results/option_prices/option_prices_nn_sig_MC_{method}_order{n}.npy', option_prices_nn_sig_MC)

option_prices_nn_sig_MC = {}
for n in orders:
    file_path = f'data/{name}/results/option_prices/option_prices_nn_sig_MC_{method}_order{n}.npy'
    option_prices_nn_sig_MC[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_MC = np.load(f'data/{name}/results/option_prices/option_prices_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(X_0, option_prices_MC, label='MC', linestyle='-', linewidth=1)
for n in orders:
    plt.plot(X_0, option_prices_nn_sig_MC[f'order{n}'], label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'data/{name}/results/plots/option_prices_nn_sig_MC_{method}.eps')
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")

