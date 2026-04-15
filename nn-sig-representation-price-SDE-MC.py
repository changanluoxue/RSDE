import torch
import torch.nn as nn
import numpy as np
import time
import numba as nb
import matplotlib.pyplot as plt

start_time = time.perf_counter()

@nb.njit(parallel=True, fastmath=True)
def compute_sig_SABR_option_prices(lw_mw, paths_dW_mw, paths_dB, rho, beta, space_grid, K, J):
    '''
    lw_mw: np.array((J+1, ))
    paths_dW_mw: np.array((J, ))
    paths_dB: np.array((M_B, J))
    return:对应单条W的欧式期权价格np.array((L+1, ))
    '''
    M_B = paths_dB.shape[0]; L = space_grid.shape[0]-1
    sqrt_1_rho2 = np.sqrt(1-rho**2)
    payoffs = np.zeros((M_B, L+1))
    '模拟M_B条标的资产价格路径'
    for mb in nb.prange(M_B):
        '仅保留当前时刻的标的资产价格'
        current_X = np.copy(space_grid)
        for j in range(J):
            v = lw_mw[j]; dW = paths_dW_mw[j]; dB = paths_dB[mb, j]
            shock = v*(rho*dW + sqrt_1_rho2*dB)
            for i in range(L+1):
                current_X[i] = current_X[i] + (current_X[i]**beta)*shock
        '计算对应M_B条路径的到期日欧式看跌期权价格'
        for i in range(L+1):
            p = K-current_X[i]
            if p > 0:
                payoffs[mb, i] = p
    '计算对应单条W路径的到期日欧式看跌期权价格'
    sum_payoffs = np.zeros(L+1)
    for mb in range(M_B):
        for i in range(L+1):
            sum_payoffs[i] += payoffs[mb, i]
    return sum_payoffs/M_B
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
        self.dense = fnn(input_size=features, hidden_sizes=(32, 32, 32), output_size=1, activation=nn.ReLU())
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.contiguous().view(-1, features) # x: (batch_size, seq_len, features) -> permute to (batch_size * seq_len, features)
        x = self.dense(x)
        x = x.view(batch_size, seq_len, -1)
        return x
class signn_l(nn.Module):
    def __init__(self, features=64):
        super(signn_l, self).__init__()
        self.dense = fnn(input_size=features, hidden_sizes=(32, 32, 32), output_size=features-1, activation=nn.ReLU())
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        x = x.contiguous().view(-1, features) # x: (batch_size, seq_len, features) -> permute to (batch_size * seq_len, features)
        x = self.dense(x)
        x = x.view(batch_size, seq_len, -1)
        return x

'通用常数系数'
name = "rHeston" # "rBergomi", "rHeston"
method = "linear" # "linear", "nonlinear"
T = 1; J = 251; M_B = 10000; M_W = 10000
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L
rho = -0.4; beta = 1
K = 110
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
order_feature = {1:4, 2:8, 3:16, 4:32, 5:64, 6:128, 7:256, 8:512, 9:1024}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy') # np.array(M_W, J)
paths_dB = np.load(f'data/{name}/simulated_data/paths_dB.npy') # np.array(M_B, J)

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
        lw = nn_model(torch.from_numpy(sig_hatW_t).float()).detach().squeeze().numpy()  # (M_W, J+1)
    if method == 'linear':
        nn_model = signn_l(features=order_feature[n])
        nn_model.load_state_dict(torch.load(f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth', map_location=device))
        nn_model.eval()
        lw = nn_model(torch.from_numpy(sig_hatW_t).float()).detach().numpy()  # (M_W, J+1, features)
        lw = np.sum(lw*sig_hatW, -1)  # (M_W, J+1)
    '模拟 SABR 并定价'
    option_prices_nn_sig_SDE_MC = np.zeros((M_W, L+1))
    for m_w in range(M_W):
        option_prices_nn_sig_SDE_MC[m_w, :] = compute_sig_SABR_option_prices(lw[m_w, :], paths_dW[m_w, :], paths_dB, rho, beta, space_grid, K, J)
        print(f'[{n}][{m_w + 1}/{M_W}]...')
    np.save(f'data/{name}/results/option_prices/option_prices_nn_sig_SDE_MC_{method}_order{n}.npy', option_prices_nn_sig_SDE_MC)  # 形状为(M_W,L+1)

option_prices_nn_sig_SDE_MC = {}
for n in orders:
    file_path = f'data/{name}/results/option_prices/option_prices_nn_sig_SDE_MC_{method}_order{n}.npy'
    option_prices_nn_sig_SDE_MC[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_SDE_MC = np.load(f'data/{name}/results/option_prices/option_prices_SDE_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(space_grid, np.mean(option_prices_SDE_MC,0), label='MC', linestyle='-', linewidth=1)
for n in orders:
    plt.plot(space_grid, np.mean(option_prices_nn_sig_SDE_MC[f'order{n}'],0), label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'data/{name}/results/plots/option_prices_SDE_MC_{method}_1.eps')
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")

