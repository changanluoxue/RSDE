import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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
def simulate_I(paths_v, paths_dW, J, M):
    '''
    return: paths_I:np.array((M, J+1))
    '''
    paths_I = np.zeros((M, J+1))

    for j in range(1, J+1):
        paths_I[:, j] = paths_I[:, j-1] + paths_v[:, j-1]*paths_dW[:, j-1]
    return paths_I
def payoff_func(K, x):
    return np.maximum(K-x, 0)

'通用常数系数'
name = "rHeston" # "rBergomi", "rHeston"
method = "nonlinear" # "linear", "nonlinear"
T = 1; J = 251; M = 10000; dt = T/J
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
order_feature = {1:4, 2:8, 3:16, 4:32, 5:64, 6:128, 7:256, 8:512, 9:1024}
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L; X_0 = np.linspace(x_low, x_high, L+1)
rho = -0.4; beta = 1
K = 110
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动和SABR路径'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')
paths_SABR = np.load(f'data/{name}/simulated_data/paths_SABR.npy')

'利用期望确定Dirichlet边值条件'
lower_boundary = np.mean(np.maximum(K-paths_SABR[:, -1, 0],0))
upper_boundary = np.mean(np.maximum(K-paths_SABR[:, -1, -1],0))

'Crank-Nicolson(显隐混合)'
def Crank_Nicolson(dt=dt, dx=dx, M=M, J=J, L=L, K=K):
        '''
        return: ndarray((M, J+1, L+1)), 期权价格矩阵
        '''
        u = np.zeros((M, J+1, L+1))
        u[:, -1, :] = payoff_func(K, space_grid)  # 终值条件
        u[:, :-1, 0] = lower_boundary  # 边值条件(Low)
        u[:, :-1, -1] = upper_boundary  # 边值条件(High)

        for j in range(J-1, -1, -1):
            main_diag = np.zeros((M, L+1)) #主对角线
            lower_diag = np.zeros((M, L+1)) #下对角线
            upper_diag = np.zeros((M, L+1)) #上对角线
            rhs = np.zeros((M, L+1)) #右端项

            for l in range(1, L):
                x = space_grid[l]
                y = lw[:, j]**2 * ((1-rho**2)*x**(2*beta) + fpw(x, j)**2)
                yy = lw[:, j+1]**2 * ((1-rho**2)*x**(2*beta) + fpw(x, j+1)**2)

                main_diag[:, l] = 4 + 2*(dt/dx**2)*y
                lower_diag[:, l] = -(dt/dx**2)*y
                upper_diag[:, l] = -(dt/dx**2)*y
                rhs[:, l] = (4-2*(dt/dx**2)*yy)*u[:, j+1, l] + (dt/dx**2)*yy*u[:, j+1, l-1] + (dt/dx**2)*yy*u[:, j+1, l+1]

            # 处理边界
            main_diag[:, 0] = 1.0
            rhs[:, 0] = u[:, j, 0]

            main_diag[:, L] = 1.0
            rhs[:, L] = u[:, j, -1]

            #构建三对角稀疏矩阵并求解线性方程组
            for m in range(M):
                diagonal = diags([main_diag[m, :], lower_diag[m, 1:], upper_diag[m, :-1]], [0, -1, 1], format='csc')
                u[m, j, :] = spsolve(diagonal, rhs[m, :])
        return u
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
    '求解PDE'
    option_prices_nn_sig_CN = np.mean(Crank_Nicolson(), 0)[0, :]
    np.save(f'data/{name}/results/option_prices/option_prices_nn_sig_CN_{method}_order{n}.npy', option_prices_nn_sig_CN)  # 形状为(L+1, )

option_prices_nn_sig_CN = {}
for n in orders:
    file_path = f'data/{name}/results/option_prices/option_prices_nn_sig_CN_{method}_order{n}.npy'
    option_prices_nn_sig_CN[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_MC = np.load(f'data/{name}/results/option_prices/option_prices_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(X_0, option_prices_MC, label='MC', linestyle='-', linewidth=1)
for n in orders:
    plt.plot(X_0, option_prices_nn_sig_CN[f'order{n}'], label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'data/{name}/results/plots/option_prices_nn_sig_CN_{method}.eps')
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")


