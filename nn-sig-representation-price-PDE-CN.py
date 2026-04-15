import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.perf_counter()

'Crank-Nicolson'
def solve_spde_one_path(lw_mw, dW_mw, u_terminal, L_p1, J, dt, dx, lower_bd, upper_bd, Coeff_A_space, Coeff_B_space, Coeff_C_space):
    '''
    求解如下单条路径的倒向偏微分方程：
    -du=0.5\partial_{xx}^2u(1-2\rho^2)x^{2\beta}v_t^2dt-\partial_xu\rho^2\beta x^{2\beta-1}v_t^2dt+\partial_xu\rho x^{\beta}v_tdW_t

    return:
        u: np.array((L+1,))
    '''
    u = u_terminal.copy()
    '截取内部节点'
    A_inner = Coeff_A_space[1:-1]
    B_inner = Coeff_B_space[1:-1]
    C_inner = Coeff_C_space[1:-1]
    '时间倒推循环'
    for j in range(J-1, -1, -1):
        lw_t = lw_mw[j]; dW_t = dW_mw[j]
        lw_sq = lw_t**2
        '1. 计算当前步的有效扩散系数和对流系数'
        '扩散项(对应二阶导): A_space*(lw_t)^2'
        diff_coeff = A_inner*lw_sq
        '对流项(对应一阶导): B_space*(lw_t)^2+C_space*(lw_t)^2*(dW_t/dt)'
        '注意：这里除以dt是为了适配下方统一的(*dt)操作，从而还原真实的dW_t'
        adv_coeff = B_inner*lw_sq+C_inner*lw_sq*(dW_t/dt)
        '2. 计算Crank-Nicolson差分参数'
        alpha = diff_coeff*dt/(2*dx**2)
        gamma = adv_coeff*dt/(4*dx)
        '3. 构造三对角矩阵(LHS, t_{j-1})'
        ab = np.zeros((3, L_p1-2))
        diag_upper = -alpha-gamma
        diag_main = 1+2*alpha
        diag_lower = -alpha+gamma
        ab[0, 1:] = diag_upper[:-1]
        ab[1, :] = diag_main
        ab[2, :-1] = diag_lower[1:]
        '4. 构造右端项(RHS, t_j)'
        u_prev = u[0:-2]
        u_curr = u[1:-1]
        u_next = u[2:]
        rhs_lower = alpha-gamma
        rhs_main = 1-2*alpha
        rhs_upper = alpha+gamma
        b = rhs_lower*u_prev+rhs_main*u_curr+rhs_upper*u_next
        '5. Dirichlet 边界条件移项处理'
        b[0] += (alpha[0]-gamma[0])*lower_bd
        b[-1] += (alpha[-1]+gamma[-1])*upper_bd
        '求解线性方程组'
        u_inner = solve_banded((1, 1), ab, b)
        '更新解'
        u[1:-1] = u_inner
        u[0] = lower_bd
        u[-1] = upper_bd
    return u
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
T = 1; J = 251; M_W = 10000
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
order_feature = {1:4, 2:8, 3:16, 4:32, 5:64, 6:128, 7:256, 8:512, 9:1024}
rho = -0.4; beta = 1
K = 110
dt = T/J
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动、波动率路径和MC基准'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')
paths_v = np.load(f'data/{name}/simulated_data/paths_v.npy')
option_prices_SDE_MC = np.load(f'data/{name}/results/option_prices/option_prices_SDE_MC.npy')

'预计算空间网格相关的项，避免在循环中重复计算'
u_terminal = np.maximum(K-space_grid, 0) # 终值条件
X_beta_all = space_grid**beta # x^{\beta}
X_2beta_all = space_grid**(2*beta) # x^{2\beta}
X_2beta_minus_1_all = space_grid**(2*beta-1) # x^{2\beta-1}
'将系数拆分为A(扩散), B(对流dt项), C(对流dW项)'
'A: 0.5*(1-2*rho^2)*x^(2beta)'
Coeff_A_space = 0.5*(1-2*rho**2)*X_2beta_all
'B: -rho^2*beta*x^(2beta-1)'
Coeff_B_space = -(rho**2)*beta*X_2beta_minus_1_all
'C: rho*x^beta'
Coeff_C_space = rho*X_beta_all

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
    '求解PDE'
    option_prices_nn_sig_PDE_CN = np.zeros((M_W, L+1))
    for m_w in range(M_W):
        option_prices_nn_sig_PDE_CN[m_w, :] = solve_spde_one_path(lw[m_w, :], paths_dW[m_w, :], u_terminal, L + 1, J, dt, dx,
                                                                  option_prices_SDE_MC[m_w, 0], option_prices_SDE_MC[m_w, -1],
                                                                  Coeff_A_space, Coeff_B_space, Coeff_C_space)
        print(f'[{n}][{m_w+1}/{M_W}]...')
    np.save(f'data/{name}/results/option_prices/option_prices_nn_sig_PDE_CN_{method}_order{n}.npy', option_prices_nn_sig_PDE_CN)  # 形状为(M_W,L+1)

option_prices_nn_sig_PDE_CN = {}
for n in orders:
    file_path = f'data/{name}/results/option_prices/option_prices_nn_sig_PDE_CN_{method}_order{n}.npy'
    option_prices_nn_sig_PDE_CN[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_SDE_MC = np.load(f'data/{name}/results/option_prices/option_prices_SDE_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(space_grid, np.mean(option_prices_SDE_MC,0), label='MC', linestyle='-', linewidth=1)
for n in orders:
    plt.plot(space_grid, np.mean(option_prices_nn_sig_PDE_CN[f'order{n}'],0), label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'data/{name}/results/plots/option_prices_PDE_CN_{method}_1.eps')
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")


