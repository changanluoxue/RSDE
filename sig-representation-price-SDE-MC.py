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

'常数系数'
name = "ou" # "ou", "mGBM"
T = 1; J = 251; M_B = 10000; M_W = 10000
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L
rho = -0.4; beta = 0.6
K = 110
N = 5; v_0 = 0.25

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy') # np.array(M_W, J)
paths_dB = np.load(f'data/{name}/simulated_data/paths_dB.npy') # np.array(M_B, J)

'输入系数矩阵解析式'
if name == 'ou':
    kappa = 1; theta = 0.25; eta = 1.2
    TA_l = np.array([v_0,
                     -kappa*(v_0-theta), eta,
                     kappa**2*(v_0-theta), 0, -kappa*eta, 0,
                     -kappa**3*(v_0-theta), 0, 0, 0, kappa**2*eta, 0, 0, 0,
                     kappa**4*(v_0-theta), 0, 0, 0, 0, 0, 0, 0, -kappa**3*eta, 0, 0, 0, 0, 0, 0, 0,
                     -kappa**5*(v_0-theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kappa**4*eta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    TA_p = np.array([0,
                     0, v_0,
                     0, -kappa*(v_0-theta), 0, eta,
                     0, kappa**2*(v_0-theta), 0, 0, 0, -kappa*eta, 0, 0,
                     0, -kappa**3*(v_0-theta), 0, 0, 0, 0, 0, 0, 0, kappa**2*eta, 0, 0, 0, 0, 0, 0,
                     0, kappa**4*(v_0-theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kappa**3*eta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
if name == 'mGBM':
    kappa = 1; theta = 0.25; sigma = 0.5; eta = 0
    Lambda = -(kappa+sigma**2/2)
    gamma = kappa*theta-sigma*eta/2
    zeta1 = v_0*Lambda + gamma
    zeta2 = v_0*sigma + eta
    TA_l = np.array([v_0,
                    zeta1, zeta2,
                    Lambda*zeta1, sigma*zeta1, Lambda*zeta2, sigma*zeta2,
                    Lambda**2*zeta1, Lambda*sigma*zeta1, Lambda*sigma*zeta1, sigma**2*zeta1, Lambda**2*zeta2, Lambda*sigma*zeta2, Lambda*sigma*zeta2, sigma**2*zeta2,
                    Lambda**3*zeta1, Lambda**2*sigma*zeta1, Lambda**2*sigma*zeta1, Lambda*sigma**2*zeta1, Lambda**2*sigma*zeta1, Lambda*sigma**2*zeta1, Lambda*sigma**2*zeta1, sigma**3*zeta1, Lambda**3*zeta2, Lambda**2*sigma*zeta2, Lambda**2*sigma*zeta2, Lambda*sigma**2*zeta2, Lambda**2*sigma*zeta2, Lambda*sigma**2*zeta2, Lambda*sigma**2*zeta2, sigma**3*zeta2,
                    Lambda**4*zeta1, Lambda**3*sigma*zeta1, Lambda**3*sigma*zeta1, Lambda**2*sigma**2*zeta1, Lambda**3*sigma*zeta1, Lambda**2*sigma**2*zeta1, Lambda**2*sigma**2*zeta1, Lambda*sigma**3*zeta1, Lambda**3*sigma*zeta1, Lambda**2*sigma**2*zeta1, Lambda**2*sigma**2*zeta1, Lambda*sigma**3*zeta1, Lambda**2*sigma**2*zeta1, Lambda*sigma**3*zeta1, Lambda*sigma**3*zeta1, sigma**4*zeta1, Lambda**4*zeta2, Lambda**3*sigma*zeta2, Lambda**3*sigma*zeta2, Lambda**2*sigma**2*zeta2, Lambda**3*sigma*zeta2, Lambda**2*sigma**2*zeta2, Lambda**2*sigma**2*zeta2, Lambda*sigma**3*zeta2, Lambda**3*sigma*zeta2, Lambda**2*sigma**2*zeta2, Lambda**2*sigma**2*zeta2, Lambda*sigma**3*zeta2, Lambda**2*sigma**2*zeta2, Lambda*sigma**3*zeta2, Lambda*sigma**3*zeta2, sigma**4*zeta2])
    TA_p = np.array([0,
                    0, v_0,
                    0, zeta1, 0, zeta2,
                    0, Lambda*zeta1, 0, sigma*zeta1, 0, Lambda*zeta2, 0, sigma*zeta2,
                    0, Lambda**2*zeta1, 0, Lambda*sigma*zeta1, 0, Lambda*sigma*zeta1, 0, sigma**2*zeta1, 0, Lambda**2*zeta2, 0, Lambda*sigma*zeta2, 0, Lambda*sigma*zeta2, 0, sigma**2*zeta2,
                    0, Lambda**3*zeta1, 0, Lambda**2*sigma*zeta1, 0, Lambda**2*sigma*zeta1, 0, Lambda*sigma**2*zeta1, 0, Lambda**2*sigma*zeta1, 0, Lambda*sigma**2*zeta1, 0, Lambda*sigma**2*zeta1, 0, sigma**3*zeta1, 0, Lambda**3*zeta2, 0, Lambda**2*sigma*zeta2, 0, Lambda**2*sigma*zeta2, 0, Lambda*sigma**2*zeta2, 0, Lambda**2*sigma*zeta2, 0, Lambda*sigma**2*zeta2, 0, Lambda*sigma**2*zeta2, 0, sigma**3*zeta2])

'计算各阶签名重构路径'
elements_number = [3, 7, 15, 31, 63] # 对应各阶signature的系数规模
for n in range(1, N+1):
    '导入路径签名'
    sig_hatW = np.load(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy')
    '张量代数与路径签名作内积'
    lw = sig_hatW @ TA_l[:elements_number[n-1]]  # np.array(M_W, J+1)
    '模拟 SABR 并定价'
    option_prices_sig_SDE_MC = np.zeros((M_W, L+1))
    for m_w in range(M_W):
        option_prices_sig_SDE_MC[m_w, :] = compute_sig_SABR_option_prices(lw[m_w, :], paths_dW[m_w, :], paths_dB, rho, beta, space_grid, K, J)
        print(f'[{n}][{m_w + 1}/{M_W}]...')
    np.save(f'data/{name}/results/option_prices/option_prices_sig_SDE_MC_order{n}.npy', option_prices_sig_SDE_MC)  # 形状为(M_W,L+1)

option_prices_sig_SDE_MC = {}
for n in range(1, N+1):
    file_path = f'data/{name}/results/option_prices/option_prices_sig_SDE_MC_order{n}.npy'
    option_prices_sig_SDE_MC[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_SDE_MC = np.load(f'data/{name}/results/option_prices/option_prices_SDE_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(space_grid, np.mean(option_prices_SDE_MC, 0), label='MC', linestyle='-', linewidth=1)
for n in range(1, N+1):
    plt.plot(space_grid, np.mean(option_prices_sig_SDE_MC[f'order{n}'],0), label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'data/{name}/results/plots/option_prices_SDE_MC_1.eps')
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")


