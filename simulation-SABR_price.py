import numpy as np
import time
import numba as nb

start_time = time.perf_counter()

def simulate_Brownianmotion(M, J, T):
    dt = T/J
    return np.sqrt(dt)*np.random.standard_normal((M, J))
@nb.njit(parallel=True, fastmath=True)
def compute_SABR_option_prices(paths_v_mw, paths_dW_mw, paths_dB, rho, beta, space_grid, K, J):
    '''
    paths_v_mw: np.array((J+1, ))
    paths_dW_mw: np.array((J, ))
    paths_dB: np.array((M_B, J))
    return:
        对应单条W的欧式期权价格np.array((L+1, ))
    '''
    M_B = paths_dB.shape[0]; L = space_grid.shape[0]-1
    sqrt_1_rho2 = np.sqrt(1-rho**2)
    payoffs = np.zeros((M_B, L+1))
    '模拟M_B条标的资产价格路径'
    for mb in nb.prange(M_B):
        '仅保留当前时刻的标的资产价格'
        current_X = np.copy(space_grid)
        for j in range(J):
            v = paths_v_mw[j]; dW = paths_dW_mw[j]; dB = paths_dB[mb, j]
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
name = "rHeston"  # "ou", "mGBM", "rBergomi", "rHeston"
T = 1; J = 251; M_B = 10000; M_W = 10000
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L
rho = -0.4; beta = 0.6
K = 110

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动与波动率过程'
paths_v = np.load(f'data/{name}/simulated_data/paths_v.npy')
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')
paths_dB = simulate_Brownianmotion(M_B, J, T)
np.save(f'data/{name}/simulated_data/paths_dB.npy', paths_dB)

'模拟SABR并定价'
option_prices = np.zeros((M_W, L+1))
for m_w in range(M_W):
    option_prices[m_w, :] = compute_SABR_option_prices(paths_v[m_w, :], paths_dW[m_w, :], paths_dB, rho, beta, space_grid, K, J)
    print(f'[{m_w+1}/{M_W}]...')
np.save(f'data/{name}/results/option_prices/option_prices_SDE_MC.npy', option_prices) # 形状为(M_W,L+1)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")