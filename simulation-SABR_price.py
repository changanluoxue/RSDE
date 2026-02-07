import numpy as np
import time

start_time = time.perf_counter()

def simulate_SABR(paths_v, paths_dW, rho, beta, x_0, T, J, M):
    '''
    paths_v:波动率过程:np.array((M, J+1))
    paths_dW:布朗运动增量:np.array((M, J))
    rho:SABR相关系数
    beta:SABR弹性指数
    x_0:初值
    T:时间长度
    J:时间网格数
    M:路径数
    return: paths_SABR:np.array((M, J+1)); paths_dB:np.array((M, J))
    '''
    dt = T/J
    paths_SABR = np.zeros((M, J+1))
    paths_SABR[:, 0] = x_0
    paths_dB = np.zeros((M, J))

    for j in range(1, J+1):
        paths_dB[:, j-1] = np.sqrt(dt)*np.random.standard_normal(M)
        paths_SABR[:, j] = (paths_SABR[:, j-1]
                            + paths_SABR[:, j-1]**beta*paths_v[:, j-1]*(rho*paths_dW[:, j-1]+np.sqrt(1-rho**2)*paths_dB[:, j-1]))
    return paths_SABR, paths_dB
def eur_put_option(paths_price, K):
    option_prices = np.maximum(K-paths_price[:,-1,:], 0)
    return option_prices

'通用常数系数'
name = "ou" # "ou", "mGBM", "rBergomi", "rHeston"
T = 1; J = 251; M=10000
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L
rho = -0.4; beta = 1
K = 110

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动与波动率过程'
paths_v = np.load(f'data/{name}/simulated_data/paths_v.npy')
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')

'模拟SABR过程'
paths_SABR = np.zeros((M, J+1, L+1)); paths_dB = np.zeros((M, J, L+1))
for i, x_0 in enumerate(space_grid):
    paths_SABR[:, :, i], paths_dB[:, :, i] = simulate_SABR(paths_v, paths_dW, rho, beta, x_0, T, J, M)
np.save(f'data/{name}/simulated_data/paths_SABR.npy', paths_SABR)
np.save(f'data/{name}/simulated_data/paths_dB.npy', paths_dB)

'计算欧式期权价格'
option_prices = np.mean(eur_put_option(paths_SABR, K), 0)
np.save(f'data/{name}/results/option_prices/option_prices_MC.npy', option_prices) #形状为(L+1, )

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")
