import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.perf_counter()


def sig_representation_SABR(paths_dW, paths_dB, rho, beta, x_0, J, M):
    '''
    paths_dW:布朗运动增量:np.array((M, J))
    paths_dB:布朗运动增量:np.array((M, J))
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
        paths_SABR[:, j] = (paths_SABR[:, j-1]
                            + np.sqrt(1-rho**2)*paths_SABR[:, j-1]**beta*lw[:, j-1]*paths_dB[:, j-1]
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

'常数系数'
name = "ou" # "ou", "mGBM"
v_0 = 0.25; T = 1; J = 251; M = 10000; dt = T/J
N = 5
x_low = 80; x_high = 120; L = 400; dx = (x_high-x_low)/L; X_0 = np.linspace(x_low, x_high, L+1)
rho = -0.4; beta = 1
K = 110

'构造时间、空间网格'
time_grid = np.linspace(0, T, J+1)
space_grid = np.linspace(x_low, x_high, L+1)

'导入布朗运动'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy') # np.array(M, J)
paths_dB = np.load(f'data/{name}/simulated_data/paths_dB.npy') # np.array(M, J, L+1)

'输入系数矩阵解析式'
if name == 'ou':
    kappa = 1; theta = 0.25; eta = 1.2
    TA_l = np.array([v_0,
                     -kappa*(v_0-theta), eta,
                     kappa**2*(v_0-theta), 0, -kappa*eta, 0,
                     -kappa**3*(v_0-theta), 0, 0, 0, kappa**2*eta, 0, 0, 0,
                     kappa**4*(v_0-theta), 0, 0, 0, 0, 0, 0, 0, -kappa**3*eta, 0, 0, 0, 0, 0, 0, 0,
                     -kappa**5*(v_0-theta), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kappa**4*eta, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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

'计算各阶签名重构路径'
def fpw(x, j, rho=rho, beta=beta):
    '''
    return: np.array(M,)
    '''
    return rho*x**beta+rho**2*beta*x**(2*beta-1)*pw[:,j]
elements_number = [3, 7, 15, 31, 63] # 对应各阶signature的系数规模
for n in range(1, N+1):
    '导入路径签名'
    sig_hatW = np.load(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy')
    '张量代数与路径签名作内积'
    lw = sig_hatW @ TA_l[:elements_number[n-1]]  # np.array(M, J+1)
    pw = simulate_I(lw, paths_dW, J, M)  # np.array(M, J+1)
    '计算SABR签名重构路径'
    sig_paths_SABR = np.zeros((M, J+1, L+1))
    for i, x_0 in enumerate(X_0):
        sig_paths_SABR[:, :, i] = sig_representation_SABR(paths_dW, paths_dB[:, :, i], rho, beta, x_0, J, M)
        print(f'进度[{n}/{N}][{i}/{L}]...')
    '欧式期权定价'
    option_prices_sig_MC = np.mean(eur_put_option(sig_paths_SABR, K), 0)
    np.save(f'data/{name}/results/option_prices/option_prices_sig_MC_order{n}.npy', option_prices_sig_MC)  # 形状为(L+1, )
option_prices_sig_MC = {}
for n in range(1, N+1):
    file_path = f'data/{name}/results/option_prices/option_prices_sig_MC_order{n}.npy'
    option_prices_sig_MC[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_MC = np.load(f'data/{name}/results/option_prices/option_prices_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(X_0, option_prices_MC, label='MC', linestyle='-', linewidth=1)
for n in range(1, N+1):
    plt.plot(X_0, option_prices_sig_MC[f'order{n}'], label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")


