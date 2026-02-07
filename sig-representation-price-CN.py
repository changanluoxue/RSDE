import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pyplot as plt

start_time = time.perf_counter()

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
        # u[:, :-1, 0] = payoff_func(K, space_grid[0])  # 边值条件(Low)
        # u[:, :-1, -1] = 0  # 边值条件(High)

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
    lw = sig_hatW @ TA_l[:elements_number[n-1]] # np.array(M, J+1)
    pw = simulate_I(lw, paths_dW, J, M) # np.array(M, J+1)
    '求解PDE'
    option_prices_sig_CN = np.mean(Crank_Nicolson(), 0)[0, :]
    np.save(f'data/{name}/results/option_prices/option_prices_sig_CN_order{n}.npy', option_prices_sig_CN)  # 形状为(L+1, )

option_prices_sig_CN = {}
for n in range(1, N+1):
    file_path = f'data/{name}/results/option_prices/option_prices_sig_CN_order{n}.npy'
    option_prices_sig_CN[f'order{n}'] = np.load(file_path)

'导入MC结果作为基准'
option_prices_MC = np.load(f'data/{name}/results/option_prices/option_prices_MC.npy')

'绘制曲线'
plt.figure(figsize=(10, 6))
plt.plot(X_0, option_prices_MC, label='MC', linestyle='-', linewidth=1)
for n in range(1, N+1):
    plt.plot(X_0, option_prices_sig_CN[f'order{n}'], label=f'N={n}', linestyle='--', linewidth=1)
plt.xlabel('Initial stock price', fontsize=12)
plt.ylabel('Option price at t=0', fontsize=12)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")


