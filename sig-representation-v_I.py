from tabulate import tabulate
import numpy as np
import time

start_time = time.perf_counter()

def simulate_I(paths_v, paths_dW, J, M_W):
    '''
    return:
        paths_I:np.array((M_W, J+1))
    '''
    paths_I = np.zeros((M_W, J+1))
    for j in range(1, J+1):
        paths_I[:, j] = paths_I[:, j-1] + paths_v[:, j-1]*paths_dW[:, j-1]
    return paths_I

'常数系数'
name = "ou" # "ou", "mGBM"
v_0 = 0.25; T = 1; J = 251; M_W = 10000; dt = T/J
N = 5

'导入波动率过程和积分过程'
paths_v = np.load(f'data/{name}/simulated_data/paths_v.npy')
paths_I = np.load(f'data/{name}/simulated_data/paths_I.npy')
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')

'定义系数解析式'
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

'计算各阶签名重构路径并保存'
paths_v_hat_all = {}; paths_I_hat_all = {}
elements_number = [3, 7, 15, 31, 63] # 对应各阶signature的系数规模
for n in range(1, N+1):
    '导入路径签名'
    sig_hatW = np.load(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy')
    '利用签名线性组合重构路径'
    paths_v_hat_all[f'order{n}'] = sig_hatW @ TA_l[:elements_number[n-1]] # np.array(M_W, J+1)
    paths_I_hat_all[f'order{n}'] = simulate_I(paths_v_hat_all[f'order{n}'], paths_dW, J, M_W) # np.array(M_W, J+1)
np.save(f'data/{name}/results/representations/paths_v_hat_all.npy', paths_v_hat_all)
np.save(f'data/{name}/results/representations/paths_I_hat_all.npy', paths_I_hat_all)

'计算误差'
mae_v_all = {}; mae_I_all = {}
std_v_all = {}; std_I_all = {}
for n in range(1, N+1):
    er_v = np.abs(paths_v - paths_v_hat_all[f'order{n}'])
    er_I = np.abs(paths_I - paths_I_hat_all[f'order{n}'])
    mae_v_all[f'order{n}'] = np.mean(er_v)
    mae_I_all[f'order{n}'] = np.mean(er_I)
    std_v_all[f'order{n}'] = np.std(np.mean(er_v, -1))
    std_I_all[f'order{n}'] = np.std(np.mean(er_I, -1))

'打印误差表'
print(f'SABR_{name}_sig_representation_v_I:')
header = [('', ''), ('v', 'MAE'), ('v', 'std.'), ('I', 'MAE'), ('I', 'std.')]
table_data = []
for n in range(1, N+1):
    row = [f"N={n}"]
    row.extend([mae_v_all[f'order{n}'], std_v_all[f'order{n}'],
                mae_I_all[f'order{n}'], std_I_all[f'order{n}']])
    table_data.append(row)
print(tabulate(table_data, headers = header, floatfmt = (".8f", ".8f", ".8f", ".8f", ".8f"), tablefmt = "grid"))

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")
