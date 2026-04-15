import torch
import torch.nn as nn
import numpy as np
from tabulate import tabulate
import time

start_time = time.perf_counter()

def simulate_I(paths_v, paths_dW, J, M_W):
    '''
    return: paths_I:np.array((M_W, J+1))
    '''
    paths_I = np.zeros((M_W, J+1))

    for j in range(1, J+1):
        paths_I[:, j] = paths_I[:, j-1] + paths_v[:, j-1]*paths_dW[:, j-1]
    return paths_I
def fnn(input_size=252, hidden_sizes=(16, 16, 16), output_size=1, activation=nn.ReLU()):
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
T = 1; J = 251; M_W = 10000; dt = T/J
orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
order_feature = {1:4, 2:8, 3:16, 4:32, 5:64, 6:128, 7:256, 8:512, 9:1024}
time_grid = np.linspace(0, T, J+1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'导入布朗运动、波动率过程和积分过程'
paths_dW = np.load(f'data/{name}/simulated_data/paths_dW.npy')
paths_v = np.load(f'data/{name}/simulated_data/paths_v.npy')
paths_I = np.load(f'data/{name}/simulated_data/paths_I.npy')

'计算签名重构路径'
paths_v_hat_all = {}; paths_I_hat_all = {}
for n in orders:
    '导入路径签名'
    sig_hatW = np.load(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy') # (M_W, J+1, features)
    timee = np.broadcast_to(np.expand_dims(np.linspace(0, T, J+1), axis=(0, -1)), (sig_hatW.shape[0], sig_hatW.shape[1], 1))
    sig_hatW_t = np.concatenate((sig_hatW, timee), axis=-1)

    '模型加载及输出'
    if method == 'nonlinear':
        nn_model = signn_lw(features=order_feature[n])
        nn_model.load_state_dict(torch.load(f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth', map_location=device))
        nn_model.eval()
        paths_v_hat_all[f'order{n}'] = nn_model(torch.from_numpy(sig_hatW_t).float()).detach().squeeze().numpy()  # (M_W, J+1)
        paths_I_hat_all[f'order{n}'] = simulate_I(paths_v_hat_all[f'order{n}'], paths_dW, J, M_W)
    if method == 'linear':
        nn_model = signn_l(features=order_feature[n])
        nn_model.load_state_dict(torch.load(f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth', map_location=device))
        nn_model.eval()
        paths_v_hat = nn_model(torch.from_numpy(sig_hatW_t).float()).detach().numpy()  # (M_W, J+1, features)
        paths_v_hat_all[f'order{n}'] = np.sum(paths_v_hat*sig_hatW, -1) # (M_W, J+1)
        paths_I_hat_all[f'order{n}'] = simulate_I(paths_v_hat_all[f'order{n}'], paths_dW, J, M_W)
np.save(f'data/{name}/results/representations/paths_v_hat_all_{method}.npy', paths_v_hat_all)
np.save(f'data/{name}/results/representations/paths_I_hat_all_{method}.npy', paths_I_hat_all)

'输出误差'
mae_v_all = {}; mae_I_all = {}
std_v_all = {}; std_I_all = {}
for n in orders:
    er_v = np.abs(paths_v - paths_v_hat_all[f'order{n}'])
    er_I = np.abs(paths_I - paths_I_hat_all[f'order{n}'])
    mae_v_all[f'order{n}'] = np.mean(er_v)
    mae_I_all[f'order{n}'] = np.mean(er_I)
    std_v_all[f'order{n}'] = np.std(np.mean(er_v, -1))
    std_I_all[f'order{n}'] = np.std(np.mean(er_I, -1))

'打印误差表'
print(f'SABR_{name}_nn_sig_representation_v_I_{method}:')
header = [('', ''), ('v', 'MAE'), ('v', 'std.'), ('I', 'MAE'), ('I', 'std.')]
table_data = []
for n in orders:
    row = [f"N={n}"]
    row.extend([mae_v_all[f'order{n}'], std_v_all[f'order{n}'],
                mae_I_all[f'order{n}'], std_I_all[f'order{n}']])
    table_data.append(row)
print(tabulate(table_data, headers = header, floatfmt = (".8f", ".8f", ".8f", ".8f", ".8f"), tablefmt = "grid"))

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")