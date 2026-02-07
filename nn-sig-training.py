import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

start_time = time.perf_counter()

def training_model_nonlinear(target_paths, input_paths, nn_model, num_epochs=5000, min_loss=1e-10, batch_size=64,
                   mse_weight=0.5, max_squared_weight=0.5, lr=1e-6):
    '''
    target_paths: 目标路径: np.array(M, J+1)
    input_paths: 输入路径: np.array(M, J+1, features)
    nn_model: 需要训练的神经网络模型
    return: nn_model, loss_history: 训练完成的神经网络模型及训练过程数据
    '''
    '将输入数据转换成张量形式'
    target_paths = torch.from_numpy(target_paths).float().to(device)
    input_paths = torch.from_numpy(input_paths).float().to(device)

    '''生成训练集及目标'''
    dataset = TensorDataset(input_paths, target_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    '''初始化模型并定义训练器'''
    nn_model = nn_model.to(device)
    criterion = MixedLoss(mse_weight=mse_weight, max_squared_weight=max_squared_weight)
    optimizer = optim.AdamW(nn_model.parameters(), lr=lr)

    '''训练模型'''
    loss_history = []
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            '前向传播'
            outputs = nn_model(inputs).squeeze()  # (batch_size, J+1)
            '计算损失'
            loss = criterion(outputs, targets)
            '反向传播和优化'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f'{nn_model.__class__.__name__}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.11f}')

        '检查停止条件'
        if loss.item() < min_loss:
            print(f'Training stopped: Loss reached threshold at Epoch{epoch}.')
            break
    return nn_model, loss_history
def training_model_linear(target_paths, input_paths, sig_hatW, nn_model, num_epochs=10000, min_loss=1e-10, batch_size=64,
                   mse_weight=0.5, max_squared_weight=0.5, lr=1e-6):
    '''
    target_paths: 目标路径: np.array(M, J+1)
    input_paths: 时间增强路径签名: np.array(M, J+1, features+1)
    nn_model: 需要训练的神经网络模型
    return: nn_model, loss_history: 训练完成的神经网络模型及训练过程数据
    '''
    '将输入数据转换成张量形式'
    target_paths = torch.from_numpy(target_paths).float().to(device)
    input_paths = torch.from_numpy(input_paths).float().to(device)
    sig_hatW = torch.from_numpy(sig_hatW).float().to(device)

    '''生成训练集及目标'''
    dataset = TensorDataset(input_paths, target_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    '''初始化模型并定义训练器'''
    nn_model = nn_model.to(device)
    criterion = MixedLoss(mse_weight=mse_weight, max_squared_weight=max_squared_weight)
    optimizer = optim.AdamW(nn_model.parameters(), lr=lr)

    '''训练模型'''
    loss_history = []
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            '前向传播'
            outputs = nn_model(inputs)  # (batch_size, J+1, features)
            batch_sig = sig_hatW[batch_idx*dataloader.batch_size:(batch_idx+1)*dataloader.batch_size]
            outputs = (batch_sig*outputs).sum(dim=-1) # (batch_size, J+1)
            '计算损失'
            loss = criterion(outputs, targets)
            '反向传播和优化'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_history.append(loss.item())
        if epoch % 10 == 0:
            print(f'{nn_model.__class__.__name__}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.11f}')

        '检查停止条件'
        if loss.item() < min_loss:
            print(f'Training stopped: Loss reached threshold at Epoch{epoch}.')
            break
    return nn_model, loss_history
class MixedLoss(nn.Module):
    def __init__(self, mse_weight=1.0, max_squared_weight=1.0):
        super(MixedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.max_squared_weight = max_squared_weight
        self.mse_loss = nn.MSELoss()

    def max_squared_loss(self, y_pred, y_true):
        squared_loss = torch.pow(y_pred-y_true, 2)
        squared_loss = squared_loss.mean(dim=-1)
        return squared_loss.max()

    def forward(self, y_pred, y_true):
        mse_loss = self.mse_loss(y_pred, y_true)
        max_squared_loss = self.max_squared_loss(y_pred, y_true)
        total_loss = self.mse_weight*mse_loss + self.max_squared_weight*max_squared_loss
        return total_loss
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

'通用常数系数'
name = "rHeston" # "rBergomi", "rHeston"
method = "linear" # "linear", "nonlinear"
T = 1; J = 251; M = 10000
lr = 1e-6
order_feature = {1:4, 2:8, 3:16, 4:32, 5:64, 6:128, 7:256, 8:512, 9:1024}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for n in range(1,10):
    '导入目标路径及签名数据'
    paths_v = np.load(f'data/{name}/simulated_data/paths_v.npy')
    sig_hatW = np.load(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy')
    timee = np.broadcast_to(np.expand_dims(np.linspace(0, T, J + 1), axis=(0, -1)), (sig_hatW.shape[0], sig_hatW.shape[1], 1))
    sig_hatW_t = np.concatenate((sig_hatW, timee), axis=-1)

    '训练及保模型存'
    order_epoch = 10000; min_loss = 1e-10;
    if method == "nonlinear":
        nn_model, loss_history = training_model_nonlinear(paths_v, sig_hatW_t, signn_lw(features=order_feature[n]),
                                                          num_epochs=order_epoch, min_loss=min_loss,
                                                          batch_size=64, mse_weight=0.5, max_squared_weight=0.5, lr=lr)
        torch.save(nn_model.state_dict(), f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth')
    if method == "linear":
        nn_model, loss_history = training_model_linear(paths_v, sig_hatW_t, sig_hatW, signn_l(features=order_feature[n]),
                                                       num_epochs=order_epoch, min_loss=min_loss,
                                                       batch_size=64, mse_weight=0.5, max_squared_weight=0.5, lr=lr)
        torch.save(nn_model.state_dict(), f'data/{name}/results/nn_models/nn_sig_model_{method}_order{n}.pth')

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")


