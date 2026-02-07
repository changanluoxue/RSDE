import numpy as np
import iisignature
import math
from scipy.linalg import hankel
import os
import time

start_time = time.perf_counter()

def simulate_ou(kappa, theta, eta, v_0, T, J, M):
    '''
    kappa:均值回复速率
    theta:长期均值
    eta:波动率
    v_0:初值
    T:时间长度
    J:时间网格数
    M:路径数
    return: paths_v:np.array((M, J+1)); paths_dW:np.array((M, J)); paths_W:np.array((M, J+1))
    '''
    dt = T/J
    paths_v = np.zeros((M, J+1))
    paths_v[:, 0] = v_0
    paths_dW = np.zeros((M, J))

    for j in range(1, J+1):
        paths_dW[:, j-1] = np.sqrt(dt)*np.random.standard_normal(M)
        paths_v[:, j] = paths_v[:, j-1] + kappa*(theta-paths_v[:, j-1])*dt + eta*paths_dW[:, j-1]
    paths_W = np.cumsum(np.hstack([np.zeros((M, 1)), paths_dW]), axis=1)
    return paths_v, paths_dW, paths_W
def simulate_mGBM(kappa, theta, sigma, eta, v_0, T, J, M):
    '''
    kappa:均值回复速率
    theta:长期均值
    sigma:波动率
    eta:
    v_0:初值
    T:时间长度
    J:时间网格数
    M:路径数
    return: paths_v:np.array((M, J+1)); paths_dW:np.array((M, J)); paths_W:np.array((M, J+1))
    '''
    dt = T/J
    paths_v = np.zeros((M, J+1))
    paths_v[:, 0] = v_0
    paths_dW = np.zeros((M, J))

    for j in range(1, J+1):
        paths_dW[:, j-1] = np.sqrt(dt)*np.random.standard_normal(M)
        paths_v[:, j] = paths_v[:, j-1] + kappa*(theta-paths_v[:, j-1])*dt + (eta+sigma*paths_v[:, j-1])*paths_dW[:, j-1]

    paths_W = np.cumsum(np.hstack([np.zeros((M, 1)), paths_dW]), axis=1)
    return paths_v, paths_dW, paths_W
def simulate_rBergomi(J, M, alpha, eta, dt, time_grid, v_0):
    '''
    This function generates M trajectories of the rBergomi volatility process.
    '''
    paths_v = np.zeros((M, J+1))
    paths_dW = np.sqrt(dt)*np.random.standard_normal((M, J))
    for j in range(1, J+1):
        paths_v[:, j] = v_0*np.exp(eta*np.sum((time_grid[j]-time_grid[0:j])**(-alpha)*paths_dW[:, 0:j], axis=-1))
    paths_W = np.cumsum(np.hstack([np.zeros((M, 1)), paths_dW]), axis=-1)
    return paths_v, paths_dW, paths_W
def simulate_rHeston(alpha, kappa, theta, sigma, v_0, T, J, M, reps=1e-6):
    '''
    This function generates M trajectories of the rHeston volatility process.
    '''
    def myls(A, B, eps):
        (m, n) = np.shape(A)
        (U, S, V) = np.linalg.svd(A); V = V.T
        r = np.sum(S > eps)
        x = np.zeros(n)
        for i in range(r):
            x = x + (np.sum(B*U[:, i])/S[i])*V[:, i]
        res = np.linalg.norm(np.dot(A, x)-B)/np.linalg.norm(B)
        return x, res
    def myls2(A, B, eps):
        (m, n) = np.shape(A)
        (Q, R) = np.linalg.qr(A)
        s = np.diag(R); r = np.sum(abs(s)>eps)
        Q = Q[:, 0:r]; R = R[0:r, 0:r]
        b1 = B[r:m+r]
        x = np.dot(np.linalg.inv(R), (np.dot(Q.T, b1)))
        return x
    def prony(xs, ws):
        M = len(xs); errbnd = 1e-12; h = np.zeros(2 * M)
        for j in range(2*M):
            h[j] = np.dot(xs**j, ws)
        C = np.zeros(M); R = np.zeros(M)
        for i in range(M):
            C[i] = h[i]; R[i] = h[i+M-1]
        H = hankel(C, R); b = -h
        q = myls2(H, b, errbnd); r = len(q); A = np.zeros((2*M, r))
        Coef = np.insert(np.flipud(q), 0, 1)
        xsnew = np.roots(Coef)
        for j in range(2*M):
            A[j, :] = xsnew**j
        (wsnew, res) = myls(A, h, errbnd); ind = np.where(np.real(xsnew)>=0); p = len(ind[0])
        assert np.sum(abs(wsnew[ind])<1e-15) == p
        ind = np.where(np.real(xsnew)<0)
        xsnew = xsnew[ind]; wsnew = wsnew[ind]
        return wsnew, xsnew
    def SOEapppr(beta, reps, dt, Tfinal):
        delta = dt/Tfinal
        h = 2*math.pi/(math.log(3) + beta*math.log(1/math.cos(1)) + math.log(1/reps))
        tlower = 1/beta*math.log(reps*math.gamma(1+beta))
        if beta >= 1:
            tupper = math.log(1/delta) + math.log(math.log(1/reps)) + math.log(beta) + 1/2
        else:
            tupper = math.log(1/delta) + math.log(math.log(1/reps))
        M = math.floor(tlower/h); N = math.ceil(tupper/h)
        xs1 = np.zeros(abs(M)); ws1 = np.zeros(abs(M))
        for n1 in range(M, 0):
            xs1[n1 - M] = -math.exp(h*n1); ws1[n1-M] = h/math.gamma(beta)*math.exp(beta*h*n1)
        (ws1new, xs1new) = prony(xs1, ws1)
        xs2 = np.zeros(N+1); ws2 = np.zeros(N+1)
        for n2 in range(N+1):
            xs2[n2] = -math.exp(h*n2)
            ws2[n2] = h/math.gamma(beta)*math.exp(beta*h*n2)
        xs = np.append(-np.real(xs1new), -np.real(xs2)); ws = np.append(np.real(ws1new), np.real(ws2))
        xs = xs/Tfinal; ws = ws/Tfinal**beta
        nexp = len(ws)
        return xs, ws, nexp
    def drift_v(a, b, v):
        return a*(b-v)
    def vol_v(c, v):
        return c*np.sqrt(v)
    dt = T/J
    paths_v = np.zeros((M, J+1))
    paths_v[:, 0] = v_0
    (xl, wl, nexp) = SOEapppr(alpha, reps, dt, dt*J)
    i1 = np.zeros((M, nexp)); i2 = np.zeros((M, nexp))
    paths_dW = np.zeros((M, J))
    for j in range(1, J+1):
        paths_dW[:, j-1] = np.sqrt(dt)*np.random.standard_normal(M)
        I1 = drift_v(kappa, theta, paths_v[:, j - 1]) * (dt ** (1 - alpha)) / math.gamma(2 - alpha) + (1 / math.gamma(1 - alpha)) * np.sum(wl * np.exp(-xl * dt) * i1, 1)
        I2 = vol_v(sigma, paths_v[:, j - 1]) * (dt ** (-alpha)) * paths_dW[:, j - 1] / math.gamma(1 - alpha) + (1 / math.gamma(1 - alpha)) * np.sum(wl * np.exp(-xl * dt) * i2, 1)
        paths_v[:, j] = np.maximum(v_0+I1+I2, 0)
        i1 = np.exp(-xl*dt)*i1 + ((1-np.exp(-xl*dt))/xl)*np.reshape(drift_v(kappa, theta, paths_v[:, j - 1]), (M, 1))
        i2 = np.exp(-xl*dt)*i2 + np.exp(-xl*dt)*np.reshape(paths_dW[:, j-1] * vol_v(sigma, paths_v[:, j - 1]), (M, 1))
    paths_W = np.cumsum(np.hstack([np.zeros((M, 1)), paths_dW]), axis=1)
    return paths_v, paths_dW, paths_W
def simulate_I(paths_v, paths_dW, J, M):
    '''
    return: paths_I:np.array((M, J+1))
    '''
    paths_I = np.zeros((M, J+1))

    for j in range(1, J+1):
        paths_I[:, j] = paths_I[:, j-1] + paths_v[:, j-1]*paths_dW[:, j-1]
    return paths_I
def compute_signatures(paths, sig_order):
    """
    paths:np.array((M,J+1,d))
    sig_order:int
    sigs:np.array((M,J+1,d_sig))
    """
    M, Jp1, d = paths.shape
    d_sig = iisignature.siglength(d, sig_order)+1
    sigs = np.zeros((M, Jp1, d_sig))
    sigs[:, :, 0] = 1
    for j in range(1, Jp1):
        sig = iisignature.sig(paths[:, :j+1], sig_order)
        sigs[:, j, 1:] = sig
    return sigs

'通用常数系数'
name = "ou" # "ou", "mGBM", "rBergomi", "rHeston"
T = 1; J = 251; M = 10000; dt = T/J
v_0 = 0.25; N = 5

'构造时间网格'
time_grid = np.linspace(0, T, J+1)

'模拟布朗运动, 积分过程I与波动率过程并储存'
if name == "ou":
    kappa = 1; theta = 0.25; eta = 1.2
    paths_v, paths_dW, paths_W = simulate_ou(kappa, theta, eta, v_0, T, J, M)
if name == "mGBM":
    kappa = 1; theta = 0.25; sigma = 0.5; eta = 0
    paths_v, paths_dW, paths_W = simulate_mGBM(kappa, theta, sigma, eta, v_0, T, J, M)
if name == "rBergomi":
    alpha = 0.2; eta = 1
    paths_v, paths_dW, paths_W = simulate_rBergomi(J, M, alpha, eta, dt, time_grid, v_0)
if name == "rHeston":
    kappa = 0.1; theta = 0.25; sigma = 0.01; alpha = 0.2
    paths_v, paths_dW, paths_W = simulate_rHeston(alpha, kappa, theta, sigma, v_0, T, J, M)
np.save(f'data/{name}/simulated_data/paths_v.npy', paths_v)
np.save(f'data/{name}/simulated_data/paths_dW.npy', paths_dW)
np.save(f'data/{name}/simulated_data/paths_W.npy', paths_W)

paths_I = simulate_I(paths_v, paths_dW, J, M)
np.save(f'data/{name}/simulated_data/paths_I.npy', paths_I)

'paths_hatW为布朗运动W的时间增强(t, W)'
paths_hatW = np.zeros((M, J+1, 2))
paths_hatW[:, :, 0] = time_grid
paths_hatW[:, :, 1] = paths_W

'计算paths_hatW在[0,T]上的各阶路径签名'
for n in range(1, N+1):
    sig_hatW = compute_signatures(paths_hatW, n)
    np.save(f'data/{name}/simulated_data/sig_hatW/sig_hatW_all_order{n}.npy', sig_hatW)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time:.2f} 秒")
