This work implements "Option pricing under non-Markovian stochastic volatility models: A deep signature approach" by Jingtang Ma, Xianglin Wu, and Wenyuan Li. https://doi.org/10.48550/arXiv.2508.15237

### Markovian Example (OU, mGBM):
* `simulation-v-I`: Used to simulate datasets v and I, while simultaneously computing W, dW, and sig_W.
* `simulation-SABR_price`: Computes the underlying asset paths, calculates option prices to serve as the Monte Carlo (MC) benchmark, and computes the Brownian motion B.
* `sig-representation-v_I`: Used to compare the differences between v, I and sig_v, sig_I.
* `sig-representation-price-SDE-MC`: Calculates option prices using Stochastic Differential Equations (SDE) and compares them with the MC benchmark.
* `sig-representation-price-PDE-CN`: Calculates option prices using Partial Differential Equations (PDE) via the Crank-Nicolson method.
* `sig-representation-price-PDE-analytical`: Calculates option prices using PDEs (Analytical solution, strictly for the beta=1 case).

### Non-Markovian Example (rHeston, rBergomi):
* `nn-sig-training`: Used for training the neural network.
* `nn-sig-representation-v_I`: Used to compare the differences between v, I and sig_v, sig_I.
* `nn-sig-representation-price-SDE-MC`: Calculates option prices using SDEs and compares them with the MC benchmark.
* `nn-sig-representation-price-PDE-CN`: Calculates option prices using PDEs via the Crank-Nicolson method.
