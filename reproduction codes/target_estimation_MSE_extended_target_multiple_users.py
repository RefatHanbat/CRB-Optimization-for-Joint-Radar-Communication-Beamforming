import numpy as np

import cvxpy as cp

import matplotlib.pyplot as plt

def dbm_to_watt(dbm):

    return 10 ** ((dbm - 30) / 10)

def cn_rand(shape, rng):

    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

def solve_eq36_extended_multiuser(H, Gamma_lin, Pt_watt, Noise_C_watt):

    K, Nt = H.shape

    W = [cp.Variable((Nt,Nt), hermitian = True) for _ in range(K + 1)]

    RX = sum(W)

    U = cp.Variable((Nt, Nt), hermitian = True)

    I = np.eye(Nt)

    block = cp.bmat([

        [U, I],

        [I, RX]
    ])

    constraints = [

        block >> 0,

        cp.real(cp.trace(RX)) <= Pt_watt
    ]

    for i in range(K + 1):

        constraints.append(W[i] >> 0)

    for k in range(K):

        hk = H[k, :].reshape(-1,1)

        Qk = hk @ hk.conj().T

        desired = cp.real(cp.trace(Qk @ W[k]))

        interference = 0

        for i in range(K + 1):

            if i != k:

                interference += cp.real(cp.trace(Qk @ W[i]))

        constraints.append(

            desired - Gamma_lin * interference >= Gamma_lin * Noise_C_watt
        )

    objective = cp.Minimize(cp.real(cp.trace(U)))

    prob = cp.Problem(objective,constraints)

    try:
        prob.solve(solver = cp.MOSEK, verbose = False)

    except:

        prob.solve(solver = cp.CLARABEL, verbose = False)

    if prob.status not in ["optimal","optimal_inaccurate"]:

        return None, None, prob.status
    
    W_values = [Wi.value for Wi in W]

    RX_value = np.zeros((Nt,Nt), dtype=complex)

    for Wi in W_values:

        RX_value += Wi

    return W_values, RX_value, prob.status


def extended_target_mse_db(RX, Noise_R_watt, Nr, L, is_evd = False):

    RX = (RX + RX.conj().T) / 2


    eigvals, eigvecs = np.linalg.eigh(RX)


    if is_evd:

        eigvals = np.maximum(eigvals, 1e-3)   

    else:

        eigvals = np.maximum(eigvals, 1e-8)


    RX_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.conj().T

    mse = (Noise_R_watt * Nr / L) * np.real(np.trace(RX_inv))

    return 10 * np.log10(mse + 1e-12)

def rank1_evd_approx(W_values):

    W_rank1_values = []

    for W in W_values:

        W = (W + W.conj().T) / 2

        eigvals, eigvecs = np.linalg.eigh(W)

        idx = np.argmax(eigvals)

        lambda_max = max(eigvals[idx], 0)

        u_max = eigvecs[:, idx].reshape(-1,1)

        W_rank1 = lambda_max * (u_max @ u_max.conj().T)

        W_rank1_values.append(W_rank1)

    RX_rank1 = np.zeros_like(W_values[0], dtype=complex)

    for W1 in W_rank1_values:

        RX_rank1 += W1

    return W_rank1_values, RX_rank1


####### Parameters ##########


Nt = 16

Nr = 20


L = 30

Pt_dbm = 30 

Pt_watt = dbm_to_watt(Pt_dbm)

Noise_C_dbm = 0

Noise_C_watt = dbm_to_watt(Noise_C_dbm)

Noise_R_dbm = 0

Noise_R_watt = dbm_to_watt(Noise_R_dbm)


Gamma_dB_list = np.arange(2, 22, 2)

def run_simulation(K, Nt, Nr, L, Pt_watt, Noise_C_watt, Noise_R_watt, Gamma_dB_list, rng):

    H = cn_rand((K, Nt), rng)

    mse_eq36_db = []
    mse_evd_db_list = []

    for Gamma_dB in Gamma_dB_list:

        Gamma_lin = 10**(Gamma_dB / 10)

        W_values, RX_opt, status = solve_eq36_extended_multiuser(

            H=H,

            Gamma_lin=Gamma_lin,

            Pt_watt=Pt_watt,

            Noise_C_watt=Noise_C_watt
        )

        if status in ["optimal", "optimal_inaccurate"]:

            mse_db = extended_target_mse_db(RX_opt, Noise_R_watt, Nr, L)

            _, RX_evd = rank1_evd_approx(W_values)

            mse_evd_db = extended_target_mse_db(RX_evd, Noise_R_watt, Nr, L, is_evd=True)

        else:

            mse_db = np.nan

            mse_evd_db = np.nan

        mse_eq36_db.append(mse_db)

        mse_evd_db_list.append(mse_evd_db)

    return np.array(mse_eq36_db), np.array(mse_evd_db_list)


Gamma_dB_list = np.arange(2, 22, 2)

rng6 = np.random.default_rng(30245109)

rng12 = np.random.default_rng(30245109 + 12)

mse_eq36_K6, mse_evd_K6 = run_simulation(

    K=6, Nt=Nt, Nr=Nr, L=L,

    Pt_watt=Pt_watt,

    Noise_C_watt=Noise_C_watt,

    Noise_R_watt=Noise_R_watt,

    Gamma_dB_list=Gamma_dB_list,

    rng=rng6
)

mse_eq36_K12, mse_evd_K12 = run_simulation(

    K=12, Nt=Nt, Nr=Nr, L=L,

    Pt_watt=Pt_watt,

    Noise_C_watt=Noise_C_watt,

    Noise_R_watt=Noise_R_watt,

    Gamma_dB_list=Gamma_dB_list,

    rng=rng12

)

plt.figure(figsize=(6,4))

plt.plot(Gamma_dB_list, mse_evd_K12, 'm--x', linewidth=2,
         label='Eigenvalue Decomposition, K = 12')

plt.plot(Gamma_dB_list, mse_eq36_K12, 'k--', linewidth=2,
         label='Convex Relaxation Bound, K = 12')

plt.plot(Gamma_dB_list, mse_eq36_K12, 'ro', markerfacecolor='none',
         label='Proposed Rank-1 Optimum, K = 12')

plt.plot(Gamma_dB_list, mse_evd_K6, 'y-d', linewidth=2,
         markerfacecolor='none',
         label='Eigenvalue Decomposition, K = 6')

plt.plot(Gamma_dB_list, mse_eq36_K6, 'b:', linewidth=2,
         label='Convex Relaxation Bound, K = 6')

plt.plot(Gamma_dB_list, mse_eq36_K6, 'r^', markerfacecolor='none',
         label='Proposed Rank-1 Optimum, K = 6')

plt.xlabel("SINR (dB)")

plt.ylabel("MSE (dB)")

plt.grid(True)

plt.legend()

plt.xlim([2, 20])

plt.savefig("Figure06.png", dpi=600, bbox_inches="tight")
plt.show()